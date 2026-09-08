"""
Nethra AI - Autonomous Spatial Planner (threaded, cadence-decoupled)

    decode (native res) -> [every N frames] YOLOv8n + ByteTrack @ imgsz 416
                        -> [other frames]  dead-reckoned cached tracklets
    -> pinhole depth -> closing rate / evasion window -> free-space vector
    -> kinematic envelope -> plan (CRUISE / MONITOR / SHIFT / YIELD)
    -> swept path + BEV minimap drawn on the NATIVE frame -> JPEG -> queue

The UI thread only dequeues finished JPEG frames + telemetry.
"""

import math
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ENGINE_VERSION = "6.2"         # app.py checks this to guarantee both files are the matched pair

# ---------------- Perception ----------------
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.40
INFER_EVERY = 3                # run YOLO/ByteTrack on every Nth frame; others reuse cached tracklets
INFER_IMGSZ = 416              # YOLO letterboxes internally and returns boxes in native frame coords
ENHANCE_WIDTH = 960            # CLAHE runs on a copy this wide (boxes scaled back to native)
JPEG_QUALITY = 78
QUEUE_DEPTH = 2                # drop-oldest queue (standalone viewer / realtime=True)
BUFFER_FRAMES = 10             # FIFO presentation buffer (Streamlit / realtime=False): absorbs inference spikes
DECIMATION_MARGIN = 1.08       # decimate slightly ahead of measured load so the buffer never starves
FOCAL_LENGTH = 700.0           # px at REF_WIDTH, auto-scaled to the native frame width
REF_WIDTH = 1920.0
DEFAULT_WIDTH_M = 1.0
TARGET_CLASSES = {0: "PERSON", 1: "BICYCLE", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK", 16: "DOG", 19: "COW"}
REAL_WIDTHS_M = {0: 0.5, 1: 0.6, 2: 1.8, 3: 0.8, 5: 2.5, 7: 2.5, 16: 0.4, 19: 1.5}

# ---------------- Planning (unchanged math) ----------------
CRITICAL_DISTANCE = 15.0
WINDOW_MONITOR = 4.0
WINDOW_ACT = 2.5
HORIZON_Y = 0.46
D_NEAR = 2.5
CLEARANCE_M = 1.2
CAR_WIDTH_M = 1.85
ESCAPE_MARGIN_PX = 48          # at 720p; scaled with resolution
ESCAPE_SMOOTHING = 0.30
DIRECTION_HYSTERESIS = 0.15
BEZIER_SAMPLES = 48
MAX_STEER_DEG = 32.0
TRACK_HISTORY = 15

# ---------------- Vehicle / kinematics (unchanged) ----------------
G = 9.81
WHEELBASE_M = 2.8
ROLLOVER_LAT_G = 0.55
ROAD_MU_DRY = 0.70
ROAD_MU_WET = 0.45
PLANNING_MARGIN_M = 1.0

# ---------------- BEV minimap (base sizes at 720p; scaled) ----------------
BEV_W, BEV_H = 176, 236
BEV_PPM = 5.0
BEV_D_MAX = 42.0
BEV_MARGIN = 14

# Palette (BGR) - no red
WHITE = (255, 255, 255); GREY = (140, 150, 160); DIM = (70, 80, 92)
NEON_CYAN = (255, 230, 0); CYAN_SOFT = (255, 200, 90); CYAN_CORE = (255, 255, 210)
NEON_GREEN = (120, 255, 60); GREEN_CORE = (220, 255, 200)
AMBER = (0, 190, 255); AMBER_HI = (40, 215, 255)
BEV_BG = (16, 12, 8); BEV_GRID = (46, 40, 30)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, torch.get_num_threads()))   # let torch use whatever the host gives

# UI scale relative to 720p; set per frame by the planner (single worker thread writes it)
_S = 1.0


def sc(v):
    return int(round(v * _S))


def load_model(model_path: str = MODEL_PATH) -> YOLO:
    print(f"[Nethra] Loading {model_path} on {DEVICE.upper()} ...")
    model = YOLO(model_path)
    model.predict(np.zeros((360, 640, 3), np.uint8), imgsz=INFER_IMGSZ, device=DEVICE, verbose=False)   # warm-up
    return model


def reset_tracker(model):
    try:
        for t in getattr(getattr(model, "predictor", None), "trackers", []) or []:
            t.reset()
    except Exception:
        pass


# ---------------- Geometry ----------------
def focal_px(frame_w):
    return FOCAL_LENGTH * (frame_w / REF_WIDTH)


def estimate_distance(cls_id, pixel_width, frame_w):
    if pixel_width <= 0:
        return float("inf")
    return REAL_WIDTHS_M.get(cls_id, DEFAULT_WIDTH_M) * focal_px(frame_w) / pixel_width


def ground_distance_at_row(y, dh, d_near=D_NEAR):
    y_h = HORIZON_Y * dh
    return d_near * (dh - 1 - y_h) / max(y - y_h, 1.0)


def row_at_ground_distance(D, dh, d_near=D_NEAR):
    y_h = HORIZON_Y * dh
    return y_h + d_near * (dh - 1 - y_h) / max(D, 0.1)


def d_near_from_entity(dist_m, y_bottom, dh):
    y_h = HORIZON_Y * dh
    return dist_m * max(y_bottom - y_h, 1.0) / (dh - 1 - y_h)


def bev_dims():
    return sc(BEV_W), sc(BEV_H), BEV_PPM * _S


def bev_homography(dw, dh, d_near):
    f = focal_px(dw)
    bw, bh, ppm = bev_dims()
    src, dst = [], []
    for D in (d_near + 0.5, 40.0):
        y = row_at_ground_distance(D, dh, d_near)
        for X in (-5.0, 5.0):
            src.append([dw / 2 + X * f / D, y])
            dst.append([bw / 2 + X * ppm, bh - sc(12) - D * ppm])
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))


_BEV = {}


def bev_background():
    bw, bh, ppm = bev_dims()
    key = (bw, bh)
    if key in _BEV:
        return _BEV[key]
    bg = np.full((bh, bw, 3), BEV_BG, np.uint8)
    ego = (bw // 2, bh - sc(12))
    fs = 0.32 * _S
    for D in range(10, int(BEV_D_MAX) + 1, 10):
        yy = int(ego[1] - D * ppm)
        cv2.line(bg, (0, yy), (bw, yy), BEV_GRID, 1, cv2.LINE_AA)
        cv2.putText(bg, f"{D}m", (sc(4), yy - sc(3)), cv2.FONT_HERSHEY_SIMPLEX, fs, DIM, 1, cv2.LINE_AA)
    for X in range(-15, 16, 5):
        xx = int(ego[0] + X * ppm)
        cv2.line(bg, (xx, 0), (xx, bh), BEV_GRID, 1, cv2.LINE_AA)
    cone = np.array([ego, (int(ego[0] - 0.75 * bh), 0), (int(ego[0] + 0.75 * bh), 0)], np.int32)
    layer = bg.copy(); cv2.fillPoly(layer, [cone], (36, 30, 18)); cv2.addWeighted(layer, 0.6, bg, 0.4, 0, bg)
    _BEV[key] = (bg, ego)
    return _BEV[key]


def to_bev(H, pts):
    return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


# ---------------- Kinematics (unchanged) ----------------
def kinematic_check(speed_kmh, hazard_dist_m, lateral_offset_m, road_mu):
    v = max(speed_kmh / 3.6, 0.1)
    D = max(hazard_dist_m - PLANNING_MARGIN_M, 1.0)
    a_max = min(road_mu, ROLLOVER_LAT_G) * G
    a_req = 2.0 * lateral_offset_m * v * v / (D * D)
    return {"a_req_g": a_req / G, "a_max_g": a_max / G,
            "steer_req_deg": math.degrees(math.atan(WHEELBASE_M * a_req / (v * v))),
            "steer_max_deg": math.degrees(math.atan(WHEELBASE_M * a_max / (v * v))),
            "feasible": a_req <= a_max,
            "brake_dist_m": v * v / (2.0 * road_mu * G),
            "can_stop": v * v / (2.0 * road_mu * G) <= hazard_dist_m}


# ---------------- Drawing primitives ----------------
def _blend_roi(frame, paint, alpha, pts, pad):
    h, w = frame.shape[:2]
    x, y, bw, bh = cv2.boundingRect(np.asarray(pts, np.int32))
    x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + bw + pad, w), min(y + bh + pad, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    layer = roi.copy()
    paint(layer, np.array([x0, y0], dtype=np.int32))
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, roi)


def glow_polyline(frame, pts, color, core, widths=(18, 8, 3), alphas=(0.18, 0.4, 1.0), closed=False):
    pts = np.asarray(pts, np.int32).reshape(-1, 1, 2)
    widths = [max(1, sc(w)) for w in widths]
    for w, a in zip(widths[:-1], alphas[:-1]):
        _blend_roi(frame, lambda layer, off, w=w: cv2.polylines(layer, [pts - off], closed, color, w, cv2.LINE_AA), a, pts, pad=w)
    cv2.polylines(frame, [pts], closed, color, widths[-1], cv2.LINE_AA)
    if core is not None:
        cv2.polylines(frame, [pts], closed, core, 1, cv2.LINE_AA)


def bezier_curve(p0, p1, p2, n=BEZIER_SAMPLES):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2 = (np.asarray(p, dtype=np.float32) for p in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def draw_label(frame, x, y, text, color, scale=0.48):
    fs = scale * _S; th_ = max(1, sc(1))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th_)
    cv2.rectangle(frame, (x, y - th - sc(5)), (x + tw + sc(8), y + sc(3)), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + sc(4), y - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th_, cv2.LINE_AA)


def draw_panel(frame, x, y, w, h, alpha=0.55):
    band = frame[y:y + h, x:x + w]
    cv2.addWeighted(np.zeros_like(band), alpha, band, 1 - alpha, 0, band)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 70, 80), 1, cv2.LINE_AA)


# ---------------- Entities ----------------
def draw_entity(frame, x1, y1, x2, y2, label, tracked_only=True, primary=False):
    color = AMBER_HI if primary else AMBER
    h, w = frame.shape[:2]
    if not tracked_only:
        vp = np.array([w / 2, h * HORIZON_Y]); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d = np.array([vp[0] - cx, vp[1] - cy]); n = np.linalg.norm(d)
        depth = int(np.clip((x2 - x1) * 0.22, sc(8), sc(44)))
        off = (d / n * depth).astype(int) if n > 1 else np.array([0, -depth])
        front = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32); back = front + off
        cv2.polylines(frame, [back], True, (0, 90, 130), 1, cv2.LINE_AA)
        for f, b in zip(front, back):
            cv2.line(frame, tuple(f), tuple(b), (0, 90, 130), 1, cv2.LINE_AA)
        if primary:
            glow_polyline(frame, front, color, None, widths=(12, 2), alphas=(0.28, 1.0), closed=True)
        else:
            cv2.polylines(frame, [front], True, color, 1, cv2.LINE_AA)
    c = max(sc(6), min(sc(18), (x2 - x1) // 5)); t = max(1, sc(2))
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (px, py), (px + dx * c, py), color, t, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * c), color, t, cv2.LINE_AA)
    draw_label(frame, x1, y1 - sc(5) if y1 > sc(24) else y2 + sc(18), label, color)


# ---------------- Swept path ----------------
def swept_path_polygon(curve, dw, dh, d_near):
    f = focal_px(dw)
    half = np.array([CAR_WIDTH_M / 2 * f / ground_distance_at_row(y, dh, d_near) for (_, y) in curve], np.float32)
    half = np.clip(half, 3, dw * 0.35)
    left = curve.copy(); right = curve.copy()
    left[:, 0] -= half; right[:, 0] += half
    return np.vstack([left, right[::-1]]).astype(np.int32), left.astype(np.int32), right.astype(np.int32)


def draw_swept_path(frame, curve, mode, d_near):
    dh, dw = frame.shape[:2]
    poly, left, right = swept_path_polygon(curve, dw, dh, d_near)
    if mode == "shift":
        fill, edge, core, fa = NEON_GREEN, NEON_GREEN, GREEN_CORE, 0.22
    elif mode == "yield":
        fill, edge, core, fa = AMBER, AMBER_HI, WHITE, 0.16
    else:
        fill, edge, core, fa = CYAN_SOFT, NEON_CYAN, CYAN_CORE, 0.12
    _blend_roi(frame, lambda layer, off: cv2.fillPoly(layer, [poly - off], fill), fa, poly, pad=2)
    for edge_pts in (left, right):
        glow_polyline(frame, edge_pts, edge, core, widths=(14, 6, 2), alphas=(0.16, 0.36, 1.0))
    ex, ey = int(curve[-1][0]), int(curve[-1][1])
    if mode == "yield":
        bar = np.array([[left[-1][0], ey], [right[-1][0], ey]], np.int32)
        glow_polyline(frame, bar, AMBER_HI, WHITE, widths=(18, 8, 4), alphas=(0.22, 0.45, 1.0))
    elif mode == "shift":
        cv2.circle(frame, (ex, ey), sc(10), NEON_GREEN, max(1, sc(2)), cv2.LINE_AA)
        cv2.circle(frame, (ex, ey), sc(3), WHITE, -1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (ex, ey), sc(5), NEON_CYAN, 1, cv2.LINE_AA)


# ---------------- BEV minimap ----------------
def draw_minimap(frame, curve, mode, entities, primary_idx, d_near):
    dh, dw = frame.shape[:2]
    bw, bh, ppm = bev_dims()
    H = bev_homography(dw, dh, d_near)
    bg, ego = bev_background()
    mm = bg.copy()

    poly, left, right = swept_path_polygon(curve, dw, dh, d_near)
    bl, br = to_bev(H, left), to_bev(H, right)
    ok = lambda p: (p[:, 1] > -50) & (p[:, 1] < bh + 50)
    keep = ok(bl) & ok(br)
    if keep.sum() >= 2:
        bl, br = bl[keep], br[keep]
        bpoly = np.vstack([bl, br[::-1]]).astype(np.int32)
        col = NEON_GREEN if mode == "shift" else AMBER if mode == "yield" else NEON_CYAN
        layer = mm.copy(); cv2.fillPoly(layer, [bpoly], col); cv2.addWeighted(layer, 0.28, mm, 0.72, 0, mm)
        cv2.polylines(mm, [bl.astype(np.int32)], False, col, 1, cv2.LINE_AA)
        cv2.polylines(mm, [br.astype(np.int32)], False, col, 1, cv2.LINE_AA)
        if mode == "yield":
            cv2.line(mm, tuple(bl[-1].astype(int)), tuple(br[-1].astype(int)), AMBER_HI, max(1, sc(2)), cv2.LINE_AA)

    for i, e in enumerate(entities):
        x1, y1, x2, y2 = e["box"]
        gx, gy = to_bev(H, [[(x1 + x2) / 2, y2]])[0]
        if not (0 <= gx < bw and 0 <= gy < bh):
            continue
        wpx = max(sc(3), int(REAL_WIDTHS_M.get(e["cls_id"], 1.0) * ppm / 2))
        col = AMBER_HI if i == primary_idx else AMBER if e["hazard"] else GREY
        cv2.rectangle(mm, (int(gx) - wpx, int(gy) - sc(3)), (int(gx) + wpx, int(gy) + sc(3)), col, -1, cv2.LINE_AA)
        if i == primary_idx:
            cv2.circle(mm, (int(gx), int(gy)), sc(7), AMBER_HI, 1, cv2.LINE_AA)
        cv2.putText(mm, f"{e['distance_m']:.0f}", (int(gx) + wpx + sc(3), int(gy) + sc(3)), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * _S, col, 1, cv2.LINE_AA)

    ew, el = int(CAR_WIDTH_M * ppm / 2), int(4.5 * ppm / 2)
    cv2.rectangle(mm, (ego[0] - ew, ego[1] - el), (ego[0] + ew, ego[1] + el), NEON_CYAN, -1, cv2.LINE_AA)
    cv2.rectangle(mm, (ego[0] - ew, ego[1] - el), (ego[0] + ew, ego[1] + el), WHITE, 1, cv2.LINE_AA)
    cv2.putText(mm, "BEV  SPATIAL MAP", (sc(6), sc(12)), cv2.FONT_HERSHEY_SIMPLEX, 0.34 * _S, GREY, 1, cv2.LINE_AA)

    x0, y0 = dw - bw - sc(BEV_MARGIN), dh - bh - sc(BEV_MARGIN)
    roi = frame[y0:y0 + bh, x0:x0 + bw]
    cv2.addWeighted(mm, 0.88, roi, 0.12, 0, roi)
    cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + bw, y0 + bh), (70, 80, 92), 1, cv2.LINE_AA)


# ---------------- HUD ----------------
STATE_COLOR = {"CRUISING": NEON_CYAN, "MONITORING": AMBER, "TRAJECTORY SHIFT": NEON_GREEN, "SAFE YIELD": AMBER_HI}


def draw_hud(frame, tel):
    h, w = frame.shape[:2]
    color = STATE_COLOR.get(tel["state"], NEON_CYAN)
    t1 = max(1, sc(1))
    band = frame[0:sc(30), 0:w]; cv2.addWeighted(np.zeros_like(band), 0.45, band, 0.55, 0, band)
    win = f"{tel['window_s']:.1f}s" if tel["window_s"] is not None else "--"
    cv2.putText(frame, f"NETHRA AI  |  {tel['state']}  |  {tel['speed_kmh']:.0f} km/h  |  WINDOW {win}  |  "
                       f"ENTITIES {tel['entities']}  |  INF {tel['infer_ms']:.0f}ms /{tel['infer_every']}f",
                (sc(12), sc(21)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * _S, color, t1, cv2.LINE_AA)

    k = tel["kinematics"]
    px, py, pw, ph = sc(12), h - sc(124), sc(292), sc(112)
    draw_panel(frame, px, py, pw, ph)
    fs = 0.4 * _S
    cv2.putText(frame, "KINEMATIC ENVELOPE", (px + sc(10), py + sc(17)), cv2.FONT_HERSHEY_SIMPLEX, fs, GREY, t1, cv2.LINE_AA)
    rows = [("LATERAL DEMAND", f"{k['a_req_g']:.2f} / {k['a_max_g']:.2f} g" if k else "--"),
            ("STEER  req / max", f"{k['steer_req_deg']:.1f} / {k['steer_max_deg']:.1f} deg" if k else "--"),
            ("STOP DISTANCE", f"{k['brake_dist_m']:.1f} m" if k else "--"),
            ("ROAD MU", f"{tel['road_mu']:.2f}")]
    for i, (lab, val) in enumerate(rows):
        yy = py + sc(38) + i * sc(17)
        cv2.putText(frame, lab, (px + sc(10), yy), cv2.FONT_HERSHEY_SIMPLEX, fs, GREY, t1, cv2.LINE_AA)
        vcol = AMBER_HI if (k and i == 0 and not k["feasible"]) else WHITE
        cv2.putText(frame, val, (px + sc(148), yy), cv2.FONT_HERSHEY_SIMPLEX, fs, vcol, t1, cv2.LINE_AA)
    if k:
        bx, by, bw_ = px + sc(10), py + ph - sc(9), pw - sc(20); t3 = max(2, sc(3))
        cv2.line(frame, (bx, by), (bx + bw_, by), (60, 70, 80), t3)
        cv2.line(frame, (bx, by), (bx + int(bw_ * min(k["a_req_g"], 1.0)), by), AMBER_HI if not k["feasible"] else NEON_GREEN, t3)
        lim = bx + int(bw_ * min(k["a_max_g"], 1.0)); cv2.line(frame, (lim, by - sc(5)), (lim, by + sc(5)), AMBER, max(1, sc(2)))

    if tel["state"] == "SAFE YIELD":
        txt = "SAFE YIELD  -  CONTROLLED DECELERATION"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.8 * _S, t1)
        x, y = (w - tw) // 2, sc(70)
        draw_panel(frame, x - sc(18), y - th - sc(14), tw + sc(36), th + sc(26), alpha=0.6)
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.8 * _S, AMBER_HI, t1, cv2.LINE_AA)

    steer = tel["steer_deg"]
    tick_x = int(w / 2 + (steer / MAX_STEER_DEG) * (w * 0.18))
    cv2.line(frame, (w // 2 - int(w * 0.18), h - sc(4)), (w // 2 + int(w * 0.18), h - sc(4)), GREY, 1, cv2.LINE_AA)
    cv2.line(frame, (tick_x, h - sc(12)), (tick_x, h - 1), color, max(2, sc(3)), cv2.LINE_AA)


# ---------------- Free space (unchanged) ----------------
def free_space_vector(hazard, others, frame_w):
    x1, y1, x2, y2 = hazard
    left_limit, right_limit = 0, frame_w
    for (ox1, oy1, ox2, oy2) in others:
        if oy2 < y1 or oy1 > y2:
            continue
        if ox2 <= x1:
            left_limit = max(left_limit, ox2)
        elif ox1 >= x2:
            right_limit = min(right_limit, ox1)
    return max(x1 - left_limit, 0), max(right_limit - x2, 0)


def steering_angle_deg(frame_w, frame_h, target):
    dx = target[0] - frame_w / 2
    dy = max((frame_h - sc(8)) - target[1], 1)
    return float(np.clip(math.degrees(math.atan2(dx, dy)), -MAX_STEER_DEG, MAX_STEER_DEG))


# ---------------- Planner ----------------
class SpatialPlanner:
    """Perception runs every INFER_EVERY frames; in between, tracklets are dead-reckoned from their
    last two observations so boxes glide instead of stepping. All planning math runs every frame."""

    def __init__(self, model):
        self.model = model
        self._target_ema = None
        self._last_dir = None
        self._tracks = {}                 # id -> deque[(t, dist)] for closing rate
        self._motion = {}                 # id -> (t_prev, box_prev, t_cur, box_cur) for dead reckoning
        self._cache = []                  # last raw detections: [(box_xyxy np.float32, cls_id, tid)]
        self._cache_t = 0.0
        self.d_near = D_NEAR
        self.last_infer_ms = 0.0
        self.frame_counter = 0

    # ---- perception ----
    def _infer(self, frame_bgr, conf, enhance, imgsz):
        h, w = frame_bgr.shape[:2]
        src, scale = frame_bgr, 1.0
        if enhance:                                    # CLAHE on a reduced copy, boxes scaled back
            scale = w / ENHANCE_WIDTH
            small = cv2.resize(frame_bgr, (ENHANCE_WIDTH, int(h / scale)), interpolation=cv2.INTER_AREA)
            lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
            lab[..., 0] = cv2.createCLAHE(2.5, (8, 8)).apply(lab[..., 0])
            src = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        t_inf = time.perf_counter()
        res = self.model.track(src, imgsz=imgsz, conf=conf, classes=list(TARGET_CLASSES.keys()), persist=True,
                               tracker="bytetrack.yaml", device=DEVICE, half=(DEVICE == "cuda"), verbose=False)[0]
        self.last_infer_ms = (time.perf_counter() - t_inf) * 1000.0
        out = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy() * scale
            clss = res.boxes.cls.cpu().numpy().astype(int)
            ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [-1] * len(xyxy)
            out = [(b.astype(np.float32), int(c), int(i)) for b, c, i in zip(xyxy, clss, ids)]
        return out

    def _update_motion(self, dets, now):
        live = set()
        for box, _, tid in dets:
            if tid < 0:
                continue
            live.add(tid)
            prev = self._motion.get(tid)
            self._motion[tid] = (prev[2], prev[3], now, box) if prev else (now, box, now, box)
        for tid in list(self._motion):
            if tid not in live:
                del self._motion[tid]

    def _dead_reckon(self, now):
        """Extrapolate each cached box along its observed velocity (capped) for intermediate frames."""
        out = []
        for box, cls_id, tid in self._cache:
            m = self._motion.get(tid) if tid >= 0 else None
            if m and m[2] > m[0]:
                vel = (m[3] - m[1]) / (m[2] - m[0])                       # px/s per corner
                dt = min(now - m[2], 0.25)                                # never extrapolate > 250 ms
                box = box + vel * dt
            out.append((box, cls_id, tid))
        return out

    # ---- per frame ----
    def process(self, frame_bgr, speed_kmh, road_mu, critical_distance=CRITICAL_DISTANCE, conf=CONF_THRESHOLD,
                enhance=False, infer_every=INFER_EVERY, imgsz=INFER_IMGSZ, render_max_height=0):
        global _S
        t0 = now = time.perf_counter()
        if render_max_height and frame_bgr.shape[0] > render_max_height:
            # Optional cost cap (Cloud Lite): downscale the source ONCE, up front. Perception, depth, planning
            # and overlays all run in this frame's own coordinates, so nothing downstream changes.
            scale = render_max_height / frame_bgr.shape[0]
            frame_bgr = cv2.resize(frame_bgr, (int(frame_bgr.shape[1] * scale), render_max_height), interpolation=cv2.INTER_LINEAR)
        dh, dw = frame_bgr.shape[:2]
        _S = dh / 720.0
        canvas = frame_bgr                      # draw on the NATIVE frame - no downscale
        self.frame_counter += 1
        inferred = (self.frame_counter % max(1, infer_every) == 1) or not self._cache
        if inferred:
            self._cache = self._infer(frame_bgr, conf, enhance, imgsz)
            self._cache_t = now
            self._update_motion(self._cache, now)
            dets = self._cache
        else:
            dets = self._dead_reckon(now)
        v_ego = speed_kmh / 3.6

        entities, live = [], set()
        for box, cls_id, tid in dets:
            bx1, by1, bx2, by2 = box
            dist = float(estimate_distance(cls_id, float(bx2 - bx1), dw))
            if tid >= 0:
                live.add(tid)
                hist = self._tracks.setdefault(tid, deque(maxlen=TRACK_HISTORY))
                if inferred:                                              # one sample per observation
                    hist.append((now, dist))
                closing = 0.0
                if len(hist) >= 4 and hist[-1][0] - hist[0][0] >= 0.2:
                    ts = np.array([p[0] for p in hist]); ds = np.array([p[1] for p in hist])
                    closing = float(max(-np.polyfit(ts - ts[0], ds, 1)[0], 0.0))
            else:
                closing = 0.0
            v_close = v_ego + closing
            window = dist / v_close if v_close > 0.05 else None
            entities.append({"id": tid, "cls_id": cls_id, "class": TARGET_CLASSES.get(cls_id, "OBJECT"),
                             "distance_m": round(dist, 1), "closing_mps": round(closing, 2),
                             "window_s": None if window is None else round(window, 2),
                             "hazard": dist < critical_distance or (window is not None and window < WINDOW_MONITOR),
                             "mpp": REAL_WIDTHS_M.get(cls_id, DEFAULT_WIDTH_M) / max(float(bx2 - bx1), 1.0),
                             "box": (int(bx1), int(by1), int(bx2), int(by2))})
        for tid in list(self._tracks):
            if tid not in live:
                del self._tracks[tid]

        ests = [d_near_from_entity(e["distance_m"], e["box"][3], dh) for e in entities
                if (e["box"][2] - e["box"][0]) >= sc(24) and e["box"][3] > HORIZON_Y * dh + sc(20)]
        if ests:
            self.d_near += 0.15 * (float(np.clip(np.median(ests), 1.5, 14.0)) - self.d_near)

        hazards = sorted((e for e in entities if e["hazard"]),
                         key=lambda e: (e["window_s"] if e["window_s"] is not None else 1e9, e["distance_m"]))
        primary = hazards[0] if hazards else None
        primary_idx = entities.index(primary) if primary else -1

        # ---- Plan (math unchanged) ----
        state, mode, direction, reason = "CRUISING", "cruise", "NONE", None
        free_l = free_r = None; free_l_m = free_r_m = None; kin = None; lateral_signed = 0.0
        raw = np.array([dw // 2, int(dh * (HORIZON_Y + 0.10))], np.float32)
        window_primary = primary["window_s"] if primary else None
        margin_px = sc(ESCAPE_MARGIN_PX)

        if primary:
            state = "MONITORING"
            act = primary["distance_m"] < critical_distance or (window_primary is not None and window_primary < WINDOW_ACT)
            if act:
                others = [e["box"] for e in entities if e is not primary]
                free_l, free_r = free_space_vector(primary["box"], others, dw)
                free_l_m, free_r_m = round(free_l * primary["mpp"], 1), round(free_r * primary["mpp"], 1)
                x1, y1, x2, y2 = primary["box"]
                clearance_px = max(margin_px, int(CLEARANCE_M / primary["mpp"]))
                cands = []
                if free_r >= clearance_px: cands.append(int(np.clip(x2 + clearance_px, 16, dw - 16)))
                if free_l >= clearance_px: cands.append(int(np.clip(x1 - clearance_px, 16, dw - 16)))
                cands = [("RIGHT" if c >= dw / 2 else "LEFT", c) for c in cands]
                cands.sort(key=lambda c: abs(c[1] - dw / 2))
                if len(cands) == 2 and self._last_dir and cands[1][0] == self._last_dir and \
                        abs(abs(cands[1][1] - dw / 2) - abs(cands[0][1] - dw / 2)) < DIRECTION_HYSTERESIS * max(abs(cands[0][1] - dw / 2), 1):
                    cands.reverse()
                if cands:
                    direction, ex = cands[0]
                else:
                    direction, ex = "NONE", dw // 2
                lateral_signed = (ex - dw / 2) * primary["mpp"]
                kin = kinematic_check(speed_kmh, primary["distance_m"], abs(lateral_signed), road_mu)
                kin["lateral_m"] = round(abs(lateral_signed), 2)
                if not cands:
                    state, mode, reason = "SAFE YIELD", "yield", "NO_FREE_SPACE"
                elif not kin["feasible"]:
                    state, mode, reason = "SAFE YIELD", "yield", "KINEMATIC"
                else:
                    state, mode = "TRAJECTORY SHIFT", "shift"; self._last_dir = direction
                raw = np.array([ex if mode == "shift" else dw // 2, y2], np.float32)
        if mode != "shift":
            self._last_dir = None

        self._target_ema = raw if self._target_ema is None else self._target_ema + ESCAPE_SMOOTHING * (raw - self._target_ema)
        target = (float(self._target_ema[0]), float(self._target_ema[1]))
        steer = steering_angle_deg(dw, dh, target) if mode == "shift" else 0.0

        # ---- Render on native frame ----
        start = (dw / 2, dh - sc(8))
        control = (dw / 2, target[1] + (start[1] - target[1]) / 3)
        curve = bezier_curve(start, control, target)
        draw_swept_path(canvas, curve, mode, self.d_near)
        for i, e in enumerate(entities):
            wtxt = f"  {e['window_s']:.1f}s" if e["window_s"] is not None else ""
            draw_entity(canvas, *e["box"], f"{e['class']}  {e['distance_m']:.1f}m{wtxt}",
                        tracked_only=not e["hazard"], primary=(i == primary_idx and mode != "cruise"))
        draw_minimap(canvas, curve, mode, entities, primary_idx, self.d_near)

        avail = None if free_l_m is None else round(min(max(free_l_m, free_r_m) / (CLEARANCE_M * 2.0), 1.0), 2)
        tel = {"state": state, "mode": mode, "reason": reason, "direction": direction,
               "lateral_m": round(lateral_signed, 2), "steer_deg": round(steer, 1),
               "speed_kmh": float(speed_kmh), "road_mu": float(road_mu),
               "entities": len(entities), "hazards": len(hazards), "window_s": window_primary,
               "nearest_m": min((e["distance_m"] for e in entities), default=None),
               "primary": ({"class": primary["class"], "distance_m": primary["distance_m"], "window_s": primary["window_s"]} if primary else None),
               "free_left_m": free_l_m, "free_right_m": free_r_m, "free_availability": avail, "kinematics": kin,
               "ground_plane_near_m": round(self.d_near, 2),
               "inferred": inferred, "infer_every": max(1, infer_every), "imgsz": imgsz,
               "infer_ms": round(self.last_infer_ms, 1), "pipeline_ms": 0.0, "resolution": f"{dw}x{dh}",
               "detections": [{k: v for k, v in e.items() if k not in ("box", "mpp", "cls_id")} for e in entities]}
        draw_hud(canvas, tel)
        tel["pipeline_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return canvas, tel


# ---------------- Threaded worker ----------------
class VideoInferenceWorker(threading.Thread):
    """
    Background planner. Two delivery modes:
      realtime=True  : paces itself to the source clock, drop-oldest queue (standalone OpenCV viewer).
      realtime=False : runs ahead into a FIFO presentation buffer; the consumer paces display at source FPS.
                       Inference spikes are absorbed by the buffer, so playback stays smooth. If average
                       throughput falls below source FPS, frames are decimated adaptively so timing holds.
    Hot-tunable: speed_kmh, road_mu, critical_distance, conf, loop, enhance, infer_every, imgsz.
    """

    def __init__(self, video_path, model, speed_kmh=45.0, road_mu=ROAD_MU_DRY, critical_distance=CRITICAL_DISTANCE,
                 conf=CONF_THRESHOLD, loop=True, enhance=False, infer_every=INFER_EVERY, imgsz=INFER_IMGSZ,
                 render_max_height=0, frame_stride=1, jpeg_quality=JPEG_QUALITY,
                 encode_jpeg=True, realtime=False):
        super().__init__(daemon=True, name="NethraSpatialPlanner")
        self.video_path, self.model = video_path, model
        self.planner = SpatialPlanner(model)
        self.realtime = realtime
        self.out_queue = queue.Queue(maxsize=QUEUE_DEPTH if realtime else BUFFER_FRAMES)
        self.speed_kmh, self.road_mu = speed_kmh, road_mu
        self.critical_distance, self.conf = critical_distance, conf
        self.loop, self.enhance, self.infer_every, self.imgsz = loop, enhance, infer_every, imgsz
        self.render_max_height, self.frame_stride, self.jpeg_quality = render_max_height, max(1, frame_stride), jpeg_quality
        self.encode_jpeg = encode_jpeg
        self._stop_evt, self._pause_evt, self._seek_to = threading.Event(), threading.Event(), None
        self.frame_idx, self.source_fps, self.error = 0, 30.0, None
        self.stats = {"fps": 0.0, "dropped": 0, "skipped": 0, "buffer": 0}

    def stop(self): self._stop_evt.set()
    def pause(self, flag): self._pause_evt.set() if flag else self._pause_evt.clear()
    def seek(self, idx): self._seek_to = int(idx)

    @property
    def running(self): return self.is_alive() and not self._stop_evt.is_set()

    def latest(self, timeout=1.0):
        """Newest frame, draining any backlog (realtime mode)."""
        try:
            item = self.out_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        while True:
            try: item = self.out_queue.get_nowait()
            except queue.Empty: return item

    def next_frame(self, timeout=1.0):
        """Next frame in order (buffered mode). Consumer paces display."""
        try:
            return self.out_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def primed(self, min_frames=5, timeout=4.0):
        """Wait until the presentation buffer has a few frames so playback starts smooth."""
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end and self.running:
            if self.out_queue.qsize() >= min_frames:
                return True
            time.sleep(0.02)
        return self.out_queue.qsize() > 0

    def _publish(self, item):
        if self.realtime:                                    # drop-oldest
            while True:
                try:
                    self.out_queue.put_nowait(item); return
                except queue.Full:
                    try: self.out_queue.get_nowait(); self.stats["dropped"] += 1
                    except queue.Empty: pass
        while not self._stop_evt.is_set():                   # FIFO, block while the buffer is full
            try:
                self.out_queue.put(item, timeout=0.2); return
            except queue.Full:
                continue

    def _rewind(self, cap):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        reset_tracker(self.model)
        self.planner._cache = []; self.planner._motion.clear(); self.planner._tracks.clear()

    def run(self):
        reset_tracker(self.model)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error = f"Could not open video: {self.video_path}"; return
        self.source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        period = 1.0 / self.source_fps
        fps_ema, proc_ema, skip_acc, deadline = None, None, 0.0, time.perf_counter()
        advanced = 1                                         # source frames represented by the next published frame
        try:
            while not self._stop_evt.is_set():
                if self._pause_evt.is_set():
                    time.sleep(0.05); deadline = time.perf_counter(); continue
                if self._seek_to is not None:
                    self._rewind(cap); cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_to); self._seek_to = None
                    deadline = time.perf_counter()
                t_frame = time.perf_counter()
                ok, frame = cap.read()
                if not ok:                                   # seamless loop: rewind + fresh tracker state
                    if self.loop:
                        self._rewind(cap); deadline = time.perf_counter(); continue
                    break
                self.frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                stride_extra = 0
                for _ in range(max(1, int(self.frame_stride)) - 1):     # source stride: skip decode work entirely
                    if cap.grab(): stride_extra += 1
                canvas, tel = self.planner.process(frame, self.speed_kmh, self.road_mu, critical_distance=self.critical_distance,
                                                   conf=self.conf, enhance=self.enhance,
                                                   infer_every=self.infer_every, imgsz=self.imgsz,
                                                   render_max_height=self.render_max_height)
                if self.encode_jpeg:
                    ok_enc, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)])
                    payload = buf.tobytes() if ok_enc else None
                else:
                    payload = canvas
                dt = max(time.perf_counter() - t_frame, 1e-6)
                proc_ema = dt if proc_ema is None else 0.85 * proc_ema + 0.15 * dt
                fps_ema = (1 / dt) if fps_ema is None else 0.9 * fps_ema + 0.1 / dt
                self.stats["fps"] = round(fps_ema, 1); self.stats["buffer"] = self.out_queue.qsize()
                tel.update({"fps": self.stats["fps"], "frame_idx": self.frame_idx, "source_fps": round(self.source_fps, 1),
                            "period_s": period, "advanced": advanced + stride_extra,
                            "dropped": self.stats["dropped"], "skipped": self.stats["skipped"],
                            "buffer": self.stats["buffer"],
                            "jpeg_kb": round(len(payload) / 1024, 1) if isinstance(payload, (bytes, bytearray)) else None})
                self._publish((payload, tel))

                if self.realtime:                            # pace to the source clock; grab-skip if behind
                    deadline += period
                    lag = time.perf_counter() - deadline
                    if lag < 0:
                        time.sleep(-lag)
                    else:
                        for _ in range(min(int(lag / period), 10)):
                            if cap.grab(): self.stats["skipped"] += 1
                        deadline = time.perf_counter()
                else:                                        # buffered: decimate only if average throughput < source FPS
                    skip_acc += max(proc_ema * DECIMATION_MARGIN / period - 1.0, 0.0)
                    advanced = 1
                    while skip_acc >= 1.0:
                        if cap.grab():
                            self.stats["skipped"] += 1; advanced += 1
                        skip_acc -= 1.0
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            cap.release()


def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "city.mp4"
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
    worker = VideoInferenceWorker(path, load_model(), speed_kmh=speed, encode_jpeg=False, loop=False, realtime=True)
    worker.start()
    print("[Nethra] 'q' quits, SPACE pauses, +/- adjusts speed.")
    paused = False
    while worker.running or not worker.out_queue.empty():
        item = worker.latest(timeout=0.5)
        if item is not None:
            cv2.imshow("Nethra AI - Spatial Planner", item[0])
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        if key == ord(" "): paused = not paused; worker.pause(paused)
        if key in (ord("+"), ord("=")): worker.speed_kmh += 5
        if key == ord("-"): worker.speed_kmh = max(0, worker.speed_kmh - 5)
    worker.stop()
    if worker.error: print("[Nethra] Worker error:", worker.error)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()