"""
Nethra AI - Autonomous Spatial Planner (threaded)

    decode -> 640x360 -> YOLOv8n + ByteTrack -> pinhole depth -> closing rate / evasion window
           -> free-space vector -> kinematic envelope -> plan (CRUISE / MONITOR / SHIFT / YIELD)
           -> swept-path footprint + BEV minimap render -> JPEG -> queue

Everything above runs inside VideoInferenceWorker; the UI thread only dequeues results.
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

ENGINE_VERSION = "6.0"         # app.py checks this to guarantee both files are the matched pair

# ---------------- Perception ----------------
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.40
INFER_SIZE = (640, 360)
DISPLAY_SIZE = (1280, 720)
JPEG_QUALITY = 82
QUEUE_DEPTH = 2
FOCAL_LENGTH = 700.0           # px at REF_WIDTH, auto-scaled
REF_WIDTH = 1920.0
DEFAULT_WIDTH_M = 1.0
TARGET_CLASSES = {0: "PERSON", 1: "BICYCLE", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK", 16: "DOG", 19: "COW"}
REAL_WIDTHS_M = {0: 0.5, 1: 0.6, 2: 1.8, 3: 0.8, 5: 2.5, 7: 2.5, 16: 0.4, 19: 1.5}

# ---------------- Planning ----------------
CRITICAL_DISTANCE = 15.0
WINDOW_MONITOR = 4.0           # s  evasion window below which we MONITOR
WINDOW_ACT = 2.5               # s  evasion window below which we act (SHIFT or YIELD)
HORIZON_Y = 0.46               # vanishing-point row (fraction of frame height)
D_NEAR = 2.5                   # m  ground distance at the bottom row of the frame
CLEARANCE_M = 1.2              # m  lateral clearance kept from the hazard's edge
CAR_WIDTH_M = 1.85
ESCAPE_MARGIN_PX = 48
ESCAPE_SMOOTHING = 0.30
DIRECTION_HYSTERESIS = 0.15
BEZIER_SAMPLES = 48
MAX_STEER_DEG = 32.0
TRACK_HISTORY = 15

# ---------------- Vehicle / kinematics (SUV-class) ----------------
G = 9.81
WHEELBASE_M = 2.8
ROLLOVER_LAT_G = 0.55
ROAD_MU_DRY = 0.70
ROAD_MU_WET = 0.45
PLANNING_MARGIN_M = 1.0

# ---------------- BEV minimap ----------------
BEV_W, BEV_H = 176, 236        # px
BEV_PPM = 5.0                  # px per metre (uniform)
BEV_D_MAX = 42.0               # m forward range
BEV_MARGIN = 14                # px inset from frame corner

# Palette (BGR) - no red
WHITE = (255, 255, 255); GREY = (140, 150, 160); DIM = (70, 80, 92)
NEON_CYAN = (255, 230, 0); CYAN_SOFT = (255, 200, 90); CYAN_CORE = (255, 255, 210)
NEON_GREEN = (120, 255, 60); GREEN_CORE = (220, 255, 200)
AMBER = (0, 190, 255); AMBER_HI = (40, 215, 255); ORANGE = (0, 150, 255)
BEV_BG = (16, 12, 8); BEV_GRID = (46, 40, 30)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: str = MODEL_PATH) -> YOLO:
    print(f"[Nethra] Loading {model_path} on {DEVICE.upper()} ...")
    model = YOLO(model_path)
    model.predict(np.zeros((INFER_SIZE[1], INFER_SIZE[0], 3), np.uint8),
                  imgsz=640, device=DEVICE, half=(DEVICE == "cuda"), verbose=False)
    return model


def reset_tracker(model):
    try:
        for t in getattr(getattr(model, "predictor", None), "trackers", []) or []:
            t.reset()
    except Exception:
        pass


# ---------------- Geometry ----------------
def f_display():
    return FOCAL_LENGTH * (DISPLAY_SIZE[0] / REF_WIDTH)


def estimate_distance(cls_id, pixel_width, infer_w):
    if pixel_width <= 0:
        return float("inf")
    return (REAL_WIDTHS_M.get(cls_id, DEFAULT_WIDTH_M) * FOCAL_LENGTH * (infer_w / REF_WIDTH)) / pixel_width


def ground_distance_at_row(y, dh, d_near=D_NEAR):
    """Flat-road pinhole ground plane: D(y) = d_near * (y_bottom - y_h) / (y - y_h)."""
    y_h = HORIZON_Y * dh
    return d_near * (dh - 1 - y_h) / max(y - y_h, 1.0)


def row_at_ground_distance(D, dh, d_near=D_NEAR):
    y_h = HORIZON_Y * dh
    return y_h + d_near * (dh - 1 - y_h) / max(D, 0.1)


def d_near_from_entity(dist_m, y_bottom, dh):
    """Invert the ground plane: the near-distance constant implied by one entity's width-depth and its
    ground-contact row. Averaged over entities this self-calibrates the ground plane every frame."""
    y_h = HORIZON_Y * dh
    return dist_m * max(y_bottom - y_h, 1.0) / (dh - 1 - y_h)


def bev_homography(dw, dh, d_near):
    """Image -> BEV homography from the same pinhole ground plane used for depth: four ground points at
    two ranges and lateral +/-5 m are projected into the image, then mapped to metric minimap px."""
    f = f_display()
    src, dst = [], []
    for D in (d_near + 0.5, 40.0):
        y = row_at_ground_distance(D, dh, d_near)
        for X in (-5.0, 5.0):
            src.append([dw / 2 + X * f / D, y])
            dst.append([BEV_W / 2 + X * BEV_PPM, BEV_H - 12 - D * BEV_PPM])
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))


_BEV = {}


def bev_background():
    """Static minimap card: range lines every 10 m, lateral grid, heading cone. Cached."""
    if "bg" in _BEV:
        return _BEV["bg"]
    bg = np.full((BEV_H, BEV_W, 3), BEV_BG, np.uint8)
    ego = (BEV_W // 2, BEV_H - 12)
    for D in range(10, int(BEV_D_MAX) + 1, 10):
        yy = int(ego[1] - D * BEV_PPM)
        cv2.line(bg, (0, yy), (BEV_W, yy), BEV_GRID, 1, cv2.LINE_AA)
        cv2.putText(bg, f"{D}m", (4, yy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, DIM, 1, cv2.LINE_AA)
    for X in range(-15, 16, 5):
        xx = int(ego[0] + X * BEV_PPM)
        cv2.line(bg, (xx, 0), (xx, BEV_H), BEV_GRID, 1, cv2.LINE_AA)
    cone = np.array([ego, (int(ego[0] - 0.75 * BEV_H), 0), (int(ego[0] + 0.75 * BEV_H), 0)], np.int32)
    layer = bg.copy(); cv2.fillPoly(layer, [cone], (36, 30, 18)); cv2.addWeighted(layer, 0.6, bg, 0.4, 0, bg)
    _BEV["bg"] = (bg, ego)
    return _BEV["bg"]


def to_bev(H, pts):
    p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


# ---------------- Kinematics ----------------
def kinematic_check(speed_kmh, hazard_dist_m, lateral_offset_m, road_mu):
    v = max(speed_kmh / 3.6, 0.1)
    D = max(hazard_dist_m - PLANNING_MARGIN_M, 1.0)
    a_max = min(road_mu, ROLLOVER_LAT_G) * G
    a_req = 2.0 * lateral_offset_m * v * v / (D * D)
    return {
        "a_req_g": a_req / G, "a_max_g": a_max / G,
        "steer_req_deg": math.degrees(math.atan(WHEELBASE_M * a_req / (v * v))),
        "steer_max_deg": math.degrees(math.atan(WHEELBASE_M * a_max / (v * v))),
        "feasible": a_req <= a_max,
        "brake_dist_m": v * v / (2.0 * road_mu * G),
        "can_stop": v * v / (2.0 * road_mu * G) <= hazard_dist_m,
    }


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
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(frame, (x, y - th - 5), (x + tw + 8, y + 3), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 4, y - 1), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_panel(frame, x, y, w, h, alpha=0.55):
    band = frame[y:y + h, x:x + w]
    cv2.addWeighted(np.zeros_like(band), alpha, band, 1 - alpha, 0, band)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 70, 80), 1, cv2.LINE_AA)


# ---------------- Entities ----------------
def draw_entity(frame, x1, y1, x2, y2, label, tracked_only=True, primary=False):
    """Amber wireframe cuboid + brackets. Tracked-only entities are thin; the primary gets a soft glow."""
    color = AMBER_HI if primary else AMBER
    h, w = frame.shape[:2]
    if not tracked_only:
        vp = np.array([w / 2, h * HORIZON_Y]); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d = np.array([vp[0] - cx, vp[1] - cy]); n = np.linalg.norm(d)
        depth = int(np.clip((x2 - x1) * 0.22, 8, 44))
        off = (d / n * depth).astype(int) if n > 1 else np.array([0, -depth])
        front = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32); back = front + off
        cv2.polylines(frame, [back], True, (0, 90, 130), 1, cv2.LINE_AA)
        for f, b in zip(front, back):
            cv2.line(frame, tuple(f), tuple(b), (0, 90, 130), 1, cv2.LINE_AA)
        if primary:
            glow_polyline(frame, front, color, None, widths=(12, 2), alphas=(0.28, 1.0), closed=True)
        else:
            cv2.polylines(frame, [front], True, color, 1, cv2.LINE_AA)
    c = max(6, min(18, (x2 - x1) // 5))
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (px, py), (px + dx * c, py), color, 2, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * c), color, 2, cv2.LINE_AA)
    draw_label(frame, x1, y1 - 5 if y1 > 24 else y2 + 18, label, color)


# ---------------- Swept path ----------------
def swept_path_polygon(curve, dh, d_near):
    """Widen the centreline Bezier to the car's physical width. Half-width in pixels follows the
    pinhole ground plane, so the footprint narrows with distance like a real road would."""
    f = f_display()
    half = np.array([CAR_WIDTH_M / 2 * f / ground_distance_at_row(y, dh, d_near) for (_, y) in curve], np.float32)
    half = np.clip(half, 3, 400)
    left = curve.copy(); right = curve.copy()
    left[:, 0] -= half; right[:, 0] += half
    return np.vstack([left, right[::-1]]).astype(np.int32), left.astype(np.int32), right.astype(np.int32)


def draw_swept_path(frame, curve, mode, d_near):
    """mode: cruise (soft cyan), shift (bright green-cyan), yield (amber, ends in a soft stop bar)."""
    poly, left, right = swept_path_polygon(curve, frame.shape[0], d_near)
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
        cv2.circle(frame, (ex, ey), 10, NEON_GREEN, 2, cv2.LINE_AA)
        cv2.circle(frame, (ex, ey), 3, WHITE, -1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (ex, ey), 5, NEON_CYAN, 1, cv2.LINE_AA)


# ---------------- BEV minimap ----------------
def draw_minimap(frame, curve, mode, entities, primary_idx, d_near):
    dh, dw = frame.shape[:2]
    H = bev_homography(dw, dh, d_near)
    bg, ego = bev_background()
    mm = bg.copy()

    # Swept path projected to BEV (edges through the homography)
    poly, left, right = swept_path_polygon(curve, dh, d_near)
    bl, br = to_bev(H, left), to_bev(H, right)
    ok = lambda p: (p[:, 1] > -50) & (p[:, 1] < BEV_H + 50)
    keep = ok(bl) & ok(br)
    if keep.sum() >= 2:
        bl, br = bl[keep], br[keep]
        bpoly = np.vstack([bl, br[::-1]]).astype(np.int32)
        col = NEON_GREEN if mode == "shift" else AMBER if mode == "yield" else NEON_CYAN
        layer = mm.copy(); cv2.fillPoly(layer, [bpoly], col); cv2.addWeighted(layer, 0.28, mm, 0.72, 0, mm)
        cv2.polylines(mm, [bl.astype(np.int32)], False, col, 1, cv2.LINE_AA)
        cv2.polylines(mm, [br.astype(np.int32)], False, col, 1, cv2.LINE_AA)
        if mode == "yield":
            cv2.line(mm, tuple(bl[-1].astype(int)), tuple(br[-1].astype(int)), AMBER_HI, 2, cv2.LINE_AA)

    # Entities: ground-contact point (bottom-centre of box) through the homography
    for i, e in enumerate(entities):
        x1, y1, x2, y2 = e["box"]
        gx, gy = to_bev(H, [[(x1 + x2) / 2, y2]])[0]
        if not (0 <= gx < BEV_W and 0 <= gy < BEV_H):
            continue
        wpx = max(3, int(REAL_WIDTHS_M.get(e["cls_id"], 1.0) * BEV_PPM / 2))
        col = AMBER_HI if i == primary_idx else AMBER if e["hazard"] else GREY
        cv2.rectangle(mm, (int(gx) - wpx, int(gy) - 3), (int(gx) + wpx, int(gy) + 3), col, -1, cv2.LINE_AA)
        if i == primary_idx:
            cv2.circle(mm, (int(gx), int(gy)), 7, AMBER_HI, 1, cv2.LINE_AA)
        cv2.putText(mm, f"{e['distance_m']:.0f}", (int(gx) + wpx + 3, int(gy) + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)

    # Ego vehicle
    ew, el = int(CAR_WIDTH_M * BEV_PPM / 2), int(4.5 * BEV_PPM / 2)
    cv2.rectangle(mm, (ego[0] - ew, ego[1] - el), (ego[0] + ew, ego[1] + el), NEON_CYAN, -1, cv2.LINE_AA)
    cv2.rectangle(mm, (ego[0] - ew, ego[1] - el), (ego[0] + ew, ego[1] + el), WHITE, 1, cv2.LINE_AA)
    cv2.putText(mm, "BEV  SPATIAL MAP", (6, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, GREY, 1, cv2.LINE_AA)

    # Composite bottom-right with a subtle border
    x0, y0 = dw - BEV_W - BEV_MARGIN, dh - BEV_H - BEV_MARGIN
    roi = frame[y0:y0 + BEV_H, x0:x0 + BEV_W]
    cv2.addWeighted(mm, 0.88, roi, 0.12, 0, roi)
    cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + BEV_W, y0 + BEV_H), (70, 80, 92), 1, cv2.LINE_AA)


# ---------------- HUD ----------------
STATE_COLOR = {"CRUISING": NEON_CYAN, "MONITORING": AMBER, "TRAJECTORY SHIFT": NEON_GREEN, "SAFE YIELD": AMBER_HI}


def draw_hud(frame, tel):
    h, w = frame.shape[:2]
    color = STATE_COLOR.get(tel["state"], NEON_CYAN)
    band = frame[0:30, 0:w]; cv2.addWeighted(np.zeros_like(band), 0.45, band, 0.55, 0, band)
    win = f"{tel['window_s']:.1f}s" if tel["window_s"] is not None else "--"
    cv2.putText(frame, f"NETHRA AI  |  {tel['state']}  |  {tel['speed_kmh']:.0f} km/h  |  WINDOW {win}  |  "
                       f"ENTITIES {tel['entities']}  |  INF {tel['infer_ms']:.0f}ms",
                (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    k = tel["kinematics"]
    px, py, pw, ph = 12, h - 124, 292, 112
    draw_panel(frame, px, py, pw, ph)
    cv2.putText(frame, "KINEMATIC ENVELOPE", (px + 10, py + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
    rows = [("LATERAL DEMAND", f"{k['a_req_g']:.2f} / {k['a_max_g']:.2f} g" if k else "--"),
            ("STEER  req / max", f"{k['steer_req_deg']:.1f} / {k['steer_max_deg']:.1f} deg" if k else "--"),
            ("STOP DISTANCE", f"{k['brake_dist_m']:.1f} m" if k else "--"),
            ("ROAD MU", f"{tel['road_mu']:.2f}")]
    for i, (lab, val) in enumerate(rows):
        yy = py + 38 + i * 17
        cv2.putText(frame, lab, (px + 10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
        vcol = AMBER_HI if (k and i == 0 and not k["feasible"]) else WHITE
        cv2.putText(frame, val, (px + 148, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, vcol, 1, cv2.LINE_AA)
    if k:
        bx, by, bw = px + 10, py + ph - 9, pw - 20
        cv2.line(frame, (bx, by), (bx + bw, by), (60, 70, 80), 3)
        cv2.line(frame, (bx, by), (bx + int(bw * min(k["a_req_g"], 1.0)), by), AMBER_HI if not k["feasible"] else NEON_GREEN, 3)
        lim = bx + int(bw * min(k["a_max_g"], 1.0)); cv2.line(frame, (lim, by - 5), (lim, by + 5), AMBER, 2)

    if tel["state"] == "SAFE YIELD":
        txt = "SAFE YIELD  -  CONTROLLED DECELERATION"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
        x, y = (w - tw) // 2, 70
        draw_panel(frame, x - 18, y - th - 14, tw + 36, th + 26, alpha=0.6)
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, AMBER_HI, 1, cv2.LINE_AA)

    steer = tel["steer_deg"]
    tick_x = int(w / 2 + (steer / MAX_STEER_DEG) * (w * 0.18))
    cv2.line(frame, (w // 2 - int(w * 0.18), h - 4), (w // 2 + int(w * 0.18), h - 4), GREY, 1, cv2.LINE_AA)
    cv2.line(frame, (tick_x, h - 12), (tick_x, h - 1), color, 3, cv2.LINE_AA)


# ---------------- Free space ----------------
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
    dy = max((frame_h - 8) - target[1], 1)
    return float(np.clip(math.degrees(math.atan2(dx, dy)), -MAX_STEER_DEG, MAX_STEER_DEG))


# ---------------- Planner ----------------
class SpatialPlanner:
    def __init__(self, model):
        self.model = model
        self._target_ema = None
        self._last_dir = None
        self._tracks = {}
        self.d_near = D_NEAR                  # self-calibrated ground-plane constant (m at bottom row)

    def _closing_rate(self, tid, dist, now):
        hist = self._tracks.setdefault(tid, deque(maxlen=TRACK_HISTORY))
        hist.append((now, dist))
        if len(hist) < 4 or hist[-1][0] - hist[0][0] < 0.2:
            return 0.0
        ts = np.array([p[0] for p in hist]); ds = np.array([p[1] for p in hist])
        return float(max(-np.polyfit(ts - ts[0], ds, 1)[0], 0.0))

    def process(self, frame_bgr, speed_kmh, road_mu, critical_distance=CRITICAL_DISTANCE, conf=CONF_THRESHOLD, enhance=False):
        t0 = now = time.perf_counter()
        iw, ih = INFER_SIZE; dw, dh = DISPLAY_SIZE; sx, sy = dw / iw, dh / ih
        small = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
        canvas = cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_LINEAR)
        if enhance:
            lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
            lab[..., 0] = cv2.createCLAHE(2.5, (8, 8)).apply(lab[..., 0])
            small = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        t_inf = time.perf_counter()
        results = self.model.track(small, imgsz=640, conf=conf, classes=list(TARGET_CLASSES.keys()), persist=True,
                                   tracker="bytetrack.yaml", device=DEVICE, half=(DEVICE == "cuda"), verbose=False)[0]
        infer_ms = (time.perf_counter() - t_inf) * 1000.0
        v_ego = speed_kmh / 3.6

        entities, live = [], set()
        if results.boxes is not None and len(results.boxes):
            xyxy = results.boxes.xyxy.cpu().numpy(); clss = results.boxes.cls.cpu().numpy().astype(int)
            ids = results.boxes.id.cpu().numpy().astype(int) if results.boxes.id is not None else [None] * len(xyxy)
            for (bx1, by1, bx2, by2), cls_id, tid in zip(xyxy, clss, ids):
                dist = float(estimate_distance(int(cls_id), float(bx2 - bx1), iw))
                closing = self._closing_rate(int(tid), dist, now) if tid is not None else 0.0
                if tid is not None: live.add(int(tid))
                v_close = v_ego + closing
                window = dist / v_close if v_close > 0.05 else None
                entities.append({"id": int(tid) if tid is not None else -1, "cls_id": int(cls_id),
                                 "class": TARGET_CLASSES.get(int(cls_id), "OBJECT"), "distance_m": round(dist, 1),
                                 "closing_mps": round(closing, 2), "window_s": None if window is None else round(window, 2),
                                 "hazard": dist < critical_distance or (window is not None and window < WINDOW_MONITOR),
                                 "mpp": REAL_WIDTHS_M.get(int(cls_id), DEFAULT_WIDTH_M) / max(float(bx2 - bx1) * sx, 1.0),
                                 "box": (int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy))})
        for tid in list(self._tracks):
            if tid not in live: del self._tracks[tid]

        # Self-calibrate the ground plane from entities with a usable footprint (robust median + EMA)
        ests = [d_near_from_entity(e["distance_m"], e["box"][3], dh) for e in entities
                if (e["box"][2] - e["box"][0]) >= 24 and e["box"][3] > HORIZON_Y * dh + 20]
        if ests:
            est = float(np.clip(np.median(ests), 1.5, 14.0))
            self.d_near += 0.15 * (est - self.d_near)

        hazards = sorted((e for e in entities if e["hazard"]), key=lambda e: (e["window_s"] if e["window_s"] is not None else 1e9, e["distance_m"]))
        primary = hazards[0] if hazards else None
        primary_idx = entities.index(primary) if primary else -1

        # ---- Plan ----
        state, mode, direction, reason = "CRUISING", "cruise", "NONE", None
        free_l = free_r = None; free_l_m = free_r_m = None; kin = None; lateral_signed = 0.0
        raw = np.array([dw // 2, int(dh * (HORIZON_Y + 0.10))], np.float32)
        window_primary = primary["window_s"] if primary else None

        if primary:
            state = "MONITORING"
            act = primary["distance_m"] < critical_distance or (window_primary is not None and window_primary < WINDOW_ACT)
            if act:
                others = [e["box"] for e in entities if e is not primary]
                free_l, free_r = free_space_vector(primary["box"], others, dw)
                free_l_m, free_r_m = round(free_l * primary["mpp"], 1), round(free_r * primary["mpp"], 1)
                x1, y1, x2, y2 = primary["box"]
                clearance_px = max(ESCAPE_MARGIN_PX, int(CLEARANCE_M / primary["mpp"]))
                # Candidate targets beside the hazard; valid only if that side has room for the clearance.
                # Direction is ego-relative: which way the CAR shifts (sign of the lateral offset).
                cands = []
                if free_r >= clearance_px: cands.append(int(np.clip(x2 + clearance_px, 16, dw - 16)))
                if free_l >= clearance_px: cands.append(int(np.clip(x1 - clearance_px, 16, dw - 16)))
                cands = [("RIGHT" if c >= dw / 2 else "LEFT", c) for c in cands]
                cands.sort(key=lambda c: abs(c[1] - dw / 2))           # least lateral shift first
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
        if mode != "shift": self._last_dir = None

        self._target_ema = raw if self._target_ema is None else self._target_ema + ESCAPE_SMOOTHING * (raw - self._target_ema)
        target = (float(self._target_ema[0]), float(self._target_ema[1]))
        steer = steering_angle_deg(dw, dh, target) if mode == "shift" else 0.0

        # ---- Render ----
        start = (dw / 2, dh - 8.0)
        control = (dw / 2, target[1] + (start[1] - target[1]) / 3)
        curve = bezier_curve(start, control, target)
        draw_swept_path(canvas, curve, mode, self.d_near)
        for i, e in enumerate(entities):
            wtxt = f"  {e['window_s']:.1f}s" if e["window_s"] is not None else ""
            draw_entity(canvas, *e["box"], f"{e['class']}  {e['distance_m']:.1f}m{wtxt}",
                        tracked_only=not e["hazard"], primary=(i == primary_idx and mode != "cruise"))
        draw_minimap(canvas, curve, mode, entities, primary_idx, self.d_near)

        avail = None
        if free_l_m is not None:
            avail = round(min(max(free_l_m, free_r_m) / (CLEARANCE_M * 2.0), 1.0), 2)
        tel = {"state": state, "mode": mode, "reason": reason, "direction": direction,
               "lateral_m": round(lateral_signed, 2), "steer_deg": round(steer, 1),
               "speed_kmh": float(speed_kmh), "road_mu": float(road_mu),
               "entities": len(entities), "hazards": len(hazards), "window_s": window_primary,
               "nearest_m": min((e["distance_m"] for e in entities), default=None),
               "primary": ({"class": primary["class"], "distance_m": primary["distance_m"], "window_s": primary["window_s"]} if primary else None),
               "free_left_m": free_l_m, "free_right_m": free_r_m, "free_availability": avail, "kinematics": kin,
               "ground_plane_near_m": round(self.d_near, 2),
               "infer_ms": round(infer_ms, 1), "pipeline_ms": 0.0,
               "detections": [{k: v for k, v in e.items() if k not in ("box", "mpp", "cls_id")} for e in entities]}
        draw_hud(canvas, tel)
        tel["pipeline_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return canvas, tel


# ---------------- Threaded worker ----------------
class VideoInferenceWorker(threading.Thread):
    """Background planner. Publishes (jpeg_bytes, telemetry); oldest frames dropped if the UI lags.
    Hot-tunable attributes: speed_kmh, road_mu, critical_distance, conf, loop, enhance."""

    def __init__(self, video_path, model, speed_kmh=45.0, road_mu=ROAD_MU_DRY, critical_distance=CRITICAL_DISTANCE,
                 conf=CONF_THRESHOLD, loop=True, enhance=False, encode_jpeg=True, realtime=True):
        super().__init__(daemon=True, name="NethraSpatialPlanner")
        self.video_path, self.model = video_path, model
        self.planner = SpatialPlanner(model)
        self.out_queue = queue.Queue(maxsize=QUEUE_DEPTH)
        self.speed_kmh, self.road_mu = speed_kmh, road_mu
        self.critical_distance, self.conf = critical_distance, conf
        self.loop, self.enhance, self.encode_jpeg, self.realtime = loop, enhance, encode_jpeg, realtime
        self._stop_evt, self._pause_evt, self._seek_to = threading.Event(), threading.Event(), None
        self.frame_idx, self.source_fps, self.error = 0, 30.0, None
        self.stats = {"fps": 0.0, "dropped": 0, "skipped": 0}

    def stop(self): self._stop_evt.set()
    def pause(self, flag): self._pause_evt.set() if flag else self._pause_evt.clear()
    def seek(self, idx): self._seek_to = int(idx)

    @property
    def running(self): return self.is_alive() and not self._stop_evt.is_set()

    def latest(self, timeout=1.0):
        try:
            item = self.out_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        while True:
            try: item = self.out_queue.get_nowait()
            except queue.Empty: return item

    def _publish(self, item):
        while True:
            try:
                self.out_queue.put_nowait(item); return
            except queue.Full:
                try: self.out_queue.get_nowait(); self.stats["dropped"] += 1
                except queue.Empty: pass

    def run(self):
        reset_tracker(self.model)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error = f"Could not open video: {self.video_path}"; return
        self.source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        period = 1.0 / self.source_fps
        fps_ema, deadline = None, time.perf_counter()
        try:
            while not self._stop_evt.is_set():
                if self._pause_evt.is_set():
                    time.sleep(0.05); deadline = time.perf_counter(); continue
                if self._seek_to is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_to); self._seek_to = None
                    reset_tracker(self.model); deadline = time.perf_counter()
                t_frame = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    if self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0); reset_tracker(self.model); deadline = time.perf_counter(); continue
                    break
                self.frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                canvas, tel = self.planner.process(frame, self.speed_kmh, self.road_mu, critical_distance=self.critical_distance,
                                                   conf=self.conf, enhance=self.enhance)
                if self.encode_jpeg:
                    ok_enc, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    payload = buf.tobytes() if ok_enc else None
                else:
                    payload = canvas
                dt = max(time.perf_counter() - t_frame, 1e-6)
                fps_ema = (1 / dt) if fps_ema is None else 0.9 * fps_ema + 0.1 / dt
                self.stats["fps"] = round(fps_ema, 1)
                tel.update({"fps": self.stats["fps"], "frame_idx": self.frame_idx, "source_fps": round(self.source_fps, 1),
                            "dropped": self.stats["dropped"], "skipped": self.stats["skipped"]})
                self._publish((payload, tel))
                if self.realtime:
                    deadline += period
                    lag = time.perf_counter() - deadline
                    if lag < 0:
                        time.sleep(-lag)
                    else:
                        for _ in range(min(int(lag / period), 10)):
                            if cap.grab(): self.stats["skipped"] += 1
                        deadline = time.perf_counter()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            cap.release()


def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "city.mp4"
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
    worker = VideoInferenceWorker(path, load_model(), speed_kmh=speed, encode_jpeg=False, loop=False)
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