"""
Nethra AI - Spatial Planner Console
UI thread only: dequeues pre-rendered JPEG frames + telemetry from the background planner in engine.py.

Run:  streamlit run app.py
Expected clips next to this file: highway.mp4, city.mp4, night.mp4 (or pick "Custom clip").
Python 3.9+ compatible (no nested f-strings, no f-string quote reuse).
"""

import html
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import deque

import streamlit as st


# ---------------- OpenCV bootstrap (Streamlit Community Cloud) ----------------
# ultralytics depends on opencv-python, whose wheel needs libGL.so.1 (apt). Cloud's apt is unreliable and
# its venv is read-only at runtime, so if the non-headless wheel won the install race we install
# opencv-python-headless into a WRITABLE folder and put it ahead of site-packages on sys.path.
# One-time cost per fresh container (~15-30 s download); no-op afterwards and on machines where cv2 imports.
def _purge_cv2_modules():
    for name in list(sys.modules):
        if name == "cv2" or name.startswith("cv2."):
            del sys.modules[name]
    # OpenCV's loader sets this guard before it fails on libGL; leaving it set makes the
    # next import raise "recursion is detected during loading of cv2 binary extensions".
    if hasattr(sys, "OpenCV_LOADER"):
        delattr(sys, "OpenCV_LOADER")


def _import_cv2():
    _purge_cv2_modules()
    importlib.invalidate_caches()
    import cv2  # noqa: F401
    return cv2


def _writable_dir():
    for base in (os.path.expanduser("~"), tempfile.gettempdir(), os.getcwd()):
        path = os.path.join(base, ".nethra_cv2")
        try:
            os.makedirs(path, exist_ok=True)
            probe = os.path.join(path, ".probe")
            with open(probe, "w") as fh:
                fh.write("ok")
            os.remove(probe)
            return path
        except OSError:
            continue
    raise RuntimeError("No writable directory available for the OpenCV fallback.")


def _ensure_headless_opencv():
    try:
        _import_cv2()
        return
    except ImportError as exc:
        text = str(exc)
        if not any(key in text for key in ("libGL", "libgthread", "libglib", "libxcb", "libSM", "libICE")):
            raise
    target = _writable_dir()
    if not os.path.isdir(os.path.join(target, "cv2")):
        with st.spinner("First boot on this server: fetching headless OpenCV (one-time, ~20 s)..."):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--no-cache-dir",
                                   "--target", target, "opencv-python-headless>=4.9"])
    if target not in sys.path:
        sys.path.insert(0, target)
    _import_cv2()


_ensure_headless_opencv()

import engine  # noqa: E402

st.set_page_config(page_title="Nethra AI | Spatial Planner", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------- Engine handshake ----------------
REQUIRED_ENGINE = "6.2"
_loaded = getattr(engine, "ENGINE_VERSION", None)
if _loaded != REQUIRED_ENGINE:
    _msg = ("**engine.py mismatch.** app.py needs engine v" + REQUIRED_ENGINE + ", but Python loaded **"
            + ("v" + _loaded if _loaded else "a pre-6.0 engine") + "** from:\n\n`" + str(engine.__file__)
            + "`\n\nFix: overwrite that file with the matching engine.py, delete the `__pycache__` folder next to it, "
            "then stop and restart `streamlit run app.py`.")
    st.error(_msg)
    st.stop()

# ---------------- Scenario library ----------------
SCENARIOS = {
    "Highway - High Speed (Rollover Risk)": {
        "file": "highway.mp4", "speed": 85.0, "mu": engine.ROAD_MU_DRY, "enhance": False,
        "blurb": "Dry highway at 85 km/h. Narrow lateral envelope: the planner favours controlled yields over sharp shifts."},
    "City - Low Speed (Safe Evasion)": {
        "file": "city.mp4", "speed": 30.0, "mu": engine.ROAD_MU_DRY, "enhance": False,
        "blurb": "Urban traffic at 30 km/h. Wide envelope: smooth swept-path trajectory shifts around entities."},
    "Night/Rain - Low Visibility": {
        "file": "night.mp4", "speed": 45.0, "mu": engine.ROAD_MU_WET, "enhance": True,
        "blurb": "Wet road (mu 0.45) with low-light enhancement on the perception input. Envelope tightens with grip."},
    "Custom clip": {"file": "", "speed": 45.0, "mu": engine.ROAD_MU_DRY, "enhance": False, "blurb": ""},
}

# ---------------- Performance profiles ----------------
PROFILES = {
    "Ultra (laptop, native res)":   {"infer_every": 3, "imgsz": 416, "render_h": 0,   "stride": 1, "jpeg": 82},
    "Balanced (720p)":              {"infer_every": 4, "imgsz": 384, "render_h": 720, "stride": 1, "jpeg": 75},
    "Cloud Lite (540p, shared CPU)": {"infer_every": 6, "imgsz": 320, "render_h": 540, "stride": 2, "jpeg": 65},
}
_ON_CLOUD = os.path.isdir("/mount/src") or os.environ.get("STREAMLIT_SHARING_MODE") is not None
DEFAULT_PROFILE_IDX = 2 if _ON_CLOUD else 0

# ---------------- Styling (plain string, not an f-string) ----------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, system-ui, sans-serif; }
.block-container { padding-top: 0.6rem; padding-bottom: 0.3rem; max-width: 100%; }

/* Ambient background: deep navy base, drifting cyan/violet auroras, faint engineering grid */
.stApp {
  background:
    radial-gradient(1200px 700px at 8% -10%, rgba(0,229,255,0.16), transparent 60%),
    radial-gradient(900px 600px at 100% 0%, rgba(123,47,255,0.20), transparent 60%),
    radial-gradient(1000px 800px at 50% 120%, rgba(0,180,140,0.12), transparent 60%),
    linear-gradient(180deg, #070B12 0%, #0A0F18 55%, #060910 100%);
  background-attachment: fixed;
}
.stApp::before {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    linear-gradient(rgba(120,160,200,0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(120,160,200,0.045) 1px, transparent 1px);
  background-size: 42px 42px;
  -webkit-mask-image: radial-gradient(ellipse at 50% 30%, rgba(0,0,0,0.9), transparent 75%);
  mask-image: radial-gradient(ellipse at 50% 30%, rgba(0,0,0,0.9), transparent 75%);
}
.stApp::after {
  content: ""; position: fixed; inset: -20%; pointer-events: none; z-index: 0;
  background:
    radial-gradient(600px 400px at 20% 30%, rgba(0,229,255,0.10), transparent 65%),
    radial-gradient(700px 500px at 80% 70%, rgba(123,47,255,0.12), transparent 65%);
  filter: blur(40px);
  animation: nethra-drift 26s ease-in-out infinite alternate;
}
@keyframes nethra-drift {
  from { transform: translate3d(-3%, -2%, 0) rotate(0deg); }
  to   { transform: translate3d(3%, 3%, 0) rotate(4deg); }
}
@keyframes nethra-pulse { 50% { filter: brightness(1.18); } }
.stApp > * { position: relative; z-index: 1; }
header[data-testid="stHeader"] { background: transparent; }

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(10,14,22,0.92), rgba(7,10,16,0.96));
  border-right: 1px solid rgba(0,229,255,0.10);
  backdrop-filter: blur(14px);
  box-shadow: inset -1px 0 0 rgba(255,255,255,0.02), 8px 0 40px rgba(0,0,0,0.35);
}
section[data-testid="stSidebar"]::before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(400px 300px at 0% 0%, rgba(0,229,255,0.10), transparent 70%);
}

.brand { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
.brand .logo { width: 34px; height: 34px; border-radius: 9px;
  background: conic-gradient(from 200deg, #00E5FF, #7B2FFF, #00E5FF);
  box-shadow: 0 0 28px rgba(0,229,255,0.45), 0 0 60px rgba(123,47,255,0.35); }
.brand h1 { font-size: 1.35rem; font-weight: 700; letter-spacing: 0.14em; color: #EAF2FF; margin: 0; }
.brand .sub { color: #5C6B7C; font-size: 0.72rem; letter-spacing: 0.14em; margin-top: 2px; }
.mono { font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace; }
h3 { margin: 0.55rem 0 0.35rem 0; letter-spacing: 0.14em; font-size: 0.64rem; color: #5C6B7C; font-weight: 600; }

.state { text-align: center; padding: 12px 10px; border-radius: 10px; font-weight: 700;
  letter-spacing: 0.16em; font-size: 0.86rem; margin-bottom: 8px; }
.st-cruise  { background: linear-gradient(180deg,#062A2E,#04191C); color: #00E5FF; border: 1px solid #0E6E7A; box-shadow: 0 0 18px rgba(0,229,255,0.18); }
.st-monitor { background: linear-gradient(180deg,#2E2208,#1B1404); color: #FFB020; border: 1px solid #8A6010; }
.st-shift   { background: linear-gradient(180deg,#07301A,#04190E); color: #39FF14; border: 1px solid #1E8A3A; box-shadow: 0 0 22px rgba(57,255,20,0.22); animation: nethra-pulse 1.1s ease-in-out infinite; }
.st-yield   { background: linear-gradient(180deg,#332208,#1C1204); color: #FFB020; border: 1px solid #A87010; box-shadow: 0 0 22px rgba(255,176,32,0.28); animation: nethra-pulse 1.4s ease-in-out infinite; }
.st-idle    { background: #0F151C; color: #5C6B7C; border: 1px solid #1E2A36; }

.matrix { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-bottom: 8px; }
.cell { text-align: center; padding: 7px 2px; border-radius: 7px; font-size: 0.6rem; letter-spacing: 0.1em; font-weight: 600;
  color: #3E4C5A; background: rgba(11,16,22,0.75); border: 1px solid rgba(120,160,200,0.12); backdrop-filter: blur(8px); }
.cell.on-cruise  { color: #00E5FF; border-color: #0E6E7A; background: #062A2E; }
.cell.on-monitor { color: #FFB020; border-color: #8A6010; background: #2E2208; }
.cell.on-shift   { color: #39FF14; border-color: #1E8A3A; background: #07301A; }
.cell.on-yield   { color: #FFB020; border-color: #A87010; background: #332208; }

.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 6px; }
.card { background: linear-gradient(180deg, rgba(20,28,40,0.72), rgba(12,18,26,0.78)); border: 1px solid rgba(120,160,200,0.14);
  border-radius: 10px; padding: 8px 10px; backdrop-filter: blur(10px);
  box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.28); }
.card:hover { border-color: rgba(0,229,255,0.28); }
.card .k { color: #5C6B7C; font-size: 0.58rem; letter-spacing: 0.12em; }
.card .v { color: #EAF2FF; font-size: 1.12rem; font-weight: 600; margin-top: 1px; }
.card .v.small { font-size: 0.9rem; }
.card .v .dim { font-size: 0.7rem; color: #5C6B7C; }
.card.warn .v { color: #FFB020; } .card.good .v { color: #39FF14; } .card.cyan .v { color: #00E5FF; }

.bar { position: relative; height: 6px; background: #0B1016; border-radius: 3px; margin-top: 6px; border: 1px solid #141C25; }
.bar .fill { position: absolute; left: 0; top: 0; height: 100%; border-radius: 3px; background: linear-gradient(90deg,#00E5FF,#39FF14); }
.bar .fill.warn { background: linear-gradient(90deg,#FFB020,#FF8A20); }
.bar .lim { position: absolute; top: -4px; width: 2px; height: 14px; background: #FFB020; }

.term { background: linear-gradient(180deg, rgba(5,8,12,0.92), rgba(3,5,8,0.96)); border: 1px solid rgba(0,229,255,0.12); border-radius: 10px;
  padding: 10px 12px; font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace; font-size: 0.66rem; line-height: 1.6;
  height: 232px; overflow: hidden; white-space: pre; color: #9FB3C8;
  box-shadow: 0 0 30px rgba(0,229,255,0.05) inset, 0 8px 24px rgba(0,0,0,0.35); }
.t-sys { color: #00E5FF; } .t-shift { color: #39FF14; font-weight: 600; } .t-plan { color: #8B9AAB; }
.t-kin { color: #FFB020; } .t-yield { color: #FFB020; font-weight: 600; } .t-err { color: #FF8A20; } .t-dim { color: #3E4C5A; }

.perf { color: #3E4C5A; font-size: 0.62rem; letter-spacing: 0.06em; font-family: 'JetBrains Mono', monospace; margin-top: 4px; }
.scen { background: rgba(11,16,22,0.7); border: 1px solid rgba(123,47,255,0.25); border-left: 2px solid #7B2FFF; border-radius: 8px;
  padding: 8px 10px; font-size: 0.7rem; color: #8B9AAB; margin: 6px 0 4px 0; backdrop-filter: blur(8px); }
div[data-testid="stImage"] img { border-radius: 14px; border: 1px solid rgba(0,229,255,0.16);
  box-shadow: 0 0 0 1px rgba(255,255,255,0.03), 0 24px 60px rgba(0,0,0,0.55), 0 0 60px rgba(0,229,255,0.10); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

BRAND_HTML = (
    '<div class="brand"><div class="logo"></div>'
    '<div><h1>NETHRA AI</h1>'
    '<div class="sub">AUTONOMOUS SPATIAL PLANNER · PURE-VISION · SWEPT-PATH ROUTING · KINEMATIC ENVELOPE</div>'
    '</div></div>'
)
st.markdown(BRAND_HTML, unsafe_allow_html=True)


# ---------------- Worker container + session state ----------------
@st.cache_resource(show_spinner=False)
def worker_container():
    """Process-wide holder for the single background planner. Survives reruns and page refreshes."""
    return {"worker": None, "scenario": None}


@st.cache_resource(show_spinner="Loading YOLOv8 weights...")
def get_model():
    return engine.load_model()


_DEFAULTS = {
    "last_telemetry": None, "term": deque(maxlen=15), "steer": 0.0, "prev_state": None,
    "prev_reason": None, "prev_primary": None, "last_ev_log": 0.0, "scenario_prev": None, "ui_fps": 0.0,
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

TELEMETRY_HZ = 8


def term(kind, msg):
    st.session_state.term.append((kind, time.strftime("%H:%M:%S") + " > " + msg))


def get_worker():
    return worker_container()["worker"]


def stop_worker():
    w = get_worker()
    if w is not None:
        w.stop()
        w.join(timeout=3.0)
    worker_container()["worker"] = None


def start_worker(path, speed, mu, enhance, critical, conf, loop, infer_every, imgsz, render_h, stride, jpegq):
    stop_worker()                                   # join the old thread before a new one touches the model
    w = engine.VideoInferenceWorker(path, get_model(), speed_kmh=speed, road_mu=mu, critical_distance=critical,
                                    conf=conf, loop=loop, enhance=enhance, infer_every=infer_every, imgsz=imgsz,
                                    render_max_height=render_h, frame_stride=stride, jpeg_quality=jpegq,
                                    realtime=False)  # buffered mode: the UI paces display at source FPS
    w.start()
    worker_container()["worker"] = w
    src = os.path.basename(path) if path else "?"
    term("sys", "[SYSTEM] Spatial planner online. Source=" + src + "  mu=" + format(mu, ".2f")
         + "  enhance=" + ("on" if enhance else "off"))
    term("plan", "[PLANNER] Nominal swept path engaged. Continuous-flow mode.")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### SIMULATION SCENARIO")
    scenario = st.selectbox("Simulation Scenario", list(SCENARIOS), label_visibility="collapsed")
    cfg = SCENARIOS[scenario]

    # Scenario switch: re-seed every scenario-bound control so telemetry matches the selection
    if scenario != st.session_state.scenario_prev:
        st.session_state["speed_kmh"] = float(cfg["speed"])
        st.session_state["road_mu"] = float(cfg["mu"])
        st.session_state["enhance"] = bool(cfg["enhance"])
        st.session_state.scenario_prev = scenario
    if cfg["blurb"]:
        st.markdown('<div class="scen">' + html.escape(cfg["blurb"]) + "</div>", unsafe_allow_html=True)
    if scenario == "Custom clip":
        video_path = st.text_input("Clip path", value="custom.mp4")
    else:
        video_path = cfg["file"]

    _stem = os.path.splitext(os.path.basename(video_path))[0] if video_path else ""
    replay_mp4 = os.path.join("recordings", _stem + ".mp4")
    replay_log = os.path.join("recordings", _stem + ".jsonl")
    has_replay = os.path.isfile(replay_mp4) and os.path.isfile(replay_log)
    mode_options = ["Live planner"] + (["Replay (pre-rendered run)"] if has_replay else [])
    run_mode = st.radio("Mode", mode_options, index=(1 if (has_replay and _ON_CLOUD) else 0), horizontal=True,
                        help="Replay plays a run recorded with record.py: identical planner output, near-zero server CPU.")
    REPLAY = run_mode.startswith("Replay")

    st.markdown("### VEHICLE")
    speed_kmh = st.slider("Simulated Vehicle Speed (km/h)", 0.0, 140.0, step=1.0, key="speed_kmh")
    road_mu = st.slider("Road friction (mu)", 0.20, 0.90, step=0.05, key="road_mu",
                        help="0.7 dry asphalt · 0.45 wet · 0.3 mud/gravel")

    st.markdown("### PERCEPTION")
    critical_distance = st.slider("Critical distance (m)", 5.0, 40.0, engine.CRITICAL_DISTANCE, 0.5)
    conf_threshold = st.slider("Detection confidence", 0.10, 0.90, engine.CONF_THRESHOLD, 0.05)
    enhance = st.checkbox("Low-light enhancement (CLAHE)", key="enhance")
    loop_video = st.checkbox("Loop clip", value=True)

    st.markdown("### COMPUTE")
    profile = st.selectbox("Performance profile", list(PROFILES), index=DEFAULT_PROFILE_IDX,
                           help="Ultra for a laptop. Cloud Lite for Streamlit Community Cloud (shared / throttled CPU).")
    prof = PROFILES[profile]
    with st.expander("Advanced compute", expanded=False):
        infer_every = st.select_slider("Perception cadence (every N frames)", options=[1, 2, 3, 4, 5, 6, 8], value=prof["infer_every"],
                                       key="ie_" + profile, help="Tracklets are dead-reckoned between perception frames; planning runs every frame.")
        imgsz = st.select_slider("YOLO input size", options=[256, 320, 384, 416, 480, 512, 640], value=prof["imgsz"], key="sz_" + profile)
        render_h = st.select_slider("Render height (px)", options=[0, 540, 720, 1080], value=prof["render_h"], key="rh_" + profile,
                                    help="0 = native. Lower = cheaper drawing + smaller JPEGs.")
        stride = st.select_slider("Source frame stride", options=[1, 2, 3], value=prof["stride"], key="st_" + profile,
                                  help="2 = process every 2nd source frame (playback speed unchanged).")
        jpegq = st.select_slider("JPEG quality", options=[55, 65, 75, 82, 90], value=prof["jpeg"], key="jq_" + profile)

    st.markdown("### CONTROL")
    c1, c2 = st.columns(2)
    engage = c1.button("▶ Engage", type="primary", use_container_width=True)
    disengage = c2.button("■ Disengage", use_container_width=True)
    reset = st.button("↺ Restart clip", use_container_width=True)
    st.caption(engine.DEVICE.upper() + " · `" + engine.MODEL_PATH + "` + ByteTrack · native-resolution HUD · BEV "
               + format(engine.BEV_D_MAX, ".0f") + " m")
    st.caption("engine v" + engine.ENGINE_VERSION + " · `" + os.path.basename(engine.__file__) + "`")

# Scenario switch while running -> clean restart on the new clip
worker = get_worker()
if engage:
    start_worker(video_path, speed_kmh, road_mu, enhance, critical_distance, conf_threshold, loop_video, infer_every, imgsz, render_h, stride, jpegq)
elif worker is not None and worker.running and worker_container()["scenario"] != scenario:
    term("sys", "[SYSTEM] Scenario switch -> " + scenario + ". Restarting stream.")
    start_worker(video_path, speed_kmh, road_mu, enhance, critical_distance, conf_threshold, loop_video, infer_every, imgsz, render_h, stride, jpegq)
worker_container()["scenario"] = scenario
if disengage:
    stop_worker()
    term("sys", "[SYSTEM] Planner disengaged by operator.")
if reset and get_worker() is not None:
    get_worker().seek(0)

worker = get_worker()
if worker is not None:                                  # hot-apply tunables (atomic attribute writes)
    worker.speed_kmh, worker.road_mu = speed_kmh, road_mu
    worker.critical_distance, worker.conf = critical_distance, conf_threshold
    worker.enhance, worker.loop = enhance, loop_video
    worker.infer_every, worker.imgsz = infer_every, imgsz
    worker.render_max_height, worker.frame_stride, worker.jpeg_quality = render_h, stride, jpegq

# ---------------- Layout ----------------
col_feed, col_panel = st.columns([2.9, 1.25])
with col_feed:
    feed = st.empty()          # single-slot container: each .image() call replaces the previous frame
    perf = st.empty()
with col_panel:
    panel = st.empty()
    st.markdown("### AUTONOMOUS ACTION TERMINAL")
    term_box = st.empty()


# ---------------- Renderers ----------------
def fmt_steer(deg):
    if abs(deg) < 0.5:
        return "0.0° · centre"
    sign = "+" if deg > 0 else "-"
    side = "Right" if deg > 0 else "Left"
    return sign + format(abs(deg), ".1f") + "° " + side


def fmt_num(v, fmt, suffix="", none="—"):
    return none if v is None else format(v, fmt) + suffix


def card(label, value, cls="", small=False, extra=""):
    vcls = "v mono small" if small else "v mono"
    return ('<div class="card ' + cls + '"><div class="k">' + label + '</div><div class="' + vcls + '">'
            + value + "</div>" + extra + "</div>")


def bar(fill_pct, warn=False, limit_pct=None):
    fill = '<div class="fill' + (" warn" if warn else "") + '" style="width:' + format(fill_pct, ".0f") + '%"></div>'
    lim = '<div class="lim" style="left:' + format(limit_pct, ".0f") + '%"></div>' if limit_pct is not None else ""
    return '<div class="bar">' + fill + lim + "</div>"


def render_terminal():
    lines = list(st.session_state.term)
    if not lines:
        body = '<span class="t-dim">terminal idle — engage the planner to begin</span>'
    else:
        body = "\n".join('<span class="t-' + kind + '">' + html.escape(text) + "</span>" for kind, text in lines)
    term_box.markdown('<div class="term">' + body + "</div>", unsafe_allow_html=True)


STATE_CLS = {"CRUISING": "cruise", "MONITORING": "monitor", "TRAJECTORY SHIFT": "shift", "SAFE YIELD": "yield"}
STATE_LABEL = {"CRUISING": "● CRUISING · CONTINUOUS FLOW", "MONITORING": "● MONITORING ENTITIES",
               "TRAJECTORY SHIFT": "● TRAJECTORY SHIFT · SWEPT PATH", "SAFE YIELD": "● SAFE YIELD · CONTROLLED DECEL"}


def render_panel(tel):
    state = tel["state"]
    cls = STATE_CLS.get(state, "idle")
    label = STATE_LABEL.get(state, "○ PLANNER DISENGAGED")
    k = tel.get("kinematics") or {}
    win = tel.get("window_s")
    lat = tel.get("lateral_m", 0.0) or 0.0
    steer = st.session_state.steer

    cells = ""
    for c, n in (("cruise", "CRUISE"), ("monitor", "MONITOR"), ("shift", "SHIFT"), ("yield", "YIELD")):
        cells += '<div class="cell' + (" on-" + c if c == cls else "") + '">' + n + "</div>"

    # Evasion window
    if win is None:
        win_txt, win_cls = "open", ""
    else:
        win_txt = format(win, ".2f") + " s"
        win_cls = "warn" if win < engine.WINDOW_ACT else "good"

    # Lateral shift (signed, ego-relative)
    if state in ("TRAJECTORY SHIFT", "SAFE YIELD"):
        lat_txt = ("+" if lat > 0 else "") + format(lat, ".2f") + " m"
    else:
        lat_txt = "0.00 m"

    # Lateral demand vs envelope
    a_req, a_max = k.get("a_req_g"), k.get("a_max_g")
    if a_req is not None:
        g_txt = format(a_req, ".2f") + " / " + format(a_max, ".2f") + " g"
        g_bar = bar(min(a_req, 1.0) * 100, warn=not k.get("feasible", True), limit_pct=min(a_max, 1.0) * 100)
        g_cls = "warn" if not k.get("feasible", True) else ""
    else:
        g_txt, g_bar, g_cls = "—", '<div class="bar"></div>', ""

    # Free-space availability
    avail = tel.get("free_availability")
    if avail is not None:
        fs_txt = "L " + format(tel["free_left_m"], ".1f") + " m · R " + format(tel["free_right_m"], ".1f") + " m"
        fs_bar = bar(avail * 100, warn=avail < 0.5)
        fs_word = "HIGH" if avail >= 0.99 else ("PARTIAL" if avail >= 0.5 else "LOW")
    else:
        fs_txt, fs_bar, fs_word = "unconstrained", bar(100), "HIGH"

    steer_txt = k.get("steer_req_deg")
    if steer_txt is not None:
        steer_txt = format(k["steer_req_deg"], ".1f") + "° / " + format(k["steer_max_deg"], ".1f") + "°"
    else:
        steer_txt = "—"

    ent_txt = str(tel["entities"]) + '<span class="dim"> · ' + str(tel["hazards"]) + " in ego-path</span>"
    shift_cls = "good" if state == "TRAJECTORY SHIFT" else ""

    block = ('<div class="state st-' + cls + '">' + label + "</div>"
             + '<div class="matrix">' + cells + "</div>"
             + '<div class="grid2">'
             + card("STEERING ANGLE", fmt_steer(steer), shift_cls)
             + card("EGO SPEED", format(tel["speed_kmh"], ".0f") + " km/h", "cyan")
             + card("EVASION WINDOW (s)", win_txt, win_cls)
             + card("TRACKED ENTITIES", ent_txt)
             + card("FREE-SPACE AVAILABILITY · " + fs_word, fs_txt, "", small=True, extra=fs_bar)
             + card("LATERAL SHIFT", lat_txt, shift_cls, small=True)
             + card("LATERAL DEMAND · req / envelope", g_txt, g_cls, small=True, extra=g_bar)
             + card("STEER · req / max", steer_txt, "", small=True)
             + card("ROAD FRICTION μ", format(tel["road_mu"], ".2f"), "", small=True)
             + card("LIVE FPS", format(tel.get("fps", 0.0), ".1f"), "good")
             + "</div>")
    panel.markdown(block, unsafe_allow_html=True)
    render_terminal()


def render_perf(tel, ui_ms):
    txt = (str(tel.get("resolution", "")) + " native · display " + format(st.session_state.ui_fps, ".1f")
           + " fps · planner " + str(tel.get("fps", 0)) + " fps · perception every " + str(tel.get("infer_every", 1))
           + "f @ " + str(tel.get("imgsz", "")) + "px " + str(tel.get("infer_ms", 0)) + " ms · frame "
           + str(tel.get("pipeline_ms", 0)) + " ms · ui " + format(ui_ms, ".0f") + " ms · buffer "
           + str(tel.get("buffer", 0)) + "/" + str(engine.BUFFER_FRAMES) + " · jpeg " + str(tel.get("jpeg_kb", 0))
           + " KB · skipped " + str(tel.get("skipped", 0)) + " · ground-plane " + str(tel.get("ground_plane_near_m", "—")) + " m")
    perf.markdown('<div class="perf">' + html.escape(txt) + "</div>", unsafe_allow_html=True)


# ---------------- Action log ----------------
def emit_actions(tel):
    s = st.session_state
    state, reason, primary, now = tel["state"], tel.get("reason"), tel.get("primary"), time.time()
    key = (primary["class"], tel["direction"]) if primary else None
    changed = state != s.prev_state or reason != s.prev_reason or key != s.prev_primary
    k = tel.get("kinematics") or {}
    win = primary["window_s"] if (primary and primary["window_s"] is not None) else None
    win_txt = format(win, ".1f") + "s" if win is not None else "open"

    if state == "TRAJECTORY SHIFT" and (changed or now - s.last_ev_log > 1.5):
        lat = tel["lateral_m"]
        term("shift", "[TRAJECTORY SHIFT] Computing swept path. Routing " + ("+" if lat > 0 else "")
             + format(lat, ".1f") + "m laterally.")
        if changed:
            term("plan", "[PLANNER] " + primary["class"] + " at " + str(primary["distance_m"]) + "m · window " + win_txt
                 + " · demand " + format(k.get("a_req_g", 0), ".2f") + "g of " + format(k.get("a_max_g", 0), ".2f")
                 + "g · steer " + fmt_steer(tel["steer_deg"]))
        s.last_ev_log = now
    elif state == "SAFE YIELD" and (changed or now - s.last_ev_log > 1.5):
        term("yield", "[SAFE YIELD] Evasion window closed. Engaging controlled deceleration.")
        if changed:
            if reason == "KINEMATIC":
                term("kin", "[KINEMATIC ENVELOPE] Lateral demand " + format(k.get("a_req_g", 0), ".2f") + "g exceeds "
                     + format(k.get("a_max_g", 0), ".2f") + "g at " + format(tel["speed_kmh"], ".0f") + " km/h · steer "
                     + format(k.get("steer_req_deg", 0), ".1f") + "° > " + format(k.get("steer_max_deg", 0), ".1f")
                     + "° max · stop distance " + format(k.get("brake_dist_m", 0), ".1f") + "m")
            else:
                term("kin", "[FREE-SPACE] Insufficient clearance beside " + primary["class"] + " at "
                     + str(primary["distance_m"]) + "m. Holding lane, yielding.")
        s.last_ev_log = now
    elif state == "MONITORING" and changed and s.prev_state in (None, "CRUISING"):
        term("plan", "[PLANNER] Entity in ego-path: " + primary["class"] + " · " + str(primary["distance_m"])
             + "m · window " + win_txt + ". Monitoring.")
    elif state == "CRUISING" and changed and s.prev_state in ("TRAJECTORY SHIFT", "SAFE YIELD"):
        term("plan", "[PLANNER] Ego-path clear. Swept path restored. Continuous flow resumed.")
    s.prev_state, s.prev_reason, s.prev_primary = state, reason, key


EMPTY = {"state": "DISENGAGED", "mode": "cruise", "reason": None, "direction": "NONE", "lateral_m": 0.0, "steer_deg": 0.0,
         "speed_kmh": 0.0, "road_mu": 0.7, "entities": 0, "hazards": 0, "window_s": None, "nearest_m": None, "primary": None,
         "free_left_m": None, "free_right_m": None, "free_availability": None, "kinematics": None,
         "infer_ms": 0.0, "pipeline_ms": 0.0, "fps": 0.0, "detections": []}


def idle_telemetry():
    t = dict(st.session_state.last_telemetry or EMPTY)
    t["state"] = "DISENGAGED"
    t["speed_kmh"], t["road_mu"] = float(speed_kmh), float(road_mu)     # mirror the current scenario controls
    return t


# ---------------- Replay mode (pre-rendered run; near-zero CPU) ----------------
if REPLAY:
    if get_worker() is not None:
        stop_worker()

    @st.cache_data(show_spinner=False)
    def load_replay(path):
        rows = [json.loads(line) for line in open(path) if line.strip()]
        return rows, (rows[-1]["t"] if rows else 0.0)

    rows, duration = load_replay(replay_log)
    feed.video(replay_mp4, autoplay=True, loop=True, muted=True)
    perf.markdown('<div class="perf">replay · recorded on a laptop with the same planner · ' + str(len(rows))
                  + " frames · " + format(duration, ".1f") + " s loop</div>", unsafe_allow_html=True)
    if "replay_t0" not in st.session_state:
        st.session_state.replay_t0 = time.time()
        term("sys", "[SYSTEM] Replay of recorded planner run: " + os.path.basename(replay_mp4))
    if rows:
        idx = 0
        while True:
            elapsed = (time.time() - st.session_state.replay_t0) % max(duration, 0.001)
            while idx + 1 < len(rows) and rows[idx + 1]["t"] <= elapsed:
                idx += 1
            if idx > 0 and rows[idx]["t"] > elapsed:
                idx = 0
            tel = rows[idx]
            st.session_state.steer += (tel["steer_deg"] - st.session_state.steer) * 0.3
            emit_actions(tel)
            render_panel(tel)
            time.sleep(1.0 / TELEMETRY_HZ)
    st.stop()

# ---------------- Idle / error ----------------
worker = get_worker()
if worker is None or not worker.running:
    if worker is not None and worker.error:
        term("err", "[ERROR] " + worker.error)
        feed.error(worker.error + "\n\nExpected clips next to app.py: highway.mp4 · city.mp4 · night.mp4 — or choose *Custom clip*.")
        stop_worker()
    else:
        feed.info("Planner disengaged. Scenario **" + scenario + "** armed — press **Engage** to start the spatial planner.")
    render_panel(idle_telemetry())
    st.stop()

# ---------------- Live consumer loop (buffered, paced at source FPS) ----------------
render_panel(st.session_state.last_telemetry or EMPTY)
worker.primed(min_frames=5, timeout=4.0)                 # let the buffer fill so the first seconds are smooth
period = 1.0 / max(worker.source_fps, 1.0)
next_t = time.perf_counter()
src_clock = next_t                      # where the source timeline says we should be
adv_ema = 1.0                           # smoothed source-frames-per-displayed-frame
last_draw, ui_fps, ui_t = 0.0, None, time.perf_counter()

while worker.running:
    item = worker.next_frame(timeout=1.0)              # FIFO: in order, nothing dropped by the UI
    if item is None:
        if worker.error:
            break
        next_t = src_clock = time.perf_counter()       # buffer ran dry: re-sync the clock instead of bursting
        continue
    t_ui = time.perf_counter()
    jpeg_bytes, tel = item
    st.session_state.last_telemetry = tel
    st.session_state.steer += (tel["steer_deg"] - st.session_state.steer) * 0.3
    emit_actions(tel)
    if jpeg_bytes is not None:
        # JPEG was encoded by OpenCV from the BGR canvas, so colours are already correct; the browser
        # decodes it directly. Overwriting the single st.empty slot releases the previous frame.
        feed.image(jpeg_bytes, use_container_width=True)
    now = time.perf_counter()
    dt = max(now - ui_t, 1e-6)
    ui_fps = (1.0 / dt) if ui_fps is None else 0.9 * ui_fps + 0.1 / dt
    ui_t = now
    st.session_state.ui_fps = ui_fps
    if now - last_draw > 1.0 / TELEMETRY_HZ:
        render_panel(tel)
        render_perf(tel, (time.perf_counter() - t_ui) * 1000.0)
        last_draw = now
    # Uniform display interval (smoothed frames-per-frame), gently corrected toward the true source
    # timeline so playback stays 1.00x without the 33/67 ms alternation integer decimation would cause.
    p = tel.get("period_s", period)
    adv = tel.get("advanced", 1)
    adv_ema += 0.08 * (adv - adv_ema)
    src_clock += p * adv
    next_t += p * adv_ema + 0.15 * (src_clock - next_t)
    delay = next_t - time.perf_counter()
    if delay > 0:
        time.sleep(delay)
    elif delay < -0.25:                                  # UI hiccup: re-sync rather than sprint
        next_t = src_clock = time.perf_counter()

if worker.error:
    term("err", "[ERROR] " + worker.error)
    feed.error(worker.error)
else:
    feed.info("Clip finished. Press **Engage** to run again.")
stop_worker()
render_panel(idle_telemetry())
