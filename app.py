"""
Nethra AI - Spatial Planner Console (Scenario Library + Kinematic telemetry)
UI thread only: dequeues pre-rendered JPEG frames + telemetry from the background brain.

Run:  streamlit run app.py
Expected clips next to this file: highway.mp4, city.mp4, night.mp4 (or pick "Custom clip").
"""

import html
import os
import time
from collections import deque

import streamlit as st

import engine

st.set_page_config(page_title="Nethra AI | Spatial Planner", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------- Engine handshake ----------------
REQUIRED_ENGINE = "6.0"
_loaded = getattr(engine, "ENGINE_VERSION", None)
if _loaded != REQUIRED_ENGINE:
    st.error(
        f"**engine.py mismatch.** app.py needs engine v{REQUIRED_ENGINE}, but Python loaded "
        f"**{'v' + _loaded if _loaded else 'a pre-6.0 engine'}** from:\n\n`{engine.__file__}`\n\n"
        "Fix: overwrite that file with the matching engine.py, delete the `__pycache__` folder next to it, "
        "then stop and restart `streamlit run app.py`."
    )
    st.stop()

# ---------------- Scenario library ----------------
SCENARIOS = {
    "Highway - High Speed (Rollover Risk)": {"file": "highway.mp4", "speed": 85, "mu": engine.ROAD_MU_DRY, "enhance": False,
                                             "blurb": "Dry highway at 85 km/h. Narrow lateral envelope: the planner favours controlled yields over sharp shifts."},
    "City - Low Speed (Safe Evasion)":     {"file": "city.mp4", "speed": 30, "mu": engine.ROAD_MU_DRY, "enhance": False,
                                             "blurb": "Urban traffic at 30 km/h. Wide envelope: smooth swept-path trajectory shifts around entities."},
    "Night/Rain - Low Visibility":         {"file": "night.mp4", "speed": 45, "mu": engine.ROAD_MU_WET, "enhance": True,
                                             "blurb": "Wet road (mu 0.45) with low-light enhancement on the perception input. Envelope tightens with grip."},
    "Custom clip":                         {"file": "", "speed": 45, "mu": engine.ROAD_MU_DRY, "enhance": False, "blurb": ""},
}

# ---------------- Styling ----------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', -apple-system, system-ui, sans-serif; }
  .block-container { padding-top: 0.6rem; padding-bottom: 0.3rem; max-width: 100%; }

  /* ---- Ambient background: deep navy base, drifting cyan/violet auroras, faint engineering grid ---- */
  .stApp {
      background:
        radial-gradient(1200px 700px at 8% -10%, rgba(0,229,255,0.16), transparent 60%),
        radial-gradient(900px 600px at 100% 0%, rgba(123,47,255,0.20), transparent 60%),
        radial-gradient(1000px 800px at 50% 120%, rgba(0,180,140,0.12), transparent 60%),
        linear-gradient(180deg, #070B12 0%, #0A0F18 55%, #060910 100%);
      background-attachment: fixed;
  }
  .stApp::before {
      content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
      background-image:
        linear-gradient(rgba(120,160,200,0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(120,160,200,0.045) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: radial-gradient(ellipse at 50% 30%, rgba(0,0,0,0.9), transparent 75%);
      -webkit-mask-image: radial-gradient(ellipse at 50% 30%, rgba(0,0,0,0.9), transparent 75%);
  }
  .stApp::after {
      content:""; position:fixed; inset:-20%; pointer-events:none; z-index:0;
      background:
        radial-gradient(600px 400px at 20% 30%, rgba(0,229,255,0.10), transparent 65%),
        radial-gradient(700px 500px at 80% 70%, rgba(123,47,255,0.12), transparent 65%);
      filter: blur(40px);
      animation: drift 26s ease-in-out infinite alternate;
  }
  @keyframes drift { from { transform: translate3d(-3%, -2%, 0) rotate(0deg); } to { transform: translate3d(3%, 3%, 0) rotate(4deg); } }
  .stApp > * { position: relative; z-index: 1; }
  header[data-testid="stHeader"] { background: transparent; }
  section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, rgba(10,14,22,0.92), rgba(7,10,16,0.96));
      border-right: 1px solid rgba(0,229,255,0.10);
      backdrop-filter: blur(14px);
      box-shadow: inset -1px 0 0 rgba(255,255,255,0.02), 8px 0 40px rgba(0,0,0,0.35);
  }
  section[data-testid="stSidebar"]::before {
      content:""; position:absolute; inset:0; pointer-events:none;
      background: radial-gradient(400px 300px at 0% 0%, rgba(0,229,255,0.10), transparent 70%);
  }
  .brand { display:flex; align-items:center; gap:14px; margin-bottom:4px; }
  .brand .logo { width:34px; height:34px; border-radius:9px; background: conic-gradient(from 200deg, #00E5FF, #7B2FFF, #00E5FF);
                 box-shadow: 0 0 22px rgba(0,229,255,0.35); }
  .brand h1 { font-size:1.35rem; font-weight:700; letter-spacing:0.14em; color:#EAF2FF; margin:0; }
  .brand .sub { color:#5C6B7C; font-size:0.72rem; letter-spacing:0.14em; margin-top:2px; }
  .mono { font-family:'JetBrains Mono', ui-monospace, Menlo, monospace; }
  h3 { margin: 0.55rem 0 0.35rem 0; letter-spacing:0.14em; font-size:0.64rem; color:#5C6B7C; font-weight:600; }
  .state { text-align:center; padding:12px 10px; border-radius:10px; font-weight:700; letter-spacing:0.16em; font-size:0.86rem; margin-bottom:8px; }
  .st-cruise  { background:linear-gradient(180deg,#062A2E,#04191C); color:#00E5FF; border:1px solid #0E6E7A; box-shadow:0 0 18px rgba(0,229,255,0.18); }
  .st-monitor { background:linear-gradient(180deg,#2E2208,#1B1404); color:#FFB020; border:1px solid #8A6010; }
  .st-shift   { background:linear-gradient(180deg,#07301A,#04190E); color:#39FF14; border:1px solid #1E8A3A; box-shadow:0 0 22px rgba(57,255,20,0.22); animation: pulse 1.1s ease-in-out infinite; }
  .st-yield   { background:linear-gradient(180deg,#332208,#1C1204); color:#FFB020; border:1px solid #A87010; box-shadow:0 0 22px rgba(255,176,32,0.28); animation: pulse 1.4s ease-in-out infinite; }
  .st-idle    { background:#0F151C; color:#5C6B7C; border:1px solid #1E2A36; }
  @keyframes blink { 50% { opacity: 0.45; } }
  @keyframes pulse { 50% { filter: brightness(1.18); } }
  .matrix { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin-bottom:8px; }
  .cell { text-align:center; padding:7px 2px; border-radius:7px; font-size:0.6rem; letter-spacing:0.1em; font-weight:600;
          color:#3E4C5A; background:rgba(11,16,22,0.75); border:1px solid rgba(120,160,200,0.12); backdrop-filter: blur(8px); }
  .cell.on-cruise { color:#00E5FF; border-color:#0E6E7A; background:#062A2E; }
  .cell.on-monitor { color:#FFB020; border-color:#8A6010; background:#2E2208; }
  .cell.on-shift { color:#39FF14; border-color:#1E8A3A; background:#07301A; }
  .cell.on-yield { color:#FFB020; border-color:#A87010; background:#332208; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-bottom:6px; }
  .card { background:linear-gradient(180deg, rgba(20,28,40,0.72), rgba(12,18,26,0.78)); border:1px solid rgba(120,160,200,0.14);
          border-radius:10px; padding:8px 10px; backdrop-filter: blur(10px);
          box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.28); }
  .card:hover { border-color: rgba(0,229,255,0.28); }
  .card .k { color:#5C6B7C; font-size:0.58rem; letter-spacing:0.12em; }
  .card .v { color:#EAF2FF; font-size:1.12rem; font-weight:600; margin-top:1px; }
  .card .v.small { font-size:0.9rem; }
  .card.warn .v { color:#FFB020; } .card.good .v { color:#39FF14; } .card.cyan .v { color:#00E5FF; }
  .bar { position:relative; height:6px; background:#0B1016; border-radius:3px; margin-top:6px; border:1px solid #141C25; }
  .bar .fill { position:absolute; left:0; top:0; height:100%; border-radius:3px; background:linear-gradient(90deg,#00E5FF,#39FF14); }
  .bar .fill.warn { background:linear-gradient(90deg,#FFB020,#FF8A20); }
  .bar .lim { position:absolute; top:-4px; width:2px; height:14px; background:#FFB020; }
  .term { background:linear-gradient(180deg, rgba(5,8,12,0.92), rgba(3,5,8,0.96)); border:1px solid rgba(0,229,255,0.12); border-radius:10px; padding:10px 12px;
          box-shadow: 0 0 30px rgba(0,229,255,0.05) inset, 0 8px 24px rgba(0,0,0,0.35);
          font-family:'JetBrains Mono', ui-monospace, Menlo, monospace; font-size:0.66rem; line-height:1.6;
          height: 232px; overflow:hidden; white-space:pre; color:#9FB3C8; }
  .t-sys{color:#00E5FF} .t-shift{color:#39FF14;font-weight:600} .t-plan{color:#8B9AAB} .t-kin{color:#FFB020} .t-yield{color:#FFB020;font-weight:600} .t-err{color:#FF8A20} .t-dim{color:#3E4C5A}
  .perf { color:#3E4C5A; font-size:0.62rem; letter-spacing:0.06em; font-family:'JetBrains Mono', monospace; margin-top:4px; }
  .scen { background:rgba(11,16,22,0.7); border:1px solid rgba(123,47,255,0.25); border-left:2px solid #7B2FFF; border-radius:8px; padding:8px 10px; font-size:0.7rem; color:#8B9AAB; margin:6px 0 4px 0; backdrop-filter: blur(8px); }
  div[data-testid="stImage"] img { border-radius:14px; border:1px solid rgba(0,229,255,0.16);
        box-shadow: 0 0 0 1px rgba(255,255,255,0.03), 0 24px 60px rgba(0,0,0,0.55), 0 0 60px rgba(0,229,255,0.10); }
  .brand .logo { box-shadow: 0 0 28px rgba(0,229,255,0.45), 0 0 60px rgba(123,47,255,0.35); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand"><div class="logo"></div>
<div><h1>NETHRA AI</h1><div class="sub">AUTONOMOUS SPATIAL PLANNER · PURE-VISION · SWEPT-PATH ROUTING · KINEMATIC ENVELOPE</div></div></div>
""", unsafe_allow_html=True)

# ---------------- Worker container + state ----------------
@st.cache_resource(show_spinner=False)
def worker_container():
    return {"worker": None, "scenario": None}


@st.cache_resource(show_spinner="Loading YOLOv8 weights...")
def get_model():
    return engine.load_model()


for k, v in {"last_telemetry": None, "term": deque(maxlen=15), "steer": 0.0, "prev_state": None,
             "prev_override": None, "prev_primary": None, "last_ev_log": 0.0, "scenario_prev": None}.items():
    st.session_state.setdefault(k, v)
TELEMETRY_HZ = 8


def term(kind, msg):
    st.session_state.term.append((kind, f"{time.strftime('%H:%M:%S')} > {msg}"))


def get_worker():
    return worker_container()["worker"]


def stop_worker():
    w = get_worker()
    if w is not None:
        w.stop()
        w.join(timeout=3.0)
    worker_container()["worker"] = None


def start_worker(path, speed, mu, enhance, critical, conf, loop):
    stop_worker()                                   # join old thread before the new one touches the model
    w = engine.VideoInferenceWorker(path, get_model(), speed_kmh=speed, road_mu=mu, critical_distance=critical,
                                    conf=conf, loop=loop, enhance=enhance)
    w.start()
    worker_container()["worker"] = w
    term("sys", f"[SYSTEM] Spatial planner online. Source={os.path.basename(path) or '?'}  mu={mu:.2f}  enhance={'on' if enhance else 'off'}")
    term("plan", "[PLANNER] Nominal swept path engaged. Continuous-flow mode.")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### SIMULATION SCENARIO")
    scenario = st.selectbox("Simulation Scenario", list(SCENARIOS), label_visibility="collapsed")
    cfg = SCENARIOS[scenario]
    scenario_changed = scenario != st.session_state.scenario_prev
    if scenario_changed:
        st.session_state["speed_kmh"] = float(cfg["speed"])      # seed slider default per scenario
        st.session_state.scenario_prev = scenario
    if cfg["blurb"]:
        st.markdown(f'<div class="scen">{cfg["blurb"]}</div>', unsafe_allow_html=True)
    video_path = st.text_input("Clip path", value="custom.mp4") if scenario == "Custom clip" else cfg["file"]

    st.markdown("### VEHICLE")
    speed_kmh = st.slider("Simulated Vehicle Speed (km/h)", 0.0, 140.0, step=1.0, key="speed_kmh")
    road_mu = st.slider("Road friction (mu)", 0.20, 0.90, float(cfg["mu"]), 0.05,
                        help="0.7 dry asphalt · 0.45 wet · 0.3 mud/gravel")

    st.markdown("### PERCEPTION")
    critical_distance = st.slider("Critical distance (m)", 5.0, 40.0, engine.CRITICAL_DISTANCE, 0.5)
    conf_threshold = st.slider("Detection confidence", 0.10, 0.90, engine.CONF_THRESHOLD, 0.05)
    enhance = st.checkbox("Low-light enhancement (CLAHE)", value=cfg["enhance"], key=f"enh_{scenario}")
    loop_video = st.checkbox("Loop clip", value=True)

    st.markdown("### CONTROL")
    c1, c2 = st.columns(2)
    engage = c1.button("▶ Engage", type="primary", use_container_width=True)
    disengage = c2.button("■ Disengage", use_container_width=True)
    reset = st.button("↺ Restart clip", use_container_width=True)
    st.caption(f"{engine.DEVICE.upper()} · `{engine.MODEL_PATH}` + ByteTrack · {engine.INFER_SIZE[0]}×{engine.INFER_SIZE[1]} → {engine.DISPLAY_SIZE[0]}×{engine.DISPLAY_SIZE[1]} · BEV {engine.BEV_D_MAX:.0f} m")
    st.caption(f"engine v{engine.ENGINE_VERSION} · `{os.path.basename(engine.__file__)}`")

# Scenario switch while running -> clean restart on the new clip
worker = get_worker()
if engage:
    start_worker(video_path, speed_kmh, road_mu, enhance, critical_distance, conf_threshold, loop_video)
elif worker is not None and worker.running and worker_container()["scenario"] != scenario:
    term("sys", f"[SYSTEM] Scenario switch → {scenario}. Restarting stream.")
    start_worker(video_path, speed_kmh, road_mu, enhance, critical_distance, conf_threshold, loop_video)
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

# ---------------- Layout ----------------
col_feed, col_panel = st.columns([2.9, 1.25])
with col_feed:
    feed = st.empty()
    perf = st.empty()
with col_panel:
    panel = st.empty()
    st.markdown("### AUTONOMOUS ACTION TERMINAL")
    term_box = st.empty()


# ---------------- Renderers ----------------
def fmt_steer(deg):
    if abs(deg) < 0.5:
        return "0.0° · centre"
    return f"{'+' if deg > 0 else '-'}{abs(deg):.1f}° {'Right' if deg > 0 else 'Left'}"


def render_terminal():
    lines = list(st.session_state.term)
    body = ('<span class="t-dim">terminal idle — engage autonomy to begin</span>' if not lines else
            "\n".join(f'<span class="t-{k}">{html.escape(t)}</span>' for k, t in lines))
    term_box.markdown(f'<div class="term">{body}</div>', unsafe_allow_html=True)


def render_panel(tel):
    state = tel["state"]
    cls = {"CRUISING": "cruise", "MONITORING": "monitor", "TRAJECTORY SHIFT": "shift", "SAFE YIELD": "yield"}.get(state, "idle")
    label = {"CRUISING": "● CRUISING · CONTINUOUS FLOW", "MONITORING": "● MONITORING ENTITIES",
             "TRAJECTORY SHIFT": "● TRAJECTORY SHIFT · SWEPT PATH", "SAFE YIELD": "● SAFE YIELD · CONTROLLED DECEL"}.get(state, "○ PLANNER DISENGAGED")
    cells = "".join(f'<div class="cell {"on-" + c if c == cls else ""}">{n}</div>'
                    for c, n in (("cruise", "CRUISE"), ("monitor", "MONITOR"), ("shift", "SHIFT"), ("yield", "YIELD")))
    k = tel.get("kinematics") or {}
    win = tel.get("window_s")
    win_cls = "warn" if (win is not None and win < engine.WINDOW_ACT) else "good" if win is not None else ""
    steer = st.session_state.steer
    lat = tel.get("lateral_m", 0.0)
    lat_txt = f"{'+' if lat > 0 else ''}{lat:.2f} m" if state in ("TRAJECTORY SHIFT", "SAFE YIELD") else "0.00 m"

    lat_req, lat_max = k.get("a_req_g"), k.get("a_max_g")
    if lat_req is not None:
        fill = min(lat_req / 1.0, 1.0) * 100; lim = min(lat_max / 1.0, 1.0) * 100
        gbar = (f'<div class="bar"><div class="fill {"warn" if not k["feasible"] else ""}" style="width:{fill:.0f}%"></div>'
                f'<div class="lim" style="left:{lim:.0f}%"></div></div>')
        g_txt = f"{lat_req:.2f} / {lat_max:.2f} g"
    else:
        gbar, g_txt = '<div class="bar"></div>', "—"

    avail = tel.get("free_availability")
    if avail is not None:
        fl, fr = tel["free_left_m"], tel["free_right_m"]
        avail_txt = f"L {fl:.1f} m · R {fr:.1f} m"
        abar = f'<div class="bar"><div class="fill {"warn" if avail < 0.5 else ""}" style="width:{avail * 100:.0f}%"></div></div>'
        avail_word = "HIGH" if avail >= 0.99 else "PARTIAL" if avail >= 0.5 else "LOW"
    else:
        avail_txt, abar, avail_word = "unconstrained", '<div class="bar"><div class="fill" style="width:100%"></div></div>', "HIGH"

    html_block = f"""
<div class="state st-{cls}">{label}</div>
<div class="matrix">{cells}</div>
<div class="grid2">
  <div class="card {'good' if state == 'TRAJECTORY SHIFT' else ''}"><div class="k">STEERING ANGLE</div><div class="v mono">{fmt_steer(steer)}</div></div>
  <div class="card cyan"><div class="k">EGO SPEED</div><div class="v mono">{tel['speed_kmh']:.0f} km/h</div></div>
  <div class="card {win_cls}"><div class="k">EVASION WINDOW (s)</div><div class="v mono">{f"{win:.2f} s" if win is not None else "open"}</div></div>
  <div class="card"><div class="k">TRACKED ENTITIES</div><div class="v mono">{tel['entities']}<span style="font-size:0.7rem;color:#5C6B7C"> · {tel['hazards']} in ego-path</span></div></div>
  <div class="card"><div class="k">FREE-SPACE AVAILABILITY · {avail_word}</div><div class="v mono small">{avail_txt}</div>{abar}</div>
  <div class="card {'good' if state == 'TRAJECTORY SHIFT' else ''}"><div class="k">LATERAL SHIFT</div><div class="v mono small">{lat_txt}</div></div>
  <div class="card {'warn' if (k and not k.get('feasible', True)) else ''}"><div class="k">LATERAL DEMAND · req / envelope</div><div class="v mono small">{g_txt}</div>{gbar}</div>
  <div class="card"><div class="k">STEER · req / max</div><div class="v mono small">{f"{k['steer_req_deg']:.1f}° / {k['steer_max_deg']:.1f}°" if k else "—"}</div></div>
  <div class="card good"><div class="k">LIVE FPS</div><div class="v mono">{tel.get('fps', 0.0)}</div></div>
  <div class="card"><div class="k">INFERENCE</div><div class="v mono">{tel.get('infer_ms', 0.0)} ms</div></div>
</div>"""
    panel.markdown(html_block, unsafe_allow_html=True)
    render_terminal()


def render_perf(tel, ui_ms):
    perf.markdown(f'<div class="perf">planner {tel.get("pipeline_ms", 0)} ms/frame · inference {tel.get("infer_ms", 0)} ms · ui {ui_ms:.0f} ms · '
                  f'frame {tel.get("frame_idx", 0)} · src {tel.get("source_fps", 0)} fps · dropped {tel.get("dropped", 0)} · skipped {tel.get("skipped", 0)} · ground-plane {tel.get("ground_plane_near_m", "—")} m</div>',
                  unsafe_allow_html=True)


# ---------------- Action log ----------------
def emit_actions(tel):
    s = st.session_state
    state, reason, primary, now = tel["state"], tel.get("reason"), tel.get("primary"), time.time()
    key = (primary["class"], tel["direction"]) if primary else None
    changed = state != s.prev_state or reason != s.prev_override or key != s.prev_primary
    k = tel.get("kinematics") or {}
    win = primary["window_s"] if primary and primary["window_s"] is not None else None
    win_txt = f"{win:.1f}s" if win is not None else "open"

    if state == "TRAJECTORY SHIFT" and (changed or now - s.last_ev_log > 1.5):
        lat = tel["lateral_m"]
        term("shift", f"[TRAJECTORY SHIFT] Computing swept path. Routing {'+' if lat > 0 else ''}{lat:.1f}m laterally.")
        if changed:
            term("plan", f"[PLANNER] {primary['class']} at {primary['distance_m']}m · window {win_txt} · "
                         f"demand {k.get('a_req_g', 0):.2f}g of {k.get('a_max_g', 0):.2f}g · steer {fmt_steer(tel['steer_deg'])}")
        s.last_ev_log = now
    elif state == "SAFE YIELD" and (changed or now - s.last_ev_log > 1.5):
        term("yield", "[SAFE YIELD] Evasion window closed. Engaging controlled deceleration.")
        if changed:
            if reason == "KINEMATIC":
                term("kin", f"[KINEMATIC ENVELOPE] Lateral demand {k.get('a_req_g', 0):.2f}g exceeds {k.get('a_max_g', 0):.2f}g "
                            f"at {tel['speed_kmh']:.0f} km/h · steer {k.get('steer_req_deg', 0):.1f}° > {k.get('steer_max_deg', 0):.1f}° max · "
                            f"stop distance {k.get('brake_dist_m', 0):.1f}m")
            else:
                term("kin", f"[FREE-SPACE] Insufficient clearance beside {primary['class']} at {primary['distance_m']}m. Holding lane, yielding.")
        s.last_ev_log = now
    elif state == "MONITORING" and changed and s.prev_state in (None, "CRUISING"):
        term("plan", f"[PLANNER] Entity in ego-path: {primary['class']} · {primary['distance_m']}m · window {win_txt}. Monitoring.")
    elif state == "CRUISING" and changed and s.prev_state in ("TRAJECTORY SHIFT", "SAFE YIELD"):
        term("plan", "[PLANNER] Ego-path clear. Swept path restored. Continuous flow resumed.")
    s.prev_state, s.prev_override, s.prev_primary = state, reason, key


EMPTY = {"state": "DISENGAGED", "mode": "cruise", "reason": None, "direction": "NONE", "lateral_m": 0.0, "steer_deg": 0.0,
         "speed_kmh": 0.0, "road_mu": 0.7, "entities": 0, "hazards": 0, "window_s": None, "nearest_m": None, "primary": None,
         "free_left_m": None, "free_right_m": None, "free_availability": None, "kinematics": None,
         "infer_ms": 0.0, "pipeline_ms": 0.0, "fps": 0.0, "detections": []}

worker = get_worker()
if worker is None or not worker.running:
    if worker is not None and worker.error:
        term("err", f"[ERROR] {worker.error}")
        feed.error(f"{worker.error}\n\nExpected clips next to app.py: highway.mp4 · city.mp4 · night.mp4 — or choose *Custom clip*.")
        stop_worker()
    else:
        feed.info(f"Planner disengaged. Scenario **{scenario}** armed — press **Engage** to start the spatial planner.")
    idle = dict(st.session_state.last_telemetry or EMPTY); idle["state"] = "DISENGAGED"
    render_panel(idle)
    st.stop()

# ---------------- Live consumer loop ----------------
render_panel(st.session_state.last_telemetry or EMPTY)
last_draw = 0.0
while worker.running:
    item = worker.latest(timeout=1.0)
    if item is None:
        if worker.error:
            break
        continue
    t_ui = time.perf_counter()
    jpeg_bytes, tel = item
    st.session_state.last_telemetry = tel
    st.session_state.steer += (tel["steer_deg"] - st.session_state.steer) * 0.3
    emit_actions(tel)
    if jpeg_bytes is not None:
        feed.image(jpeg_bytes, use_container_width=True)
    now = time.perf_counter()
    if now - last_draw > 1.0 / TELEMETRY_HZ:
        render_panel(tel)
        render_perf(tel, (time.perf_counter() - t_ui) * 1000.0)
        last_draw = now

if worker.error:
    term("err", f"[ERROR] {worker.error}")
    feed.error(worker.error)
else:
    feed.info("Clip finished. Press **Engage** to run again.")
stop_worker()
idle = dict(st.session_state.last_telemetry or EMPTY); idle["state"] = "DISENGAGED"
render_panel(idle)