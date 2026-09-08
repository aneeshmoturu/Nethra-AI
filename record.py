"""
Nethra AI - offline recorder.
Runs the full planner on a clip and writes an annotated MP4 + telemetry JSONL that app.py can replay
with near-zero CPU (Replay mode). Run this on a laptop, commit the outputs under recordings/.

    python record.py city.mp4 --speed 30 --mu 0.70
    python record.py highway.mp4 --speed 85 --mu 0.70
    python record.py night.mp4 --speed 45 --mu 0.45 --enhance
"""

import argparse
import json
import os

import cv2

import engine

ap = argparse.ArgumentParser()
ap.add_argument("clip")
ap.add_argument("--speed", type=float, default=45.0)
ap.add_argument("--mu", type=float, default=engine.ROAD_MU_DRY)
ap.add_argument("--enhance", action="store_true")
ap.add_argument("--critical", type=float, default=engine.CRITICAL_DISTANCE)
ap.add_argument("--infer-every", type=int, default=2)
ap.add_argument("--imgsz", type=int, default=480)
ap.add_argument("--render-h", type=int, default=720, help="0 = native")
ap.add_argument("--out", default="recordings")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
stem = os.path.splitext(os.path.basename(args.clip))[0]
mp4_path = os.path.join(args.out, stem + ".mp4")
log_path = os.path.join(args.out, stem + ".jsonl")

model = engine.load_model()
planner = engine.SpatialPlanner(model)
cap = cv2.VideoCapture(args.clip)
if not cap.isOpened():
    raise SystemExit("Could not open " + args.clip)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
writer, n = None, 0
with open(log_path, "w") as log:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        canvas, tel = planner.process(frame, args.speed, args.mu, critical_distance=args.critical, enhance=args.enhance,
                                      infer_every=args.infer_every, imgsz=args.imgsz, render_max_height=args.render_h)
        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
            if not writer.isOpened():                       # fall back if no H.264 encoder is available
                writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(canvas)
        tel["t"] = round(n / fps, 3)
        tel.pop("detections", None)
        log.write(json.dumps(tel) + "\n")
        n += 1
        if n % 60 == 0:
            print("frame", n, tel["state"])
cap.release(); writer.release()
print("wrote", mp4_path, "and", log_path, "-", n, "frames at", round(fps, 1), "fps")
print("NOTE: if the browser won't play the mp4, re-encode once:  ffmpeg -i", mp4_path, "-c:v libx264 -pix_fmt yuv420p -movflags +faststart", mp4_path.replace(".mp4", "_web.mp4"))
