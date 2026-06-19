"""
Padel pipeline test on Kaggle.

Downloads a Premier Padel match clip, runs player tracking + TrackNetV3 ball
detection, saves an annotated video + JSON results to /kaggle/working/.
"""
import json
import os
import subprocess
import sys
import time

WORK = "/kaggle/working"
REPO = f"{WORK}/padel-scorekeeper"
TNV3 = f"{WORK}/TrackNetV3"

def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=check)

# ── 1. disable CUDA (Kaggle P100 sm_60 unsupported) ──────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ── 2. install deps ───────────────────────────────────────────────────────────
sh("pip -q install ultralytics supervision yt-dlp gdown")

# ── 3. clone repo ─────────────────────────────────────────────────────────────
if not os.path.isdir(REPO):
    sh(f"git clone https://github.com/sheikhBasit/padel-scorekeeper {REPO}")
else:
    sh(f"git -C {REPO} pull --ff-only")

sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, REPO)

# ── 4. TrackNetV3 ─────────────────────────────────────────────────────────────
print("\n[tracknet] setting up ...", flush=True)
sh(f"bash {REPO}/setup_tracknet.sh {TNV3}")
print("[tracknet] ready", flush=True)

# ── 5. download test video ────────────────────────────────────────────────────
VIDEO = f"{WORK}/padel_test.mp4"
if not os.path.exists(VIDEO):
    print("\n[video] downloading Premier Padel clip ...", flush=True)
    sh(f'yt-dlp -f "best[height<=480]" --match-filter "duration < 600" '
       f'-o "{VIDEO}" '
       f'"ytsearch1:premier padel highlights overhead 2024"')
    print(f"[video] saved to {VIDEO}", flush=True)

# ── 6. run pipeline ───────────────────────────────────────────────────────────
import cv2
import numpy as np

sys.path.insert(0, f"{REPO}/src")
from calibrate_court import CourtMapper, COURT_PTS_M
from stream_pipeline import StreamPipeline

# Court corners calibrated for this Premier Padel camera angle (640×360)
CORNERS = np.float32([
    [90,  82],   # TL far-left
    [555, 82],   # TR far-right
    [625, 305],  # BR near-right
    [12,  305],  # BL near-left
])

cap = cv2.VideoCapture(VIDEO)
W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"\n[video] {W}x{H} @ {FPS:.1f}fps, {TOTAL} frames ({TOTAL/FPS:.0f}s)", flush=True)

homography, _ = cv2.findHomography(CORNERS, COURT_PTS_M)
mapper = CourtMapper(homography, CORNERS, (W, H))

pipeline = StreamPipeline(
    mapper=mapper,
    fps=FPS,
    tracknet_repo=TNV3,
    tracknet_ckpt=f"{TNV3}/ckpts/TrackNet_best.pt",
    inpaint_ckpt=f"{TNV3}/ckpts/InpaintNet_best.pt",
    names={"A": "Team A", "B": "Team B"},
)

# Annotate colours
NEAR  = (38, 149, 255)
FAR   = (60, 220, 60)
BALL  = (0, 240, 255)

def annotate(frame, result):
    out = frame.copy()
    H_inv = np.linalg.inv(mapper.H)
    near_n = far_n = 0
    for p in result["players"]:
        color = NEAR if p["half"] == "near" else FAR
        if p["half"] == "near":
            near_n += 1; lbl = f"A{near_n}"
        else:
            far_n += 1;  lbl = f"B{far_n}"
        pt = cv2.perspectiveTransform(
            np.array([[[p["x_m"], p["y_m"]]]], dtype=np.float32), H_inv)[0][0]
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(out, (px, py), 18, (0,0,0), 5)
        cv2.circle(out, (px, py), 18, color, 2)
        cv2.putText(out, lbl, (px-12, py+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
        cv2.putText(out, lbl, (px-12, py+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    ball = result.get("ball")
    if ball:
        cv2.circle(out, (int(ball["x"]), int(ball["y"])), 9, BALL, -1)
        cv2.circle(out, (int(ball["x"]), int(ball["y"])), 11, (255,255,255), 1)
    spd = result.get("speed_kmh")
    if spd:
        cv2.putText(out, f"{spd:.0f} km/h", (10, out.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4)
        cv2.putText(out, f"{spd:.0f} km/h", (10, out.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, BALL, 2)
    sc = result["score"]
    txt = f"A {sc['pts_a']} ({sc['a']}) – ({sc['b']}) {sc['pts_b']} B"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(out, (0,0), (tw+16, th+16), (0,0,0), -1)
    cv2.putText(out, txt, (8, th+8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    return out

# Process 300 frames starting at frame 250 (past title cards)
START   = 250
N_TEST  = 300
OUT_VID = f"{WORK}/padel_annotated.mp4"
OUT_JSON= f"{WORK}/padel_results.json"

writer = cv2.VideoWriter(OUT_VID,
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         FPS, (W, H))
cap.set(cv2.CAP_PROP_POS_FRAMES, START)

all_results = []
players_found = balls_found = 0
t0 = time.time()

for i in range(N_TEST):
    ok, frame = cap.read()
    if not ok:
        break
    result = pipeline.process(frame, START + i)
    all_results.append(result)
    if result["players"]:
        players_found += 1
    if result["ball"]:
        balls_found += 1
    writer.write(annotate(frame, result))
    if (i+1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  frame {START+i:4d} | players={len(result['players'])} "
              f"ball={'YES' if result['ball'] else 'no '} "
              f"speed={result.get('speed_kmh') or '--':>6} | "
              f"{elapsed:.1f}s elapsed", flush=True)

cap.release()
writer.release()

with open(OUT_JSON, "w") as f:
    json.dump(all_results, f, indent=2)

n = len(all_results)
print(f"\n── Results ─────────────────────────────────")
print(f"Frames processed : {n}")
print(f"Player detected  : {players_found}/{n} ({100*players_found//n}%)")
print(f"Ball detected    : {balls_found}/{n} ({100*balls_found//n}%)")
print(f"Annotated video  : {OUT_VID}")
print(f"JSON results     : {OUT_JSON}")
print(f"Total time       : {time.time()-t0:.1f}s")

# Save 6 sample annotated frames as images for quick preview
cap2 = cv2.VideoCapture(OUT_VID)
for idx, fi in enumerate(range(0, n, n//6)):
    cap2.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, f = cap2.read()
    if ok:
        cv2.imwrite(f"{WORK}/sample_{idx:02d}_frame{START+fi}.jpg", f)
cap2.release()
print("Sample frames saved.")
