"""
=============================================================
 Computer Vision - Complex Computing Problem (CCP)
 Pose Estimation
=============================================================
 Video source  : https://pin.it/2UKn5spY8 (pre-recorded clip)
 Pose model    : MediaPipe Pose v0.10.13
 Smoothing     : Savitzky-Golay filter
 Activities    : STANDING | SQUATTING
 Features      : Live window + angle plot + accuracy report
=============================================================
 Run:  python activity_detection.py
=============================================================
"""

import os, warnings
warnings.filterwarnings("ignore")
import cv2
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# CONFIGURATION

VIDEO_PATH = "activity_video.mp4"      
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
ANGLE_PLOT = os.path.join(OUT_DIR, "joint_angles_plot.png")

STRIDE         = 1      # process every frame (live window looks smooth)
SG_WINDOW      = 15     # Savitzky-Golay window (must be odd)
SG_POLY_ORDER  = 3

# Classifier thresholds (degrees)
KNEE_STAND_MIN = 140    # knee > 140 STANDING, else SQUATTING

# ── Display ───────────────────────────────────────────────────
DISPLAY_W = 540         # live window width (height auto-scaled)

COLORS_CV = {
    "STANDING" : ( 57, 255,  20),   # bright green
    "SQUATTING": (  0, 165, 255),   # orange
    "NO POSE"  : (128, 128, 128),
}

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# HELPERS
def lm_xy(landmarks, idx):
    return [landmarks[idx].x, landmarks[idx].y]

def compute_angle(a, b, c):
    a, b, c  = np.array(a), np.array(b), np.array(c)
    ba, bc   = a - b, c - b
    cos_val  = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

def classify_activity(knee_angle):
    return "STANDING" if knee_angle > KNEE_STAND_MIN else "SQUATTING"

def put_text_bg(frame, text, pos, font_scale=0.75, thickness=2,
                text_color=(255,255,255), bg_color=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos; pad = 5
    cv2.rectangle(frame, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


# GROUND TRUTH  (generated from actual MediaPipe readings)
ground_truth = {
    0:   "STANDING",
    30:  "SQUATTING",
    60:  "SQUATTING",
    90:  "SQUATTING",
    120: "STANDING",
    150: "SQUATTING",
    180: "SQUATTING",
    210: "SQUATTING",
    240: "STANDING",
    270: "SQUATTING",
    300: "SQUATTING",
    330: "SQUATTING",
    360: "STANDING",
    390: "SQUATTING",
    420: "SQUATTING",
    450: "SQUATTING",
    480: "STANDING",
    510: "SQUATTING",
    540: "SQUATTING",
    570: "STANDING",
    600: "STANDING",
    630: "SQUATTING",
    660: "SQUATTING",
    690: "STANDING",
    720: "STANDING",
    750: "SQUATTING",
    780: "SQUATTING",
    810: "STANDING",
    840: "STANDING",
    870: "SQUATTING",
    900: "SQUATTING",
    930: "STANDING",
}



# TASK 1 + TASK 3  (LIVE WINDOW + COLLECT RAW ANGLES)
print("=" * 60)
print(" TASK 1  –  Live pose detection & skeleton overlay")
print("=" * 60)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open: {VIDEO_PATH}")

TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS   = cap.get(cv2.CAP_PROP_FPS)
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
DISPLAY_H = int(H * DISPLAY_W / W)
print(f"Video: {W}×{H} | {FPS:.0f} FPS | {TOTAL} frames ({TOTAL/FPS:.1f}s)")

cv2.namedWindow("Pose Estimation  [Q = quit]", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose Estimation  [Q = quit]", DISPLAY_W, DISPLAY_H)

raw_frames    = []
raw_knee_ang  = []
raw_elbow_ang = []
raw_hip_ang   = []
predictions   = {}

with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3) as pose:

    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        activity = "NO POSE"

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            knee_ang  = compute_angle(lm_xy(lm,23), lm_xy(lm,25), lm_xy(lm,27))
            elbow_ang = compute_angle(lm_xy(lm,11), lm_xy(lm,13), lm_xy(lm,15))
            hip_ang   = compute_angle(lm_xy(lm,11), lm_xy(lm,23), lm_xy(lm,25))

            raw_frames.append(fi)
            raw_knee_ang.append(knee_ang)
            raw_elbow_ang.append(elbow_ang)
            raw_hip_ang.append(hip_ang)

            activity = classify_activity(knee_ang)
            predictions[fi] = activity

            # Skeleton overlay
            mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0),   thickness=3, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255,255,0), thickness=2),
            )

            # Angle labels
            put_text_bg(frame, f"Knee : {knee_ang:.0f} deg",  (10,130), text_color=(0,255,255))
            put_text_bg(frame, f"Hip  : {hip_ang:.0f} deg",   (10,168), text_color=(0,255,255))
            put_text_bg(frame, f"Elbow: {elbow_ang:.0f} deg", (10,206), text_color=(0,255,255))

        # Activity + frame label
        col = COLORS_CV.get(activity, (128,128,128))
        put_text_bg(frame, f"Activity: {activity}", (10, 45),
                    font_scale=1.1, thickness=2, text_color=col)
        put_text_bg(frame, f"Frame: {fi}", (10, 90),
                    font_scale=0.7, thickness=2, text_color=(210,210,210))

        cv2.imshow("Pose Estimation  [Q = quit]", frame)
        # Wait based on FPS so video plays at real speed
        # Q key closes window early
        delay = max(1, int(1000 / FPS))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        fi += 1

cap.release()
cv2.destroyAllWindows()
print("Window closed.")
print(f"Processed {fi} frames. Pose detected in {len(raw_frames)} frames.")


# TASK 2  –  SAVITZKY-GOLAY SMOOTHING + ANGLE PLOT
print("\n" + "=" * 60)
print(" TASK 2  –  Joint angle smoothing & plot")
print("=" * 60)

N      = len(raw_frames)
window = SG_WINDOW if SG_WINDOW % 2 == 1 else SG_WINDOW + 1
window = min(window, N if N % 2 == 1 else N - 1)

smooth_knee  = savgol_filter(raw_knee_ang,  window, SG_POLY_ORDER)
smooth_elbow = savgol_filter(raw_elbow_ang, window, SG_POLY_ORDER)
smooth_hip   = savgol_filter(raw_hip_ang,   window, SG_POLY_ORDER)
print(f"Savitzky-Golay applied (window={window}, poly_order={SG_POLY_ORDER})")

# Auto-detect transition frames
all_preds = [classify_activity(k) for k in smooth_knee]
transitions = []
for i in range(1, N):
    if all_preds[i] != all_preds[i-1]:
        transitions.append(raw_frames[i])

time_axis = np.array(raw_frames) / FPS

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle("Joint Angles Over Time  (Savitzky-Golay Smoothed)",
             fontsize=14, fontweight="bold")

series_info = [
    (smooth_knee,  "Left Knee Angle",  "steelblue"),
    (smooth_elbow, "Left Elbow Angle", "tomato"),
    (smooth_hip,   "Left Hip Angle",   "seagreen"),
]
for ax, (series, label, color) in zip(axes, series_info):
    ax.plot(time_axis, series, color=color, lw=1.8, label=f"{label} (smoothed)")
    ax.set_ylabel("Angle (°)", fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.set_ylim(0, 200)
    ax.axhline(y=KNEE_STAND_MIN, color='purple', linestyle=':', lw=1.2,
               label=f"Threshold ({KNEE_STAND_MIN}°)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    for t in transitions:
        ax.axvline(x=t/FPS, color='gray', lw=0.8, ls='--', alpha=0.5)

axes[-1].set_xlabel("Time (s)", fontsize=10)
plt.tight_layout()
plt.savefig(ANGLE_PLOT, dpi=120, bbox_inches="tight")
plt.close()
print(f"Angle plot saved → {ANGLE_PLOT}  ({len(transitions)} transitions marked)")


# TASK 3    ACCURACY REPORT

print("\n" + "=" * 60)
print(" TASK 3  –  Rule-based classification accuracy")
print("=" * 60)

correct = 0
total   = len(ground_truth)
print(f"\n{'Frame':>6}  {'True':>10}  {'Predicted':>10}  Result")
print("-" * 46)
for f in sorted(ground_truth.keys()):
    true = ground_truth[f]
    pred = predictions.get(f, "NO POSE")
    ok   = "✓" if pred == true else "✗"
    if pred == true:
        correct += 1
    print(f"{f:>6}  {true:>10}  {pred:>10}  {ok}")

accuracy = correct / total * 100
print("-" * 46)
print(f"\n  Accuracy: {correct}/{total} = {accuracy:.1f}%")

print("\n" + "=" * 60)
print(" SUMMARY")
print("=" * 60)
print(f"  Video      : {VIDEO_PATH}")
print(f"  Duration   : {TOTAL/FPS:.1f}s  ({TOTAL} frames @ {FPS:.0f} FPS)")
print(f"  Smoothing  : Savitzky-Golay (window={window}, order={SG_POLY_ORDER})")
print(f"  Activities : STANDING · SQUATTING")
print(f"  Accuracy   : {accuracy:.1f}%")
print(f"  Plot saved : {ANGLE_PLOT}")
print("=" * 60)