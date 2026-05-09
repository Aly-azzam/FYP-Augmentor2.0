import cv2
import json
from collections import deque
from ultralytics import YOLO

# ================= PATHS =================
model_path = "backend/models/yolo/best.pt"

expert_video = r"C:\Users\User\Desktop\videos FYP\wrist_forearm1.mp4"
learner_video = r"C:\Users\User\Desktop\videos FYP\wrist_forearm.mp4"

dtw_file = "progress_alignment.json"
output_video = "ghost_progress_stabilized.mp4"

# ================= SETTINGS =================
alpha = 0.35
draw_debug_box = True

# Stabilization settings
smoothing_window = 15
fixed_w = 300
fixed_h = 300

model = YOLO(model_path)

def clamp(v, min_v, max_v):
    return max(min_v, min(v, max_v))

def get_yolo_box(frame):
    results = model(frame, verbose=False)[0]
    if len(results.boxes) == 0:
        return None
    best_box = max(results.boxes, key=lambda b: float(b.conf[0]))
    return tuple(map(int, best_box.xyxy[0]))

def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2

# ================= LOAD ALIGNMENT =================
with open(dtw_file, "r") as f:
    dtw_alignment = json.load(f)

dtw_alignment = {int(k): int(v) for k, v in dtw_alignment.items()}

# ================= VIDEO SETUP =================
expert_cap = cv2.VideoCapture(expert_video)
learner_cap = cv2.VideoCapture(learner_video)

fps = int(learner_cap.get(cv2.CAP_PROP_FPS))
w = int(learner_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(learner_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# ================= INITIAL OFFSET ONLY =================
ret_e, first_expert = expert_cap.read()
ret_l, first_learner = learner_cap.read()

if not ret_e or not ret_l:
    print("ERROR: cannot read expert or learner video")
    exit()

expert_box = get_yolo_box(first_expert)
learner_box = get_yolo_box(first_learner)

if expert_box is None or learner_box is None:
    print("ERROR: YOLO could not detect expert or learner on first frame")
    exit()

expert_start_x, expert_start_y = box_center(expert_box)
learner_start_x, learner_start_y = box_center(learner_box)

dx = learner_start_x - expert_start_x
dy = learner_start_y - expert_start_y

print(f"Initial offset only: dx={dx}, dy={dy}")

# ================= PRELOAD EXPERT FRAMES =================
expert_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

expert_frames = []
while True:
    ret, frame = expert_cap.read()
    if not ret:
        break
    expert_frames.append(frame)

print(f"Loaded expert frames: {len(expert_frames)}")

learner_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ================= STABILIZATION STATE =================
center_history = deque(maxlen=smoothing_window)
last_expert_crop = None
last_expert_center = None
last_expert_frame_idx = None

learner_frame_idx = 0

while True:
    ret_l, learner_frame = learner_cap.read()
    if not ret_l:
        break

    output = learner_frame.copy()

    expert_frame_idx = dtw_alignment.get(learner_frame_idx, None)

    if expert_frame_idx is not None:
        expert_frame_idx = clamp(expert_frame_idx, 0, len(expert_frames) - 1)
        expert_frame = expert_frames[expert_frame_idx]

        box = get_yolo_box(expert_frame)

        if box is not None:
            raw_cx, raw_cy = box_center(box)

            # ===== SMOOTH CENTER =====
            center_history.append((raw_cx, raw_cy))

            smooth_cx = int(sum(p[0] for p in center_history) / len(center_history))
            smooth_cy = int(sum(p[1] for p in center_history) / len(center_history))

            # ===== FIXED CROP SIZE =====
            ex1 = clamp(smooth_cx - fixed_w // 2, 0, expert_frame.shape[1] - 1)
            ey1 = clamp(smooth_cy - fixed_h // 2, 0, expert_frame.shape[0] - 1)
            ex2 = clamp(smooth_cx + fixed_w // 2, 0, expert_frame.shape[1] - 1)
            ey2 = clamp(smooth_cy + fixed_h // 2, 0, expert_frame.shape[0] - 1)

            expert_crop = expert_frame[ey1:ey2, ex1:ex2]

            last_expert_crop = expert_crop
            last_expert_center = (smooth_cx, smooth_cy)
            last_expert_frame_idx = expert_frame_idx

    # ================= OVERLAY =================
    if last_expert_crop is not None and last_expert_center is not None:
        ghost = last_expert_crop.copy()
        gh, gw = ghost.shape[:2]

        expert_cx, expert_cy = last_expert_center

        ghost_center_x = int(expert_cx + dx)
        ghost_center_y = int(expert_cy + dy)

        ox1 = int(ghost_center_x - gw // 2)
        oy1 = int(ghost_center_y - gh // 2)
        ox2 = ox1 + gw
        oy2 = oy1 + gh

        if ox1 < 0:
            ghost = ghost[:, -ox1:]
            ox1 = 0
        if oy1 < 0:
            ghost = ghost[-oy1:, :]
            oy1 = 0
        if ox2 > w:
            ghost = ghost[:, :w - ox1]
            ox2 = w
        if oy2 > h:
            ghost = ghost[:h - oy1, :]
            oy2 = h

        if ghost.size > 0:
            roi = output[oy1:oy2, ox1:ox2]
            blended = cv2.addWeighted(ghost, alpha, roi, 1 - alpha, 0)
            output[oy1:oy2, ox1:ox2] = blended

            if draw_debug_box:
                cv2.rectangle(output, (ox1, oy1), (ox2, oy2), (255, 255, 255), 2)
                cv2.putText(
                    output,
                    f"STABILIZED EXPERT GHOST | expert frame {last_expert_frame_idx}",
                    (ox1, max(30, oy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    out.write(output)
    learner_frame_idx += 1

expert_cap.release()
learner_cap.release()
out.release()

print(f"Saved -> {output_video}")