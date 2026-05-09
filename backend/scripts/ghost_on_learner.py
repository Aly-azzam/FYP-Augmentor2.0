import cv2
import json
from ultralytics import YOLO

model = YOLO("backend/models/yolo/best.pt")

learner_video = r"C:\Users\User\Desktop\videos FYP\wrist_forearm.mp4"

with open("expert_trajectory.json", "r") as f:
    trajectory = json.load(f)

traj_dict = {item["frame"]: (item["x"], item["y"]) for item in trajectory}

cap = cv2.VideoCapture(learner_video)

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "ghost_soft_aligned.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# first learner detection for global offset
ret, first_frame = cap.read()
if not ret:
    print("ERROR: cannot read learner video")
    exit()

first_results = model(first_frame)[0]

if len(first_results.boxes) > 0:
    box = first_results.boxes[0]
    lx1, ly1, lx2, ly2 = map(int, box.xyxy[0])
    learner_start_x = (lx1 + lx2) // 2
    learner_start_y = (ly1 + ly2) // 2

    first_expert_frame = min(traj_dict.keys())
    ex0, ey0 = traj_dict[first_expert_frame]

    dx = learner_start_x - ex0
    dy = learner_start_y - ey0
else:
    dx, dy = 0, 0

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_idx = 0
last_point = None
alpha_follow = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    overlay = frame.copy()

    # expert point
    if frame_idx in traj_dict:
        ex, ey = traj_dict[frame_idx]
        last_point = (ex, ey)
    elif last_point is not None:
        ex, ey = last_point
    else:
        frame_idx += 1
        out.write(frame)
        continue

    # learner detection for soft alignment
    results = model(frame)[0]

    base_x = ex + dx
    base_y = ey + dy

    if len(results.boxes) > 0:
        box = results.boxes[0]
        lx1, ly1, lx2, ly2 = map(int, box.xyxy[0])
        learner_x = (lx1 + lx2) // 2
        learner_y = (ly1 + ly2) // 2

        ghost_x = int(base_x * (1 - alpha_follow) + learner_x * alpha_follow)
        ghost_y = int(base_y * (1 - alpha_follow) + learner_y * alpha_follow)
    else:
        ghost_x = int(base_x)
        ghost_y = int(base_y)

    if 0 <= ghost_x < w and 0 <= ghost_y < h:
        cv2.circle(overlay, (ghost_x, ghost_y), 14, (255, 255, 255), -1)
        cv2.circle(overlay, (ghost_x, ghost_y), 18, (0, 0, 0), 2)

    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    out.write(frame)

    frame_idx += 1

cap.release()
out.release()

print("Saved -> ghost_soft_aligned.mp4")