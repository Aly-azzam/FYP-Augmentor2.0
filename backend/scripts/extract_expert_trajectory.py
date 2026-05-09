import cv2
import json
from ultralytics import YOLO

model = YOLO("backend/models/yolo/best.pt")

video_path = r"C:\Users\User\Desktop\videos FYP\wrist_forearm1.mp4"

cap = cv2.VideoCapture(video_path)

trajectory = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    if len(results.boxes) > 0:
        box = results.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        trajectory.append({
            "frame": frame_idx,
            "x": cx,
            "y": cy
        })

    frame_idx += 1

cap.release()

with open("expert_trajectory.json", "w") as f:
    json.dump(trajectory, f, indent=2)

print("Trajectory saved -> expert_trajectory.json")