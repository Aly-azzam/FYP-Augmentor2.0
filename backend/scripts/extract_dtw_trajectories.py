import cv2
import json
import math
from ultralytics import YOLO

model = YOLO("backend/models/yolo/best.pt")

expert_video = r"C:\Users\User\Desktop\videos FYP\wrist_forearm1.mp4"
learner_video = r"C:\Users\User\Desktop\videos FYP\wrist_forearm.mp4"

def get_box_center(frame):
    results = model(frame, verbose=False)[0]
    if len(results.boxes) == 0:
        return None

    best_box = max(results.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    return {
        "x": (x1 + x2) // 2,
        "y": (y1 + y2) // 2
    }

def extract(video_path, output_json):
    cap = cv2.VideoCapture(video_path)
    points = []
    frame_idx = 0
    last_point = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        point = get_box_center(frame)

        if point is None and last_point is not None:
            point = last_point

        if point is not None:
            point["frame"] = frame_idx
            points.append(point)
            last_point = {"x": point["x"], "y": point["y"]}

        frame_idx += 1

    cap.release()

    with open(output_json, "w") as f:
        json.dump(points, f, indent=2)

    print(f"Saved {output_json} with {len(points)} points")

extract(expert_video, "expert_dtw_trajectory.json")
extract(learner_video, "learner_dtw_trajectory.json")