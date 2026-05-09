import cv2
from ultralytics import YOLO

model = YOLO("backend/models/yolo/best.pt")

video_path = r"C:\Users\User\Desktop\videos FYP\wrist_forearm1.mp4"

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "expert_yolo_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    out.write(frame)

cap.release()
out.release()

print("Done -> expert_yolo_output.mp4")