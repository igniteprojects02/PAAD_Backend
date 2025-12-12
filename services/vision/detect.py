from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8s.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # run inference
    annotated = results[0].plot()

    cv2.imshow("YOLO Real-Time Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
