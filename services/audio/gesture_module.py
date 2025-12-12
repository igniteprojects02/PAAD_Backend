import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

def detect_gesture(camera):
    ret, frame = camera.read()
    if not ret:
        return "No camera detected"

    # YOLO expects BGR (OpenCV format), so feed frame directly
    results = model.predict(source=frame, save=False, conf=0.5)

    detected = []

    # Process YOLO results
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detected.append(label)

            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    if not detected:
        return "No objects detected"

    return detected


# ---- MAIN LOOP ----
if __name__ == "__main__":
    camera = cv2.VideoCapture(0)

    while True:
        objects = detect_gesture(camera)
        print(objects)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
