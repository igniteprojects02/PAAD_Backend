# app.py
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
from sklearn.cluster import KMeans
import threading
import time
import math

app = Flask(__name__)

# Load YOLO model (yolov8s = good speed + accuracy)
model = YOLO("yolov8s.pt")

# EasyOCR reader (instantiate once)
ocr_reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if available & configured

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Global metadata and lock for thread safety
latest_metadata = {}
meta_lock = threading.Lock()


def get_dominant_color(roi, k=3):
    """Return dominant color (R,G,B) using KMeans on ROI."""
    if roi.size == 0:
        return (0, 0, 0)
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    pixels = img.reshape((h * w, 3)).astype(float)

    # KMeans clustering
    k = min(k, len(pixels))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)].astype(int)
    return tuple(int(c) for c in dominant)  # (R,G,B)


def rgb_to_color_name(rgb):
    """Map an RGB tuple to a nearest basic color name (simple)."""
    # Basic palette (R,G,B)
    basic_colors = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "gray": (128, 128, 128),
        "orange": (255, 165, 0),
        "brown": (150, 75, 0)
    }
    r, g, b = rgb
    best = None
    best_dist = float('inf')
    for name, (cr, cg, cb) in basic_colors.items():
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_dist:
            best_dist = d
            best = name
    return best


def detect_shape(roi):
    """Return a simple shape label: circle / rectangle / irregular."""
    if roi.size == 0:
        return "unknown"
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"
    # choose the largest contour
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 100:  # too small
        return "small/irregular"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    # decide shape by number of vertices
    vert = len(approx)
    if vert >= 8:
        return "circle/rounded"
    elif vert == 4:
        return "rectangle/square"
    elif vert == 3:
        return "triangle"
    else:
        return "irregular"


def generate_frames():
    global latest_metadata
    while True:
        start_time = time.time()
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO (you can adjust conf threshold)
        results = model(frame, conf=0.45, verbose=False)  # returns a Results object list
        annotated = results[0].plot()  # annotated numpy array (BGR)

        # prepare metadata list for this frame
        frame_metadata = []
        for box in results[0].boxes:
            try:
                cls_idx = int(box.cls[0])
            except Exception:
                # no boxes
                continue
            label = model.names[cls_idx]

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # ensure ROI within frame bounds
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1] - 1, x2); y2 = min(frame.shape[0] - 1, y2)
            width = x2 - x1
            height = y2 - y1

            roi = frame[y1:y2, x1:x2].copy()

            # OCR for likely text-containing objects (add others as needed)
            ocr_text = []
            if label in ["book", "laptop", "tv", "cell phone", "bottle", "book", "clock", "microwave", "oven", "remote"]:
                try:
                    ocr_text = ocr_reader.readtext(roi, detail=0)
                except Exception:
                    ocr_text = []

            # Dominant color
            dom_rgb = get_dominant_color(roi) if roi.size != 0 else (0, 0, 0)
            color_name = rgb_to_color_name(dom_rgb)

            # Shape
            shape_label = detect_shape(roi)

            # Compose metadata
            item = {
                "class": label,
                "confidence": float(box.conf[0]) if hasattr(box, "conf") else None,
                "bbox": [x1, y1, x2, y2],
                "size": {"w": width, "h": height},
                "ocr_text": " ".join(ocr_text) if ocr_text else "",
                "dominant_rgb": {"r": int(dom_rgb[0]), "g": int(dom_rgb[1]), "b": int(dom_rgb[2])},
                "color_name": color_name,
                "shape": shape_label
            }

            # Draw text overlays on annotated frame
            label_text = f"{label} {item['ocr_text']}".strip()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated, f"Color: {color_name}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(annotated, f"Size: {width}x{height}", (x1, y2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            cv2.putText(annotated, f"Shape: {shape_label}", (x1, y2 + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            frame_metadata.append(item)

        # update global metadata (thread-safe)
        with meta_lock:
            latest_metadata = {
                "timestamp": time.time(),
                "objects": frame_metadata
            }

        # encode annotated frame as JPEG
        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        # yield frame for streaming to webpage
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # small sleep if you need to throttle (optional)
        # time.sleep(max(0, 0.01 - (time.time() - start_time)))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/metadata')
def metadata():
    # return latest metadata as JSON
    with meta_lock:
        return jsonify(latest_metadata)


if __name__ == '__main__':
    # Turn off flask debug if you prefer; debug is helpful during dev
    app.run(host='0.0.0.0', port=5000, debug=True)
