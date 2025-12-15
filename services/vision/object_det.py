# services/vision/object_det.py
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from sklearn.cluster import KMeans

# --- 1. GLOBAL INITIALIZATION (Run once on startup) ---
print("👁️ Vision Module: Loading YOLOv8s Model (this may take a moment)...")
try:
    # Ensure you have yolov8s.pt downloaded in your main folder
    model = YOLO("yolov8s.pt") 
    print("✅ Vision: YOLO Model Loaded.")
except Exception as e:
    print(f"❌ Vision Error: Could not load YOLO. {e}")
    model = None

print("📖 Vision Module: Loading OCR Reader...")
try:
    # gpu=True is faster if you have NVIDIA, otherwise False is safe for laptops
    ocr_reader = easyocr.Reader(['en'], gpu=False) 
    print("✅ Vision: OCR Loaded.")
except Exception as e:
    print(f"❌ Vision Warning: OCR failed to load. {e}")
    ocr_reader = None

# --- 2. HELPER FUNCTIONS ---

def get_dominant_color(roi, k=3):
    """Return dominant color (R,G,B) using KMeans."""
    if roi.size == 0: return (0, 0, 0)
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    pixels = img.reshape((h * w, 3)).astype(float)
    
    # Use fewer iterations (n_init=1) to be faster
    k = min(k, len(pixels))
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=1).fit(pixels)
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)].astype(int)
    return tuple(int(c) for c in dominant)

def rgb_to_color_name(rgb):
    """Simple map of RGB to nearest color name."""
    basic_colors = {
        "black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0),
        "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
        "cyan": (0, 255, 255), "magenta": (255, 0, 255), "gray": (128, 128, 128),
        "orange": (255, 165, 0), "brown": (150, 75, 0)
    }
    r, g, b = rgb
    best_dist = float('inf')
    best_name = "unknown"
    for name, (cr, cg, cb) in basic_colors.items():
        d = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name

def detect_shape(roi):
    """Simple shape detection logic."""
    if roi.size == 0: return ""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return ""
    
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100: return "" # Too small
    
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    vert = len(approx)
    
    if vert == 3: return "triangular"
    elif vert == 4: return "rectangular"
    elif vert >= 8: return "round"
    return "irregular"

# --- 3. THE MAIN LOGIC ---

def analyze_image(frame):
    """
    Input: frame (OpenCV image from Flutter/Webcam)
    Output: String (Natural language description for TTS)
    """
    if model is None: return "Error: AI Model not loaded."
    
    # Run YOLO (Turn off verbose to keep terminal clean)
    results = model(frame, conf=0.45, verbose=False)
    
    found_descriptions = []

    # Loop through detected boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 1. Get Class Name (What is it?)
            cls_idx = int(box.cls[0])
            label = model.names[cls_idx]
            
            # 2. Extract the Object (ROI)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            h, w, _ = frame.shape
            # Clamp coordinates to ensure we don't crash
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0: continue # Skip if empty

            description_parts = []
            
            # --- THE POLITE PATCH: Only describe appearance for non-humans ---
            if label != "person":
                # 3. Detect Color
                dom_rgb = get_dominant_color(roi)
                color_name = rgb_to_color_name(dom_rgb)
                description_parts.append(color_name)
                
                # 4. Detect Shape (Optional)
                shape = detect_shape(roi)
                if shape: description_parts.append(shape)
            
            # Add the Object Name (e.g., "Person", "Book")
            description_parts.append(label)

            # 5. SMART OCR (The Upgrade)
            # Only run if object likely has text. Added 'cup' and 'box' to list.
            text_objects = ["book", "laptop", "cell phone", "bottle", "sign", "paper", "cup", "box", "tv"]
            
            if label in text_objects and ocr_reader:
                try:
                    # A. Pre-process: Grayscale + Upscale 2x
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    large_roi = cv2.resize(gray_roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    
                    # B. Read with tuned parameters
                    # contrast_ths=0.05 allows reading fainter text
                    # adjust_contrast=0.5 improves sharpness
                    text_list = ocr_reader.readtext(
                        large_roi, 
                        detail=0, 
                        paragraph=True,
                        contrast_ths=0.05, 
                        adjust_contrast=0.5
                    )
                    
                    if text_list:
                        detected_text = " ".join(text_list)
                        # C. Filter out noise (single letters like 'i', 'e')
                        if len(detected_text) > 2: 
                            description_parts.append(f"saying {detected_text}")
                            print(f"   📖 OCR Success: {detected_text}") # Debug log
                except Exception as e:
                    # Don't crash if OCR fails, just skip text reading
                    print(f"   ⚠️ OCR Warning: {e}")

            # Combine: "Red rectangular Book saying Python"
            full_desc = " ".join(description_parts)
            found_descriptions.append(full_desc)

    if not found_descriptions:
        return "Nothing detected."
    
    # Join all objects: "I see a Red Book, and a Black Phone"
    final_response = "I see " + ", and ".join(found_descriptions)
    return final_response