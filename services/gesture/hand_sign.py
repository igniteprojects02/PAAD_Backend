# services/gesture/hand_sign.py
import collections
from .gesture_detector import initialize_gesture_system, detect_landmarks
from .gesture_classifier import classify_gesture
from .gesture_mapper import map_gesture_to_text

# --- 1. GLOBAL MEMORY (To fix flickering) ---
# Since we don't have a "while loop", we store the history in global variables.
# This ensures that if you send Frame 1, then Frame 2, the server remembers Frame 1.

# Initialize MediaPipe ONCE
print("✋ Gesture: Loading MediaPipe Hands...")
hands = initialize_gesture_system()

# Memory buffers
recent_gestures = collections.deque(maxlen=8) # Stores last 8 frames
stable_gesture = None
stable_count = 0
REQUIRED_STABLE_FRAMES = 6

def get_gesture(frame):
    """
    Input: Image frame from Flutter
    Output: The text command (e.g., "Hello") OR None if no stable gesture found.
    """
    global stable_gesture, stable_count

    # 1. DETECT LANDMARKS
    # We use their detector function
    pts, results, handedness = detect_landmarks(frame, hands)

    if pts is None:
        # If no hand seen, clear history slightly to prevent stuck gestures
        recent_gestures.append(None)
        return "No hand detected"

    # 2. CLASSIFY
    # We use their classifier function
    raw_gesture = classify_gesture(pts, handedness)
    
    # 3. SMOOTHING LOGIC (The "Anti-Glitch" System)
    # Add to history
    recent_gestures.append(raw_gesture)

    # Find the most common gesture in the last 8 frames
    try:
        # Remove None values from list
        valid_gestures = [g for g in recent_gestures if g is not None]
        if valid_gestures:
            # Pick the most frequent one
            smoothed_gesture = max(set(valid_gestures), key=valid_gestures.count)
        else:
            smoothed_gesture = None
    except Exception:
        smoothed_gesture = raw_gesture

    # 4. CHECK STABILITY
    # Only return the result if we have seen the SAME gesture for X frames in a row
    if smoothed_gesture == stable_gesture:
        stable_count += 1
    else:
        stable_gesture = smoothed_gesture
        stable_count = 1

    # 5. FINAL RESULT
    if stable_gesture and stable_count >= REQUIRED_STABLE_FRAMES:
        # Convert "thumbs_up" -> "Yes"
        final_text = map_gesture_to_text(stable_gesture)
        if final_text:
            return f"Gesture detected: {final_text}"
    
    # If not stable yet, return raw status for debugging (optional)
    return "Analyzing gesture..."