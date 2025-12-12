# gesture_module/gesture_to_voice.py
import cv2
import time
import collections
import mediapipe as mp

from .gesture_detector import initialize_gesture_system, detect_landmarks
from .gesture_classifier import classify_gesture
from .gesture_mapper import map_gesture_to_text
from .speak import speak_text

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def start_gesture_to_voice(camera_index=0,
                           required_stable_frames=6,
                           cooldown_seconds=2.0,
                           show_label=True):
    """
    camera_index: cv2 camera id
    required_stable_frames: number of consecutive frames required for stability
    cooldown_seconds: minimal seconds between speaking same gesture
    show_label: overlay the gesture name on the frame
    """
    hands = initialize_gesture_system()
    cap = cv2.VideoCapture(camera_index)
    # optionally set a reasonable frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # stability / smoothing
    stable_gesture = None
    stable_count = 0
    last_spoken_time = 0.0

    # small recent history to smooth instantaneous fluctuation (optional)
    recent = collections.deque(maxlen=8)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame not available.")
                break

            pts, results, handedness = detect_landmarks(frame, hands)

            # draw landmarks for debugging/visual feedback
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # classify gesture using pts + handedness
            gesture = classify_gesture(pts, handedness)

            # smoothing: add to recent buffer and compute mode
            recent.append(gesture)
            # compute a simple mode ignoring None
            try:
                # pick most common non-None in recent
                items = [g for g in recent if g is not None]
                smoothed = max(set(items), key=items.count) if items else None
            except Exception:
                smoothed = gesture

            # stability tracking
            if smoothed == stable_gesture:
                stable_count += 1
            else:
                stable_gesture = smoothed
                stable_count = 1

            now = time.time()
            if stable_gesture and stable_count >= required_stable_frames:
                # only speak if cooldown passed
                if now - last_spoken_time > cooldown_seconds:
                    text = map_gesture_to_text(stable_gesture)
                    if text:
                        print(f"Speaking ({stable_gesture}): {text}")
                        speak_text(text)
                        last_spoken_time = now

            # overlay label
            if show_label:
                label = stable_gesture if stable_gesture else "No gesture"
                cv2.putText(frame, f"{label}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Gesture Window", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
