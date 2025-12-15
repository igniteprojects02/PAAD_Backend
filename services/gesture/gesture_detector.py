# gesture_module/gesture_detector.py
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

def initialize_gesture_system(max_num_hands=1, det_conf=0.6, track_conf=0.6):
    """Return a MediaPipe Hands object."""
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf
    )

def landmarks_to_pixel_coords(hand_landmarks, image_width, image_height):
    """Convert normalized landmarks to list of 21 (x,y,z) pixel coords."""
    pts = []
    for lm in hand_landmarks.landmark:
        pts.append((int(lm.x * image_width), int(lm.y * image_height), lm.z))
    return pts

def detect_landmarks(frame, hands):
    """
    Process a BGR frame with MediaPipe.
    Returns:
      pts: list of 21 (x,y,z) pixel coords or None
      results: raw MediaPipe results object
      handedness: 'Left' or 'Right' (string) or None
    """
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # optimize: mark not writeable for performance
    frame_rgb.flags.writeable = False
    results = hands.process(frame_rgb)
    frame_rgb.flags.writeable = True

    if not results.multi_hand_landmarks:
        return None, results, None

    hand_landmarks = results.multi_hand_landmarks[0]
    pts = landmarks_to_pixel_coords(hand_landmarks, w, h)

    # handedness may be available
    handedness = None
    if results.multi_handedness:
        try:
            handedness = results.multi_handedness[0].classification[0].label
        except Exception:
            handedness = None

    return pts, results, handedness
