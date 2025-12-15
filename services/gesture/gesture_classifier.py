# gesture_module/gesture_classifier.py
import math

# utils
def _angle(a, b, c):
    """Angle at b (degrees) formed by a-b-c (2D)."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    m1 = math.hypot(v1[0], v1[1])
    m2 = math.hypot(v2[0], v2[1])
    if m1*m2 == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, dot/(m1*m2)))
    return math.degrees(math.acos(cosang))

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def hand_scale(pts):
    """
    Return a scale factor for the hand so thresholds are relative.
    Use distance wrist(0) -> middle_mcp(9) as baseline.
    """
    try:
        return max(1.0, dist(pts[0], pts[9]))
    except Exception:
        return 1.0

# finger straightness via angle at PIP: MCP - PIP - TIP
def finger_extended(pts, mcp_id, pip_id, tip_id, angle_threshold=60):
    angle = _angle(pts[mcp_id], pts[pip_id], pts[tip_id])
    return angle > angle_threshold  # larger angle ≈ straighter

def thumb_extended(pts, handedness, hand_scale_val, thresh_ratio=0.35):
    """
    More robust thumb test:
    - Check distance of thumb tip (4) from wrist(0) relative to hand scale
    - Also check angle of thumb CMC-MCP-TIP
    - handedness used to interpret horizontal direction if needed
    """
    wrist = pts[0]
    thumb_tip = pts[4]
    d = dist(wrist, thumb_tip)
    # if thumb tip is far enough from wrist (relative), consider extended
    if d > hand_scale_val * thresh_ratio:
        return True
    # fallback angle check: MCP(2)-IP(3)-TIP(4)
    angle = _angle(pts[2], pts[3], pts[4])
    return angle > 40

def classify_gesture(pts, handedness=None):
    """
    pts: list of 21 (x,y,z) pixel coords OR None
    handedness: 'Left' or 'Right' or None
    returns label string or None
    """
    if pts is None:
        return None

    scale = hand_scale(pts)

    # index: mcp=5, pip=6, tip=8
    index_ext  = finger_extended(pts, 5, 6, 8, angle_threshold=60)
    middle_ext = finger_extended(pts, 9, 10, 12, angle_threshold=60)
    ring_ext   = finger_extended(pts, 13, 14, 16, angle_threshold=60)
    pinky_ext  = finger_extended(pts, 17, 18, 20, angle_threshold=60)

    thumb_ext = thumb_extended(pts, handedness, scale, thresh_ratio=0.28)

    # OK sign: thumb tip (4) close to index tip (8) relative to hand size
    ok_dist = dist(pts[4], pts[8])
    ok_thresh = scale * 0.25

    # peace: idx+mid extended, ring+pinky folded
    if index_ext and middle_ext and (not ring_ext) and (not pinky_ext):
        return "peace"

    # pointing: index only
    if index_ext and (not middle_ext) and (not ring_ext) and (not pinky_ext):
        return "point"

    # thumbs up: thumb extended and other fingers folded
    if thumb_ext and (not index_ext) and (not middle_ext) and (not ring_ext) and (not pinky_ext):
        return "thumbs_up"

    # fist: all folded
    if (not index_ext) and (not middle_ext) and (not ring_ext) and (not pinky_ext):
        return "fist"

    # open palm / stop: all extended
    if index_ext and middle_ext and ring_ext and pinky_ext:
        return "open_palm"  # caller can treat as stop if needed

    # ok sign
    if ok_dist < ok_thresh:
        return "ok"

    # rock (index+pinky)
    if index_ext and (not middle_ext) and (not ring_ext) and pinky_ext:
        return "rock"

    # call me (thumb + pinky)
    if thumb_ext and (not index_ext) and (not middle_ext) and (not ring_ext) and pinky_ext:
        return "callme"

    return None
