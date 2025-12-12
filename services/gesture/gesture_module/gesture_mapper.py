# gesture_module/gesture_mapper.py
gesture_map = {
    "thumbs_up": "Yes",
    "fist": "Stop",
    "open_palm": "Hello",
    "point": "Look here",
    "peace": "Peace",
    "ok": "Okay",
    "rock": "Rock on",
    "callme": "Call me"
}

def map_gesture_to_text(label):
    if label is None:
        return None
    return gesture_map.get(label, None)
