# main.py
from gesture_module.gesture_to_voice import start_gesture_to_voice

if __name__ == "__main__":
    start_gesture_to_voice(camera_index=0,
                           required_stable_frames=6,
                           cooldown_seconds=2.0,
                           show_label=True)
