import base64
import cv2
import numpy as np
from flask import Flask
from flask_socketio import SocketIO, emit

# --- 1. IMPORT VISION MODULE ---
try:
    from services.vision.object_det import analyze_image
    print("✅ Vision Module: CONNECTED.")
except ImportError as e:
    print(f"❌ Vision Module Error: {e}")
    analyze_image = None

# --- 2. IMPORT GESTURE MODULE ---
try:
    from services.gesture.hand_sign import get_gesture
    print("✅ Gesture Module: CONNECTED.")
except ImportError as e:
    print(f"⚠️ Gesture Module: Not found or broken. {e}")
    get_gesture = None

# --- 3. IMPORT AUDIO MODULE ---
try:
    from services.audio.speech import process_audio
    print("✅ Audio Module: CONNECTED.")
except ImportError as e:
    print(f"⚠️ Audio Module: Not found. {e}")
    process_audio = None

# --- 4. SERVER SETUP ---
app = Flask(__name__)
# cors_allowed_origins="*" allows the Flutter app to connect from any IP
socketio = SocketIO(app, cors_allowed_origins="*")

def base64_to_image(base64_string):
    """
    Helper: Decodes the raw text string from Flutter back into an image
    """
    try:
        # Sometimes the string comes with "data:image/jpeg;base64,", remove it
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
            
        decoded_data = base64.b64decode(base64_string)
        np_data = np.frombuffer(decoded_data, np.uint8)
        image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Image Decode Error: {e}")
        return None

# --- 5. CONNECTION EVENTS ---
@socketio.on('connect')
def handle_connect():
    print("📱 Flutter App Connected!")
    emit('server_response', {'result': "Connected to Backend"})

@socketio.on('disconnect')
def handle_disconnect():
    print("🔌 Device Disconnected.")

# --- 6. MAIN ROUTER (The Switchboard) ---
@socketio.on('stream_data')
def handle_stream(data):
    """
    Receives JSON: { "mode": "vision", "image": "base64..." }
    """
    mode = data.get('mode')
    payload = data.get('image') # The image or audio data
    
    response_text = None  # Default to None so we don't send empty messages
    
    # ==========================
    #    MODE: VISION ASSIST
    # ==========================
    if mode == "vision":
        if analyze_image:
            frame = base64_to_image(payload)
            if frame is not None:
                # Returns: "I see a Red Book saying Python"
                # We ALWAYS send vision results because the user triggered it manually
                response_text = analyze_image(frame)
            else:
                response_text = "Error: Bad Image Data"
        else:
            response_text = "Vision Module is not loaded."

    # ==========================
    #   MODE: GESTURE ASSIST
    # ==========================
    elif mode == "gesture":
        if get_gesture:
            frame = base64_to_image(payload)
            if frame is not None:
                raw_result = get_gesture(frame)
                
                # --- FILTERING LOGIC ---
                # Only send to phone if it's a real detected gesture.
                # Ignore "Analyzing..." or "No hand detected" to prevent spam.
                if "detected:" in raw_result:
                    response_text = raw_result
        else:
            # Only send this once, maybe add logic to not spam this error
            pass 

    # ==========================
    #   MODE: HEARING ASSIST
    # ==========================
    elif mode == "audio":
        if process_audio:
            # Audio payload handling will depend on Team B's code
            # Assuming payload is raw audio data or base64
            response_text = process_audio(payload)
        else:
            response_text = "Audio mode is under maintenance."

    # --- SEND RESULT BACK TO FLUTTER ---
    # Only emit if we have a valid response (Not None)
    if response_text:
        print(f"📤 Sending to App ({mode}): {response_text}")
        emit('server_response', {'result': response_text})

# --- 7. START ENGINE ---
if __name__ == '__main__':
    print("🚀 PAAD Server Starting...")
    print("👉 Tell Flutter Dev to connect to port 5000")
    # debug=True allows the server to reload if you change code
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)