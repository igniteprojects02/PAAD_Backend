import socketio
import base64
import cv2
import time

sio = socketio.Client()

@sio.event
def connect():
    print("✅ GESTURE CLIENT: Connected!")

@sio.on('server_response')
def on_message(data):
    # Only print if it's not the "Analyzing" spam
    if "detected" in str(data['result']):
        print(f"\n✋ GESTURE RECOGNIZED: {data['result']}")

def stream_video():
    cap = cv2.VideoCapture(0)
    print("🎥 Streaming for 10 seconds... Make a Hand Sign (Peace/ThumbsUp)!")
    
    start_time = time.time()
    
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret: break
        
        # Resize to speed up upload (optional)
        frame = cv2.resize(frame, (320, 240))
        
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        sio.emit('stream_data', {'mode': 'gesture', 'image': img_str})
        
        # Don't flood the server, 15 FPS is enough
        time.sleep(0.06) 

    print("🛑 Stream finished.")
    cap.release()
    sio.disconnect()

if __name__ == '__main__':
    try:
        sio.connect('http://localhost:5000')
        stream_video()
    except Exception as e:
        print(f"Error: {e}")