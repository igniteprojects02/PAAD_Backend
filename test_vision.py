import socketio
import base64
import cv2
import time

sio = socketio.Client()

@sio.event
def connect():
    print("✅ VISION CLIENT: Connected!")

@sio.on('server_response')
def on_message(data):
    result_text = data['result']
    print(f"\n📩 SERVER SAYS: {result_text}")
    
    # FIX: Don't quit on the "Connected" welcome message.
    # Only quit if it's an actual Vision result (which usually starts with "I see" or "Error")
    if "Connected" in result_text:
        return 
        
    print("✅ Test Complete. Exiting.")
    sio.disconnect()

def send_photo():
    cap = cv2.VideoCapture(0)
    print("📸 Opening Webcam... Hold up an object!")
    time.sleep(2) # Warmup time
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert to Base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        print("📤 Sending frame to server...")
        sio.emit('stream_data', {'mode': 'vision', 'image': img_str})
    else:
        print("❌ Camera failed.")

if __name__ == '__main__':
    try:
        sio.connect('http://localhost:5000')
        send_photo()
        sio.wait()
    except Exception as e:
        print(f"Error: {e}")