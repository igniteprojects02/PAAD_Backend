import socketio
import base64
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import time

sio = socketio.Client()

@sio.event
def connect():
    print("✅ AUDIO CLIENT: Connected! Sending audio now...")
    # As soon as we connect, we send the pre-recorded file
    send_audio_file()

@sio.on('server_response')
def on_message(data):
    result_text = data['result']
    
    # Ignore the "Welcome" message
    if "Connected" in result_text:
        return

    print(f"\n🗣️ FINAL RESULT: {result_text}")
    sio.disconnect()

def record_audio():
    fs = 44100  # Sample rate
    seconds = 5  # Duration
    
    print("\n🎙️  Recording for 5 seconds... (Speak clearly!)")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    print("✅ Recording finished.")

    # CRITICAL FIX: Convert audio from Float to 16-bit PCM Integer
    # (Google Speech API hates raw floats)
    data_int16 = (myrecording * 32767).astype(np.int16)

    filename = "test_audio_input.wav"
    write(filename, fs, data_int16)
    return filename

def send_audio_file():
    filename = "test_audio_input.wav"
    
    if not os.path.exists(filename):
        print("❌ Error: Audio file not found.")
        return

    # Convert to Base64
    with open(filename, "rb") as wav_file:
        encoded_string = base64.b64encode(wav_file.read()).decode('utf-8')

    print("📤 Uploading audio to server...")
    sio.emit('stream_data', {'mode': 'audio', 'image': encoded_string})

    # Cleanup
    time.sleep(1) # Wait a sec so file isn't locked
    if os.path.exists(filename):
        os.remove(filename)

if __name__ == '__main__':
    try:
        # 1. Record Offline (No connection yet)
        record_audio()
        
        # 2. Connect (This triggers the send_audio_file function immediately)
        print("🔌 Connecting to server...")
        sio.connect('http://localhost:5000')
        sio.wait()
        
    except Exception as e:
        print(f"Error: {e}")