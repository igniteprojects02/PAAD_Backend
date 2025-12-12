from flask import Flask, render_template, Response, request
import cv2
from tts_module import speak_text
from speech_module import recognize_speech
from gesture_module import detect_gesture

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # Access webcam

@app.route('/')
def index():
    return render_template('index.html')

# Text-to-Speech endpoint for blind users
@app.route('/tts', methods=['POST'])
def tts():
    text = request.form['text']
    speak_text(text)
    return "Text spoken successfully!"

# Speech-to-Text endpoint for deaf users
@app.route('/stt')
def stt():
    recognized_text = recognize_speech()
    return recognized_text

# Gesture-to-Voice endpoint for speech-impaired
@app.route('/gesture')
def gesture():
    gesture_name = detect_gesture(camera)
    speak_text(f"You did {gesture_name}")
    return f"Gesture detected: {gesture_name}"

if __name__ == '__main__':
    app.run(debug=True)
