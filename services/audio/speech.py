# services/audio/speech.py
import speech_recognition as sr
import base64
import io
import tempfile
import os

# Initialize recognizer once
recognizer = sr.Recognizer()

def process_audio(base64_audio):
    """
    Input: Base64 encoded audio string from Flutter.
    Output: Text string (What was said).
    """
    if not base64_audio:
        return "Error: Empty audio data"

    try:
        # 1. Clean the data (Remove headers if present)
        # Flutter might send "data:audio/wav;base64,..."
        if "," in base64_audio:
            base64_audio = base64_audio.split(",")[1]

        # 2. Decode the audio
        audio_bytes = base64.b64decode(base64_audio)

        # 3. Save to a temporary file
        # SpeechRecognition library struggles with raw bytes, 
        # so saving to a temp file is the safest hackathon method.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        # 4. Recognize
        text = "Could not understand audio"
        try:
            with sr.AudioFile(temp_path) as source:
                # Listen to the file
                audio_data = recognizer.record(source)
                # Send to Google
                text = recognizer.recognize_google(audio_data)
                print(f"👂 Heard: {text}")
        except sr.UnknownValueError:
            text = "Unintelligible speech"
        except sr.RequestError:
            text = "API Unavailable"
        finally:
            # Cleanup: Delete the temp file
            os.remove(temp_path)

        return text

    except Exception as e:
        print(f"❌ Audio Processing Error: {e}")
        return "Error processing audio"