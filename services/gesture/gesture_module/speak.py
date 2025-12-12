import pygame
from gtts import gTTS
import tempfile
import os
import threading

# Initialize pygame mixer once
pygame.mixer.init()

def _play_audio(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    # Wait until playback finishes
    while pygame.mixer.music.get_busy():
        pass
    os.remove(file_path)

def speak_text(text):
    if not text:
        return

    # Create temporary MP3 file
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)

    tts = gTTS(text=text, lang='en')
    tts.save(path)

    # Play in background thread
    threading.Thread(target=_play_audio, args=(path,), daemon=True).start()
