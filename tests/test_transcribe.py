import os
import wave

import numpy as np
import whisper

"""
Transcribe probe audio from 10 sec probe recording.

How to run:
    poetry run python tests/test_transcribe.py

Note: This script loads WAV files directly without ffmpeg dependency.
"""


def load_wav_file(filepath):
    """Load a WAV file and convert to format expected by Whisper."""
    with wave.open(filepath, "rb") as wf:
        # Get audio parameters
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        # Read audio data
        audio_data = wf.readframes(n_frames)

        # Convert to numpy array
        if sampwidth == 2:  # 16-bit audio
            audio = np.frombuffer(audio_data, dtype=np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        # Convert to float32 and normalize to [-1, 1]
        audio = audio.astype(np.float32) / 32768.0

        # Convert stereo to mono if needed
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if framerate != 16000:
            # Simple resampling (for better quality, use scipy.signal.resample)
            duration = len(audio) / framerate
            target_length = int(duration * 16000)
            audio = np.interp(np.linspace(0, len(audio), target_length), np.arange(len(audio)), audio)

        return audio


# Check if audio file exists
if not os.path.exists("out.wav"):
    print("❌ Error: out.wav not found. Please run test_record.py first.")
    exit(1)

print("Loading audio file...")
try:
    audio = load_wav_file("out.wav")
    print(f"Audio loaded: {len(audio) / 16000:.1f} seconds")
except Exception as e:
    print(f"❌ Error loading audio: {e}")
    exit(1)

print("Loading Whisper model (this may take a moment)...")
model = whisper.load_model("base")  # Using 'base' model - faster than 'small'

print("Transcribing audio...")
result = model.transcribe(audio, language="en")

print("\n" + "=" * 50)
print("TRANSCRIPTION:")
print("=" * 50)
print(result["text"])
print("=" * 50)
