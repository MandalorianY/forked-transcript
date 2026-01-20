import os
import sys
import wave
from datetime import datetime

# Platform-specific audio library import
if sys.platform == "win32":
    import pyaudiowpatch as pyaudio
else:
    import pyaudio

"""
Quick test: 10 seconds record from the default microphone

How to run:
    poetry run python tests/test_record.py
"""

DUR = 10
RATE = 16000  # Lower rate for better compatibility with Whisper
CH = 1  # Mono for microphone
FRAMES = 1024
OUTPUT_DIR = "src/saved_audio"

print("Initializing audio...")
pa = pyaudio.PyAudio()

# Get default input device (microphone)
default_input = pa.get_default_input_device_info()
print(f"Using device: {default_input['name']}")
print(f"Recording for {DUR} seconds...")

# Create output file in saved_audio directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(OUTPUT_DIR, f"test_record_{timestamp}.wav")

wf = wave.open(output_file, "wb")
wf.setnchannels(CH)
wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
wf.setframerate(RATE)

stream = pa.open(
    format=pyaudio.paInt16,
    channels=CH,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES,
    input_device_index=default_input["index"],
)

# Record in chunks with progress indicator
num_chunks = int(RATE / FRAMES * DUR)
frames = []
for i in range(num_chunks):
    data = stream.read(FRAMES)
    frames.append(data)
    if i % 10 == 0:  # Print progress every ~0.6 seconds
        progress = (i / num_chunks) * 100
        print(f"Recording... {progress:.0f}%")

print("Recording complete. Saving file...")
wf.writeframes(b"".join(frames))
stream.close()
pa.terminate()
wf.close()
print(f"Saved to {output_file}")
