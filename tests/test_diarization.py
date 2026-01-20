"""
Test script for the speaker diarization functionality
poetry run python tests/test_diarization.py
"""

import os

from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# Load environment variables from .env file
load_dotenv()
TOKEN = os.getenv("HUGGINGFACE_TOKEN")
INPUT_AUDIO = r"src\saved_audio\recording_20251017_132927_dev2_Headphones__2-_Realtek_R__Audio___Loopback_.wav"


class PyannoteDiarizer:
    def __init__(self, hf_token: str):
        try:
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        except Exception as e:
            self.pipeline = None
            print(f"Error initializing PyannoteDiarizer: {e}")

    def diarize(self, audio_path: str):
        """
        Perform speaker diarization on an audio file
        Returns a list of (start_time, end_time, speaker) tuples
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized.")

        try:
            with ProgressHook() as hook:
                diarization = self.pipeline(audio_path, hook=hook)
                # Convert diarization output to a more manageable format
                segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segments.append((turn.start, turn.end, speaker))
                return segments
        except Exception as e:
            print(f"Error during diarization: {e}")
            return None


diarizer = PyannoteDiarizer(TOKEN)
output = diarizer.diarize(INPUT_AUDIO)

if output:
    for start, end, speaker in output:
        print(f"{speaker} speaks between t={start:.2f}s and t={end:.2f}s")
