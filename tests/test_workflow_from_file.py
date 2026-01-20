"""
Test full transcription workflow with existing audio files.
Simulates the app behavior with both microphone and loopback audio.

Usage:
    1. Update the file paths below
    2. Run: poetry run python tests/test_workflow_from_file.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio import AudioTranscriber

MICROPHONE_AUDIO_PATH = r"src\saved_audio\recording_20251017_132927_dev1_Microphone_Array__2-_Intel__Sma.wav"
LOOPBACK_AUDIO_PATH = r"src\saved_audio\recording_20251017_132927_dev2_Headphones__2-_Realtek_R__Audio___Loopback_.wav"


def main():
    """Test transcription workflow with microphone and loopback audio."""

    audio_files = []
    device_names = []

    print("\nChecking audio files...")

    if os.path.exists(MICROPHONE_AUDIO_PATH):
        audio_files.append(MICROPHONE_AUDIO_PATH)
        device_names.append("Microphone Array (2- Intel® Sma")
        print(f"✓ Microphone: {os.path.basename(MICROPHONE_AUDIO_PATH)}")
    else:
        print(f"⚠ Microphone file not found: {MICROPHONE_AUDIO_PATH}")

    if os.path.exists(LOOPBACK_AUDIO_PATH):
        audio_files.append(LOOPBACK_AUDIO_PATH)
        device_names.append("Headphones (2- Realtek(R) Audio) [Loopback]")
        print(f"✓ Loopback: {os.path.basename(LOOPBACK_AUDIO_PATH)}")
    else:
        print(f"⚠ Loopback file not found: {LOOPBACK_AUDIO_PATH}")

    if not audio_files:
        print("\n❌ No audio files found. Please update the paths in the script.")
        print("\nAvailable files in src/saved_audio/:")
        audio_dir = "src/saved_audio"
        if os.path.exists(audio_dir):
            files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
            for f in sorted(files)[-5:]:
                print(f"  - {f}")
        return

    # Initialize transcriber
    print("\n" + "=" * 20, "INITIALIZING WHISPER MODEL", "=" * 20)
    print("Loading Whisper model (base)...")
    transcriber = AudioTranscriber(model_name="base")
    transcriber.load_model()
    print("✓ Model loaded and cached for all transcriptions")

    # Transcribe
    print("\n" + "=" * 20, "TRANSCRIBING SEPARATE AUDIO STREAMS", "=" * 20)

    result = transcriber.transcribe_multiple(audio_files=audio_files, device_names=device_names)

    # Display results
    print("\n" + "=" * 20, "TRANSCRIPTION RESULTS", "=" * 20)
    print(result["combined_text"])

    # Summary
    print("\n" + "=" * 20, "SUMMARY", "=" * 20)

    if result["segments"]:
        print(f"✓ Total segments: {len(result['segments'])}")

        # Count by device
        mic_count = sum(1 for s in result["segments"] if s["speaker"] == "Microphone")
        sys_count = sum(1 for s in result["segments"] if s["speaker"] == "System Audio")

        print(f"  - Microphone: {mic_count} segment(s)")
        print(f"  - System Audio: {sys_count} segment(s)")


if __name__ == "__main__":
    main()
