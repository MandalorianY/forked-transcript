"""
Test workflow for interactive device selection, recording, and transcription.

This is a debug test file for testing the full audio capture and
transcription workflow with multiple devices.

How to run:
    poetry run python tests/test_full_workflow.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio import (
    AudioCapture,
    AudioTranscriber,
    categorize_devices,
    get_audio_level,
    mix_wav_files,
    save_audio_array,
)

DURATION = 15
OUTPUT_DIR = "src/saved_audio"


def display_devices_with_levels(capture):
    """Display all audio devices with categorization and audio levels."""
    print("\nAVAILABLE AUDIO DEVICES")
    print("-" * 50)

    devices = capture.list_devices()
    categorized = categorize_devices(devices)

    print("\nINPUT DEVICES:")
    if categorized["input"]:
        for dev in categorized["input"]:
            level = capture.get_audio_level(dev["index"], duration=0.2)
            level_bar = "█" * int(level * 20)
            print(f"  [{dev['index']:2d}] {dev['name']}")
            print(f"       Channels: {dev['maxInputChannels']}, Rate: {dev['defaultSampleRate']:.0f}Hz")
            print(f"       Level: [{level_bar:<20}] {level:.3f}")
    else:
        print("  No input devices found")

    print("\nLOOPBACK DEVICES (System Audio):")
    if categorized["loopback"]:
        for dev in categorized["loopback"]:
            print(f"  [{dev['index']:2d}] {dev['name']}")
            print(f"       Channels: {dev['maxInputChannels']}, Rate: {dev['defaultSampleRate']:.0f}Hz")
    else:
        print("  No loopback devices found")

    print("\nOUTPUT DEVICES:")
    for dev in categorized["output"]:
        print(f"  [{dev['index']:2d}] {dev['name']}")
        print(f"       Channels: {dev['maxOutputChannels']}, Rate: {dev['defaultSampleRate']:.0f}Hz")

    print("\nGUIDE:")
    print("- Select microphone for voice recording")
    print("- Select loopback device for system audio (Teams/speakers)")
    print("- Loopback devices only capture audio when something is playing")


def interactive_test():
    """Interactive test for device selection and recording."""
    print("\nSPEECH TO TEXT - RECORD & TRANSCRIBE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    capture = AudioCapture(frames_per_buffer=1024)

    display_devices_with_levels(capture)

    print("\nSELECT DEVICES:")
    print("-" * 50)
    print("First, select your microphone for voice input:")
    mic_index = input("Enter microphone device index: ").strip()
    print("\nNow, select the LOOPBACK device for Teams/system audio:")
    loopback_index = input("Enter loopback device index: ").strip()

    try:
        mic_index = int(mic_index)
        loopback_index = int(loopback_index)

        devices = capture.list_devices()
        mic_info = devices[mic_index]
        loopback_info = devices[loopback_index]

        if mic_info["maxInputChannels"] <= 0:
            print("\n❌ Error: Selected microphone is not an input device")
            return

        if loopback_info["maxInputChannels"] <= 0:
            print("\n❌ Error: Selected device is not an input device")
            return

        device_indices = [mic_index, loopback_index]
        device_names = [mic_info["name"], loopback_info["name"]]
        channels_list = [min(mic_info["maxInputChannels"], 2), min(loopback_info["maxInputChannels"], 2)]
        rates = [int(mic_info["defaultSampleRate"]), int(loopback_info["defaultSampleRate"])]

        print("\n✓ Selected devices:")
        print(f"  Microphone: {mic_info['name']}")
        print(f"  Loopback:   {loopback_info['name']}")

        confirm = input("\nProceed with recording? (y/n): ").strip().lower()
        if confirm != "y":
            print("Recording cancelled.")
            return

        print("\nRECORDING AUDIO")
        print("-" * 50)
        for i, name in enumerate(device_names):
            print(f"Device {i + 1}: {name}")
            print(f"Settings: {rates[i]}Hz, {channels_list[i]} channel(s)")
        print(f"Duration: {DURATION} seconds")
        print("-" * 50)

        audio_files = capture.record_multi_device(
            device_indices=device_indices,
            device_names=device_names,
            channels_list=channels_list,
            rates=rates,
            duration=DURATION,
            output_dir=OUTPUT_DIR,
        )

        print("\n✓ Recordings saved:")
        for file in audio_files:
            print(f"  - {file}")

        transcribe = input("\nTranscribe recordings? (y/n): ").strip().lower()

        if transcribe == "y":
            print("\n" + "=" * 80)
            print("MIXING AND TRANSCRIBING AUDIO")
            print("=" * 80)

            print("Mixing audio files...")
            mixed_audio = mix_wav_files(audio_files)
            print(f"✓ Mixed audio length: {len(mixed_audio) / 48000:.1f} seconds")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mixed_file = os.path.join(OUTPUT_DIR, f"recording_{timestamp}_mixed.wav")

            print(f"Saving mixed audio to: {mixed_file}")
            save_audio_array(mixed_audio, mixed_file, rate=48000)
            print("✓ Mixed audio saved")

            audio_level = get_audio_level(mixed_audio)
            print(f"Audio level: {audio_level:.6f}")

            if audio_level < 0.001:
                print("⚠ Warning: Audio is very quiet or silent")

            print("\nLoading Whisper model (base)...")
            transcriber = AudioTranscriber(model_name="base")
            transcriber.load_model()
            print("✓ Model loaded")

            print("Transcribing...")
            result = transcriber.transcribe(mixed_file, verbose=False)

            print("\nTRANSCRIPTION RESULTS")
            print("-" * 50)
            if result["text"].strip():
                print(result["text"])
            else:
                print("(No speech detected)")
            print("-" * 50)

            if "language" in result:
                print(f"\nDetected Language: {result['language']}")

            print("\nCleaning up temporary files...")
            os.remove(mixed_file)
            print("✓ Cleanup complete")

        print("\n✓ TEST COMPLETE")
        print("-" * 50)

    except ValueError:
        print("\n❌ Invalid device index. Please enter a number.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main entry point."""
    try:
        interactive_test()
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
