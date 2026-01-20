"""
Streamlit UI for speech-to-text recording and transcription.

This is the main application interface providing two modes:
- User Mode: Simplified interface with automatic device detection and unlimited recording
- Dev Mode: Advanced interface with manual device selection and fixed duration recording

The application handles:
- Multi-device audio recording (microphone + system loopback)
- Real-time transcription using Whisper
- Speaker diarization using pyannote.audio
- Transcript history and export

Run with: poetry run streamlit run src/ui/app.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from st_copy import copy_button

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.audio import AudioCapture, AudioTranscriber, MeetingSummarizer, PyannoteDiarizer, categorize_devices

OUTPUT_DIR = "src/saved_audio"
TRANSCRIPT_DIR = "src/saved_transcripts"
SUMMARY_DIR = "src/saved_summary"


def initialize_session_state():
    """Initialize session state variables."""
    if "dev_mode" not in st.session_state:
        st.session_state.dev_mode = False
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "stop_recording" not in st.session_state:
        st.session_state.stop_recording = False
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""
    if "transcript_history" not in st.session_state:
        st.session_state.transcript_history = []
    if "audio_files" not in st.session_state:
        st.session_state.audio_files = []
    if "recording_start_time" not in st.session_state:
        st.session_state.recording_start_time = None
    if "recording_thread" not in st.session_state:
        st.session_state.recording_thread = None
    if "recording_complete" not in st.session_state:
        st.session_state.recording_complete = False
    if "transcriber" not in st.session_state:
        print("\n" + "=" * 20, "INITIALIZING WHISPER MODEL", "=" * 20)
        print("Loading Whisper model (base)...")
        st.session_state.transcriber = AudioTranscriber(model_name="base")
        st.session_state.transcriber.load_model()
        print("✓ Model loaded and cached for all transcriptions")
    if "diarizer" not in st.session_state:
        load_dotenv()
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            # Initialize diarizer but don't load pipeline yet (lazy loading)
            st.session_state.diarizer = PyannoteDiarizer(hf_token=hf_token)
            print("✓ Diarization enabled (pipeline will load on first use)")
        else:
            print("⚠ HUGGINGFACE_TOKEN not found, diarization disabled")
            st.session_state.diarizer = None
    if "summarizer" not in st.session_state:
        load_dotenv()
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            # Initialize summarizer but don't load client yet (lazy loading)
            st.session_state.summarizer = MeetingSummarizer(api_key=openai_key, model="gpt-4o-mini")
            print("✓ Summarization enabled (client will load on first use)")
        else:
            print("⚠ OPENAI_API_KEY not found, summarization disabled")
            st.session_state.summarizer = None
    if "device_names" not in st.session_state:
        st.session_state.device_names = []


def get_default_microphone(devices):
    """
    Get the default microphone device.

    Prefers "Microphone Array" if available, otherwise returns the first
    available input device.

    Args:
        devices: List of device dictionaries from AudioCapture.list_devices()

    Returns:
        Device dictionary or None if no input devices found
    """
    categorized = categorize_devices(devices)

    for dev in categorized["input"]:
        if "microphone array" in dev["name"].lower():
            return dev

    if categorized["input"]:
        return categorized["input"][0]

    return None


def get_all_loopback_devices(devices):
    """
    Get all loopback devices for system audio capture.

    Args:
        devices: List of device dictionaries from AudioCapture.list_devices()

    Returns:
        List of loopback device dictionaries
    """
    categorized = categorize_devices(devices)
    return categorized["loopback"]


def save_transcript_json(result, audio_files):
    """
    Save transcript to JSON file with timestamps and metadata.

    Creates a JSON file in the saved_transcripts directory containing the
    full transcription, segments, speaker labels, and metadata.

    Args:
        result: Transcription result dictionary from transcribe_multiple()
        audio_files: List of audio file paths that were transcribed

    Returns:
        Path to the saved JSON file
    """
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"transcript_{timestamp}.json"
    json_filepath = os.path.join(TRANSCRIPT_DIR, json_filename)

    # Prepare JSON data
    json_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "combined_text": result["combined_text"],
        "segments": result["segments"],
        "separate_transcripts": result["transcripts"],
        "audio_files": [os.path.basename(f) for f in audio_files],
        "num_devices": result["num_devices"],
    }

    # Save to JSON file
    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Transcript saved to: {json_filepath}")
    return json_filepath


def save_summary(summary_text, timestamp=None):
    """
    Save meeting summary to text file.

    Creates a text file in the saved_summary directory containing the
    generated meeting minutes.

    Args:
        summary_text: Summary text to save
        timestamp: Optional timestamp string, defaults to current time

    Returns:
        Path to the saved summary file
    """
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_filename = f"summary_{timestamp}.txt"
    summary_filepath = os.path.join(SUMMARY_DIR, summary_filename)

    # Save to text file
    with open(summary_filepath, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n✓ Summary saved to: {summary_filepath}")
    return summary_filepath


def start_recording_thread(device_indices, device_names, channels_list, rates):
    """
    Start recording in a background thread for unlimited duration.

    Creates a daemon thread that records from multiple devices simultaneously
    until stopped by the user. Used in User Mode for flexible recording duration.

    Args:
        device_indices: List of device indices to record from
        device_names: List of device names for file naming
        channels_list: List of channel counts for each device
        rates: List of sample rates for each device

    Returns:
        RecordingThread instance that can be stopped via stop_flag.set()
    """
    import queue
    import sys
    import threading
    import wave

    # Platform-specific audio library import
    if sys.platform == "win32":
        import pyaudiowpatch as pyaudio
    else:
        import pyaudio

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files = []

    for i, name in enumerate(device_names):
        clean_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name)
        clean_name = clean_name.replace(" ", "_")
        if len(clean_name) > 50:
            clean_name = clean_name[:50]
        filename = f"recording_{timestamp}_dev{i + 1}_{clean_name}.wav"
        output_files.append(os.path.join(OUTPUT_DIR, filename))

    print("\n" + "=" * 80)
    print("RECORDING STARTED (UNLIMITED)")
    print("=" * 80)
    for i, name in enumerate(device_names):
        print(f"Device {i + 1}: {name}")
        print(f"Settings: {rates[i]}Hz, {channels_list[i]} channel(s)")
    print("Duration: Until stopped")
    print("-" * 80)

    stop_flag = threading.Event()

    class RecordingThread(threading.Thread):
        def __init__(self):
            super().__init__()
            self.daemon = True
            self.output_files = output_files
            self.error = None
            self.stop_flag = stop_flag
            self.completed = False
            self.saved_files = []

        def run(self):
            pa = pyaudio.PyAudio()

            try:
                streams = []

                for i, (device_index, channels, rate) in enumerate(zip(device_indices, channels_list, rates)):
                    try:
                        stream = pa.open(
                            format=pyaudio.paInt16,
                            channels=channels,
                            rate=rate,
                            input=True,
                            frames_per_buffer=1024,
                            input_device_index=device_index,
                        )
                        streams.append(stream)
                    except Exception as e:
                        for s in streams:
                            s.close()
                        # Don't call pa.terminate() - it causes crashes
                        self.error = f"Failed to open stream for device {i + 1}: {e}"
                        return

                queues = [queue.Queue() for _ in device_indices]
                stop_event = threading.Event()
                threads = []

                is_loopback = ["loopback" in name.lower() for name in device_names]

                def record_thread(stream, name, is_loop, channels, frames_queue, stop_evt):
                    silence = b"\x00" * (1024 * 2 * channels)
                    consecutive_errors = 0
                    max_consecutive_errors = 100

                    while not stop_evt.is_set():
                        try:
                            data = stream.read(1024, exception_on_overflow=False)
                            frames_queue.put(data)
                            consecutive_errors = 0
                        except Exception:
                            consecutive_errors += 1
                            frames_queue.put(silence)
                            if consecutive_errors >= max_consecutive_errors:
                                break
                            time.sleep(0.01)

                for i, (stream, name, is_loop, channels) in enumerate(
                    zip(streams, device_names, is_loopback, channels_list)
                ):
                    thread = threading.Thread(
                        target=record_thread, args=(stream, name, is_loop, channels, queues[i], stop_event)
                    )
                    thread.daemon = True
                    thread.start()
                    threads.append(thread)

                start_time = time.time()

                while not self.stop_flag.is_set():
                    time.sleep(0.1)

                stop_event.set()

                for thread in threads:
                    thread.join(timeout=2.0)

                elapsed_time = time.time() - start_time
                print(f"\n✓ Recording stopped after {elapsed_time:.1f} seconds")

                for i, (output_file, q) in enumerate(zip(output_files, queues)):
                    frames = []
                    while not q.empty():
                        try:
                            frames.append(q.get_nowait())
                        except Exception:
                            break

                    if len(frames) > 0:
                        audio_data = b"".join(frames)
                        with wave.open(output_file, "wb") as wf:
                            wf.setnchannels(channels_list[i])
                            wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                            wf.setframerate(rates[i])
                            wf.writeframes(audio_data)

                        # Calculate volume level
                        import numpy as np

                        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        volume_level = np.abs(audio_array).mean()

                        print(f"✓ Saved: {output_file} (Volume: {volume_level:.4f})")
                    else:
                        print(f"⚠ No audio data for device {i + 1}")

                # Stop streams (don't close - causes hangs on Windows with loopback devices)
                for i, stream in enumerate(streams):
                    try:
                        if stream.is_active():
                            stream.stop_stream()
                    except Exception:
                        pass

                # Store files in thread-local variable
                self.saved_files = [f for f in output_files if os.path.exists(f)]
                self.completed = True

            except Exception as e:
                self.error = str(e)
                print(f"❌ Recording thread error: {e}", flush=True)
                import traceback

                traceback.print_exc()
            finally:
                # Don't call pa.terminate() - it causes crashes on Windows
                # PyAudio instance will be garbage collected
                pass

    thread = RecordingThread()
    thread.start()
    return thread


def user_mode_ui(capture, devices):
    """
    Simple user interface with automatic device detection.

    Provides a streamlined experience with:
    - Automatic microphone and loopback device detection
    - Unlimited recording duration (start/stop buttons)
    - Automatic transcription after recording
    - Transcript history with copy functionality

    Args:
        capture: AudioCapture instance
        devices: List of available audio devices
    """
    st.title("Speech to Text - POC")

    mic_device = get_default_microphone(devices)
    loopback_devices = get_all_loopback_devices(devices)

    if not mic_device:
        st.error("❌ No microphone found")
        return

    # Single toggle button for Start/Stop Recording
    if not st.session_state.recording:
        if st.button("🔴 Start Recording", type="primary", use_container_width=True, key="record_btn"):
            st.session_state.recording = True
            st.session_state.stop_recording = False
            st.session_state.transcript = ""  # Clear previous transcript
            st.session_state.audio_files = []  # Clear previous audio files
            st.session_state.recording_complete = False  # Reset completion flag
            st.session_state.recording_start_time = time.time()
            st.rerun()
    else:
        if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True, key="record_btn"):
            if st.session_state.recording_thread:
                st.session_state.recording_thread.stop_flag.set()
            st.session_state.stop_recording = True
            st.rerun()

    st.markdown("---")

    if st.session_state.recording and not st.session_state.stop_recording:
        if st.session_state.recording_thread is None:
            device_indices = [mic_device["index"]]
            device_names = [mic_device["name"]]
            channels_list = [min(mic_device["maxInputChannels"], 2)]
            rates = [int(mic_device["defaultSampleRate"])]

            for loopback in loopback_devices:
                device_indices.append(loopback["index"])
                device_names.append(loopback["name"])
                channels_list.append(min(loopback["maxInputChannels"], 2))
                rates.append(int(loopback["defaultSampleRate"]))

            # Store device names for transcription
            st.session_state.device_names = device_names

            try:
                st.session_state.recording_thread = start_recording_thread(
                    device_indices, device_names, channels_list, rates
                )
            except Exception as e:
                st.error(f"Recording failed: {e}")
                print(f"\n❌ Error: {e}")
                st.session_state.recording = False
                st.session_state.recording_thread = None
                st.rerun()

        if st.session_state.recording_thread:
            if st.session_state.recording_thread.is_alive():
                elapsed = time.time() - st.session_state.recording_start_time
                st.info(f"🎙️ Recording... {elapsed:.1f}s")
                time.sleep(0.1)
                st.rerun()

    # Also check when stop_recording is True - the thread might still be finishing
    if st.session_state.recording and st.session_state.stop_recording:
        if st.session_state.recording_thread and st.session_state.recording_thread.is_alive():
            st.info("⏹️ Stopping recording...")
            time.sleep(0.1)
            st.rerun()

    # Check if recording thread has completed (whether stopped by user or naturally)
    if st.session_state.recording_thread and not st.session_state.recording_thread.is_alive():
        # Thread finished - check for errors or completion
        if st.session_state.recording_thread.error:
            st.error(f"Recording error: {st.session_state.recording_thread.error}")
            st.session_state.recording = False
            st.session_state.recording_thread = None
            st.rerun()
        elif st.session_state.recording_thread.completed:
            # Thread completed successfully, copy files and trigger transcription
            st.session_state.audio_files = st.session_state.recording_thread.saved_files
            st.session_state.recording_complete = True
            st.session_state.recording = False
            st.session_state.stop_recording = False
            st.session_state.recording_thread = None  # Clear thread reference to avoid re-checking
            st.rerun()
        else:
            # Wait a bit more for the thread to complete
            time.sleep(0.2)
            st.rerun()

    if st.session_state.recording_complete and st.session_state.audio_files:
        with st.spinner("Transcribing separate audio streams..."):
            print("\n" + "=" * 80)
            print("TRANSCRIBING SEPARATE AUDIO STREAMS")
            print("=" * 80)

            try:
                print("\n✓ Using cached Whisper model")

                # Transcribe each device separately with diarization
                result = st.session_state.transcriber.transcribe_multiple(
                    st.session_state.audio_files, st.session_state.device_names, diarizer=st.session_state.diarizer
                )

                st.session_state.transcript = result["combined_text"]

                # Save transcript to JSON file
                if st.session_state.transcript and st.session_state.transcript != "(No speech detected)":
                    save_transcript_json(result, st.session_state.audio_files)

                # Add to transcript history with timestamp
                if st.session_state.transcript and st.session_state.transcript != "(No speech detected)":
                    transcript_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "text": st.session_state.transcript,
                        "language": result["transcripts"][0]["language"] if result["transcripts"] else "unknown",
                        "separate_transcripts": result["transcripts"],
                    }
                    st.session_state.transcript_history.append(transcript_entry)

                print("\nTRANSCRIPTION RESULTS")
                print("-" * 50)
                if st.session_state.transcript:
                    print(st.session_state.transcript)
                else:
                    print("(No speech detected)")
                print("-" * 50)
                print("=" * 80)

            except Exception as e:
                st.error(f"Transcription failed: {e}")
                print(f"\n❌ Error: {e}")
                import traceback

                traceback.print_exc()

        # Generate summary if summarizer is available and transcript exists
        if (
            st.session_state.summarizer
            and st.session_state.transcript
            and st.session_state.transcript != "(No speech detected)"
        ):
            with st.spinner("Generating meeting summary..."):
                print("\n" + "=" * 80)
                print("GENERATING MEETING SUMMARY")
                print("=" * 80)

                try:
                    summary = st.session_state.summarizer.summarize(st.session_state.transcript)

                    if summary:
                        # Save summary to file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_summary(summary, timestamp)

                        # Store summary in transcript history entry
                        if st.session_state.transcript_history:
                            st.session_state.transcript_history[-1]["summary"] = summary

                        print("\nSUMMARY GENERATED")
                        print("-" * 50)
                        print(summary)
                        print("-" * 50)
                        print("=" * 80)
                    else:
                        print("⚠ Summary generation returned no result")

                except Exception as e:
                    print(f"⚠ Summary generation failed: {e}")
                    import traceback

                    traceback.print_exc()

        # Clear recording state but DON'T rerun - let the UI continue naturally
        st.session_state.recording = False
        st.session_state.stop_recording = False
        st.session_state.recording_thread = None
        st.session_state.recording_complete = False

    st.header("Transcripts")

    if st.session_state.transcript_history:
        st.success(f"✅ {len(st.session_state.transcript_history)} transcript(s) in this session")

        # Display all transcripts in reverse order (newest first)
        for i, entry in enumerate(reversed(st.session_state.transcript_history), 1):
            recording_num = len(st.session_state.transcript_history) - i + 1
            with st.expander(
                f"Recording #{recording_num} - {entry['timestamp']} ({entry['language']})", expanded=(i == 1)
            ):
                st.text_area(
                    "Transcribed text:",
                    value=entry["text"],
                    height=150,
                    key=f"transcript_{recording_num}",
                    disabled=False,
                    label_visibility="visible",
                )
                copy_button(entry["text"], tooltip="Copy to Clipboard", copied_label="✅ Copied!")

                # Display summary if available
                if "summary" in entry and entry["summary"]:
                    st.markdown("---")
                    st.subheader("📝 Meeting Summary")
                    st.markdown(entry["summary"])
                    copy_button(
                        entry["summary"],
                        tooltip="Copy Summary",
                        copied_label="✅ Copied!",
                        key=f"summary_copy_{entry.get('timestamp', recording_num)}",
                    )
    else:
        st.info("No transcripts yet. Start recording to generate transcripts.")


def dev_mode_ui(capture, devices):
    """
    Advanced developer interface with manual controls.

    Provides detailed control over:
    - Manual device selection (microphone and loopback)
    - Fixed recording duration (5-60 seconds)
    - Detailed terminal output for debugging

    Args:
        capture: AudioCapture instance
        devices: List of available audio devices
    """
    st.title("Speech to Text - POC")
    st.markdown("Advanced device selection and recording")

    categorized = categorize_devices(devices)

    st.header("Device Selection")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Microphone")
        input_options = {f"[{dev['index']}] {dev['name']}": dev["index"] for dev in categorized["input"]}
        if input_options:
            mic_selection = st.selectbox(
                "Select microphone device:", options=list(input_options.keys()), key="mic_select"
            )
            mic_index = input_options[mic_selection]
        else:
            st.warning("No input devices found")
            mic_index = None

    with col2:
        st.subheader("System Audio (Loopback)")
        loopback_options = {f"[{dev['index']}] {dev['name']}": dev["index"] for dev in categorized["loopback"]}
        if loopback_options:
            loopback_selection = st.selectbox(
                "Select loopback device:", options=list(loopback_options.keys()), key="loopback_select"
            )
            loopback_index = loopback_options[loopback_selection]
        else:
            st.warning("No loopback devices found")
            loopback_index = None

    st.markdown("---")
    st.header("Recording")

    duration = st.slider("Recording duration (seconds):", min_value=5, max_value=60, value=15, step=5)

    col1, col2 = st.columns([1, 3])

    with col1:
        if not st.session_state.recording:
            if st.button("🔴 Start Recording", type="primary", use_container_width=True):
                if mic_index is not None and loopback_index is not None:
                    st.session_state.recording = True
                    st.session_state.transcript = ""
                    st.rerun()
                else:
                    st.error("Please select both microphone and loopback devices")
        else:
            if st.button("⏹️ Stop & Reset", type="secondary", use_container_width=True):
                st.session_state.recording = False
                st.session_state.transcript = ""
                st.session_state.audio_files = []
                st.rerun()

    with col2:
        if st.session_state.recording:
            st.info("🎙️ Recording in progress... Check terminal for details")

    if st.session_state.recording and mic_index is not None and loopback_index is not None:
        with st.spinner("Recording audio..."):
            print("\n" + "=" * 80)
            print("RECORDING STARTED")
            print("=" * 80)

            mic_info = devices[mic_index]
            loopback_info = devices[loopback_index]

            device_indices = [mic_index, loopback_index]
            device_names = [mic_info["name"], loopback_info["name"]]
            channels_list = [min(mic_info["maxInputChannels"], 2), min(loopback_info["maxInputChannels"], 2)]
            rates = [int(mic_info["defaultSampleRate"]), int(loopback_info["defaultSampleRate"])]

            print(f"Device 1: {mic_info['name']}")
            print(f"Settings: {rates[0]}Hz, {channels_list[0]} channel(s)")
            print(f"Device 2: {loopback_info['name']}")
            print(f"Settings: {rates[1]}Hz, {channels_list[1]} channel(s)")
            print(f"Duration: {duration} seconds")
            print("-" * 80)

            try:
                audio_files = capture.record_multi_device(
                    device_indices=device_indices,
                    device_names=device_names,
                    channels_list=channels_list,
                    rates=rates,
                    duration=duration,
                    output_dir=OUTPUT_DIR,
                )

                st.session_state.audio_files = audio_files

                print("\n✓ Recordings saved:")
                for file in audio_files:
                    print(f"  - {file}")

            except Exception as e:
                st.error(f"Recording failed: {e}")
                print(f"\n❌ Error: {e}")
                st.session_state.recording = False
                st.rerun()

        with st.spinner("Transcribing separate audio streams..."):
            print("\n" + "=" * 80)
            print("TRANSCRIBING SEPARATE AUDIO STREAMS")
            print("=" * 80)

            try:
                print("\n✓ Using cached Whisper model")

                # Transcribe each device separately with diarization
                result = st.session_state.transcriber.transcribe_multiple(
                    st.session_state.audio_files, device_names, diarizer=st.session_state.diarizer
                )

                st.session_state.transcript = result["combined_text"]

                print("\nTRANSCRIPTION RESULTS")
                print("-" * 50)
                if st.session_state.transcript:
                    print(st.session_state.transcript)
                else:
                    print("(No speech detected)")
                print("-" * 50)
                print("=" * 80)

            except Exception as e:
                st.error(f"Transcription failed: {e}")
                print(f"\n❌ Error: {e}")
                import traceback

                traceback.print_exc()

        st.session_state.recording = False
        st.rerun()

    st.markdown("---")
    st.header("Transcript")

    if st.session_state.transcript:
        st.success("✅ Transcription complete!")
        st.text_area("Transcribed text:", value=st.session_state.transcript, height=200, key="transcript_display")

        if st.button("📋 Copy to Clipboard"):
            st.code(st.session_state.transcript, language=None)
            st.info("Select and copy the text above")
    else:
        st.info("No transcript yet. Start recording to generate a transcript.")

    st.markdown("---")
    st.caption("💡 Tip: Check the terminal for detailed progress and debug information")


def main():
    """
    Main Streamlit application entry point.

    Initializes session state, loads Whisper model, sets up diarization,
    and renders the appropriate UI mode (User or Dev).
    """
    initialize_session_state()

    capture = AudioCapture(frames_per_buffer=1024)
    devices = capture.list_devices()

    with st.sidebar:
        st.header("Settings")
        dev_mode = st.checkbox(
            "Dev Mode", value=st.session_state.dev_mode, help="Enable device selection and fixed duration recording"
        )

        if dev_mode != st.session_state.dev_mode:
            st.session_state.dev_mode = dev_mode
            st.session_state.recording = False
            st.session_state.stop_recording = False
            st.session_state.transcript = ""
            st.session_state.audio_files = []
            st.rerun()

        # Show detected devices in sidebar for user mode
        if not st.session_state.dev_mode:
            st.markdown("---")
            st.subheader("Detected Devices")

            mic_device = get_default_microphone(devices)
            loopback_devices = get_all_loopback_devices(devices)

            if mic_device:
                st.markdown(f"🎙️ **Microphone:**  \n{mic_device['name']}")
            else:
                st.warning("No microphone detected")

            if loopback_devices:
                st.markdown(f"🔊 **System Audio:**  \nRecording {len(loopback_devices)} loopback device(s)")
                with st.expander("View loopback devices"):
                    for i, dev in enumerate(loopback_devices, 1):
                        st.text(f"{i}. {dev['name']}")
            else:
                st.info("No loopback devices found")

    if st.session_state.dev_mode:
        dev_mode_ui(capture, devices)
    else:
        user_mode_ui(capture, devices)


if __name__ == "__main__":
    main()
