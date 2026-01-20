"""
Refactored Streamlit UI for speech-to-text recording and transcription.

This new version separates recording (client-side) from processing (server-side):
- Recording Page: Record audio locally and upload to API server
- Jobs Page: View ongoing and completed transcription jobs
- Transcript Page: View detailed results of completed transcriptions

The processing is now handled by a Flask API server with async queuing.
"""

import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from requests.exceptions import ConnectionError

from src.audio import AudioCapture, categorize_devices
from src.client import APIClient
from src.config import ConfigManager

# Configuration defaults
OUTPUT_DIR = "src/saved_audio"

# Page configuration
st.set_page_config(page_title="Speech-to-Text App", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")


def initialize_session_state():
    """Initialize session state variables."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Recording"
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "stop_recording" not in st.session_state:
        st.session_state.stop_recording = False
    if "stop_recording_event" not in st.session_state:
        st.session_state.stop_recording_event = threading.Event()
    if "recording_start_time" not in st.session_state:
        st.session_state.recording_start_time = None
    if "recording_thread" not in st.session_state:
        st.session_state.recording_thread = None
    if "audio_files" not in st.session_state:
        st.session_state.audio_files = []

    # API Settings with three-tier configuration
    # Tier 1: Defaults from ConfigManager.DEFAULTS
    # Tier 2: Environment variables from .env
    # Tier 3: UI overrides (stored in session state)

    # Initialize UI override values (empty string means "not set by user")
    if "ui_api_base_url" not in st.session_state:
        st.session_state.ui_api_base_url = ""  # Empty = not overridden by UI
    if "ui_llm_api_base_url" not in st.session_state:
        st.session_state.ui_llm_api_base_url = ""  # Empty = not overridden by UI
    if "ui_llm_model" not in st.session_state:
        st.session_state.ui_llm_model = ""  # Empty = not overridden by UI
    if "ui_llm_api_key" not in st.session_state:
        st.session_state.ui_llm_api_key = ""  # Empty = not overridden by UI
    if "ui_huggingface_token" not in st.session_state:
        st.session_state.ui_huggingface_token = ""  # Empty = not overridden by UI

    # Initialize API client with effective configuration (three-tier precedence)
    if "api_client" not in st.session_state:
        api_url = ConfigManager.get("API_BASE_URL", st.session_state.ui_api_base_url)
        st.session_state.api_client = APIClient(api_url)
    if "selected_job_id" not in st.session_state:
        st.session_state.selected_job_id = None
    if "jobs_cache" not in st.session_state:
        st.session_state.jobs_cache = []
    if "jobs_last_refresh" not in st.session_state:
        st.session_state.jobs_last_refresh = None

    # Model configuration UI overrides (empty string = use env/default)
    if "ui_whisper_model" not in st.session_state:
        st.session_state.ui_whisper_model = ""
    if "ui_diarization_model" not in st.session_state:
        st.session_state.ui_diarization_model = ""

    # Processing settings
    if "enable_diarization" not in st.session_state:
        st.session_state.enable_diarization = True
    if "enable_summarization" not in st.session_state:
        st.session_state.enable_summarization = True


def get_effective_config(key: str) -> str:
    """
    Get effective configuration value using three-tier precedence.

    Priority: UI override > Environment variable > Default
    """
    ui_key = f"ui_{key.lower()}"
    ui_override = st.session_state.get(ui_key, "")
    return ConfigManager.get(key, ui_override)


def get_api_base_url() -> str:
    """Get effective API base URL."""
    return get_effective_config("API_BASE_URL")


def get_llm_config() -> dict:
    """Get effective LLM configuration."""
    return {
        "base_url": get_effective_config("LLM_API_BASE_URL"),
        "model": get_effective_config("LLM_MODEL"),
        "api_key": get_effective_config("OPENAI_API_KEY"),
    }


def get_whisper_model() -> str:
    """Get effective Whisper model name."""
    return get_effective_config("WHISPER_MODEL")


def get_diarization_model() -> str:
    """Get effective diarization model name."""
    return get_effective_config("DIARIZATION_MODEL")


def check_api_connection() -> bool:
    """Check if the API server is accessible."""
    try:
        health = st.session_state.api_client.health_check()
        return health.get("status") == "healthy"
    except ConnectionError:
        return False


def get_default_microphone(devices):
    """Get the default microphone device."""
    categorized = categorize_devices(devices)

    for dev in categorized["input"]:
        if "microphone array" in dev["name"].lower():
            return dev

    if categorized["input"]:
        return categorized["input"][0]

    return None


def get_all_loopback_devices(devices):
    """Get all loopback devices for system audio capture."""
    categorized = categorize_devices(devices)
    return categorized["loopback"]


def get_local_recordings() -> list:
    """Get all local recording files with metadata."""
    recordings = []
    output_dir = Path(OUTPUT_DIR)

    if not output_dir.exists():
        return recordings

    # Find all audio files
    for ext in ["*.wav", "*.mp3", "*.m4a", "*.flac"]:
        for file_path in output_dir.glob(ext):
            if file_path.is_file():
                stat = file_path.stat()
                recordings.append(
                    {
                        "path": str(file_path),
                        "name": file_path.name,
                        "size": stat.st_size,
                        "size_mb": stat.st_size / (1024 * 1024),
                        "modified": datetime.fromtimestamp(stat.st_mtime),
                    }
                )

    # Sort by modification time (newest first)
    recordings.sort(key=lambda x: x["modified"], reverse=True)
    return recordings


def _init_recording_state():
    defaults = {
        "recording": False,
        "stop_recording": False,
        "recording_start_time": None,
        "audio_files": [],
        "stop_recording_event": None,
        "recording_result": None,
        "recording_thread": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.fragment
def _recording_controls_fragment(capture, mic_device, loopback_devices):
    """Start/Stop buttons + thread management (fragment reruns on click)."""

    if not st.session_state.recording:
        if st.button("🔴 Start Recording", type="primary", use_container_width=True):
            st.session_state.recording = True
            st.session_state.stop_recording = False
            st.session_state.recording_start_time = time.time()
            st.session_state.audio_files = []

            stop_event = threading.Event()
            result_container = {"audio_files": []}

            st.session_state.stop_recording_event = stop_event
            st.session_state.recording_result = result_container

            def record_audio(stop_event_param, result_param):
                devices_to_record = []
                if mic_device:
                    devices_to_record.append(mic_device)
                devices_to_record.extend(loopback_devices)

                try:
                    audio_files = capture.record_multiple_unlimited(
                        devices_to_record, lambda: stop_event_param.is_set()
                    )
                    result_param["audio_files"] = audio_files
                except Exception as e:
                    print(f"Recording error: {e}")
                    result_param["audio_files"] = []

            t = threading.Thread(
                target=record_audio,
                args=(stop_event, result_container),
                daemon=True,  # prevents hanging on app restart
            )
            st.session_state.recording_thread = t
            t.start()

            # If you want the rest of the app to immediately reflect recording=True
            st.rerun()

    else:
        if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True):
            st.session_state.stop_recording = True
            st.session_state.recording = False

            if st.session_state.stop_recording_event is not None:
                st.session_state.stop_recording_event.set()

            t = st.session_state.recording_thread
            if t is not None:
                t.join(timeout=5.0)

            result = st.session_state.recording_result or {}
            st.session_state.audio_files = result.get("audio_files", [])

            st.rerun()


@st.fragment(run_every="0.5s")
def _recording_status_fragment():
    """Auto-updating timer/status without full-script reruns."""
    if st.session_state.recording and st.session_state.recording_start_time:
        elapsed = time.time() - st.session_state.recording_start_time
        st.metric("Recording Time", f"{int(elapsed)}s")
        st.info("🔴 Recording in progress... Click 'Stop Recording' when finished.")
    elif st.session_state.audio_files and not st.session_state.recording:
        st.success("✅ Recording completed! You can now upload for processing.")


# --- Your modified page ------------------------------------------------------


def recording_page():
    """Main recording interface."""
    _init_recording_state()

    st.title("🎙️ Audio Recording")

    # Check API connection (but don't block recording)
    server_available = check_api_connection()

    if not server_available:
        st.warning(
            "⚠️ API server is offline. You can still record audio locally and upload later when server is available."
        )

    # Audio device setup
    capture = AudioCapture(frames_per_buffer=1024)
    devices = capture.list_devices()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Recording Controls")

        # Device information
        mic_device = get_default_microphone(devices)
        loopback_devices = get_all_loopback_devices(devices)

        if mic_device:
            st.success(f"🎙️ Microphone: {mic_device['name']}")
        else:
            st.error("❌ No microphone detected")
            return

        if loopback_devices:
            st.success(f"🔊 System Audio: {len(loopback_devices)} device(s) detected")
            with st.expander("View system audio devices"):
                for i, dev in enumerate(loopback_devices, 1):
                    st.text(f"{i}. {dev['name']}")
        else:
            st.warning("⚠️ No system audio devices found")

        # ✅ Fragments replace the manual sleep/rerun loop
        _recording_controls_fragment(capture, mic_device, loopback_devices)
        _recording_status_fragment()

    with col2:
        st.subheader("Processing Settings")

        # Get current effective model values
        current_whisper = get_whisper_model()
        current_diarization = get_diarization_model()

        # Processing options - these modify UI overrides
        whisper_options = ["tiny", "base", "small", "medium", "large"]
        try:
            current_index = whisper_options.index(current_whisper)
        except ValueError:
            # Custom model (HuggingFace ID or local path) - show in help text
            current_index = whisper_options.index("base")  # fallback
            st.info(f"Custom Whisper model: `{current_whisper}`. Override in Settings to change.")

        st.session_state.ui_whisper_model = st.selectbox(
            "Whisper Model",
            options=whisper_options,
            index=current_index,
            help="Larger models are more accurate but slower. Configure custom models in Settings.",
            key="whisper_model_selector",
        )

        st.session_state.enable_diarization = st.checkbox(
            "Speaker Diarization",
            value=st.session_state.enable_diarization,
            help=f"Identify different speakers using {current_diarization}",
        )

        st.session_state.enable_summarization = st.checkbox(
            "Generate Summary",
            value=st.session_state.enable_summarization,
            help="Create meeting summary (requires OpenAI API key)",
        )

    # Upload section
    if st.session_state.audio_files and not st.session_state.recording:
        st.markdown("---")
        st.subheader("Upload for Processing")

        # Show recorded files info
        valid_files = [f for f in st.session_state.audio_files if os.path.exists(f)]
        total_size = sum(os.path.getsize(f) for f in valid_files)
        total_size_mb = total_size / (1024 * 1024)

        st.info(f"📊 Recorded {len(valid_files)} audio file(s) - Total size: {total_size_mb:.1f} MB")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📁 Show Saved Files", use_container_width=True):
                st.success(f"✅ {len(valid_files)} audio file(s) already saved")
                for file in valid_files:
                    st.text(f"📁 {os.path.basename(file)}")
                    st.caption(f"Full path: {file}")

        with col2:
            # Only show upload button if server is available
            if server_available:
                if st.button("🚀 Upload & Process", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Uploading audio file..."):
                            if not valid_files:
                                st.error("❌ No audio files to upload")
                                return

                            # Use the first file for processing (usually microphone)
                            main_audio_file = valid_files[0]

                            # Prepare processing options
                            options = {
                                "whisper_model": get_whisper_model(),
                                "diarization_model": get_diarization_model(),
                                "enable_diarization": st.session_state.enable_diarization,
                                "enable_summarization": st.session_state.enable_summarization,
                            }

                            # Add HuggingFace token if available (needed for model downloads)
                            hf_token = get_effective_config("HUGGINGFACE_TOKEN")
                            if hf_token:
                                options["huggingface_token"] = hf_token

                            # Add LLM settings if summarization is enabled
                            if st.session_state.enable_summarization:
                                llm_config = get_llm_config()
                                if llm_config["base_url"]:
                                    options["llm_api_base_url"] = llm_config["base_url"]
                                if llm_config["api_key"]:
                                    options["llm_api_key"] = llm_config["api_key"]
                                if llm_config["model"]:
                                    options["llm_model"] = llm_config["model"]

                            # Upload to server
                            result = st.session_state.api_client.upload_audio_file(main_audio_file, options)

                            job_id = result["job_id"]
                            st.success(f"✅ Uploaded! Job ID: {job_id}")
                            st.info("📋 Check the 'Jobs' page to monitor progress.")

                            # Clear the recording
                            st.session_state.audio_files = []

                            # Switch to jobs page
                            time.sleep(2)
                            st.session_state.current_page = "Jobs"
                            st.rerun()

                    except Exception as e:
                        st.error(f"❌ Upload failed: {e}")
            else:
                st.button(
                    "🚀 Upload & Process",
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                    help="Server offline - upload not available",
                )

        with col3:
            if st.button("🗑️ Clear Recording", use_container_width=True):
                st.session_state.audio_files = []
                st.rerun()

    # Local recordings section
    st.markdown("---")
    st.subheader("📂 Local Recordings")

    local_recordings = get_local_recordings()

    if not local_recordings:
        st.info("No local recordings found. Record audio above to save files locally.")
    else:
        st.success(f"Found {len(local_recordings)} local recording(s)")

        # Display recordings in a table-like format
        for idx, recording in enumerate(local_recordings):
            with st.expander(
                f"📁 {recording['name']} ({recording['size_mb']:.1f} MB) - {recording['modified'].strftime('%Y-%m-%d %H:%M')}"
            ):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.caption(f"**Path:** {recording['path']}")
                    st.caption(f"**Size:** {recording['size_mb']:.2f} MB")
                    st.caption(f"**Modified:** {recording['modified'].strftime('%Y-%m-%d %H:%M:%S')}")

                with col2:
                    if server_available:
                        if st.button("🚀 Upload", key=f"upload_{idx}", use_container_width=True):
                            try:
                                with st.spinner("Uploading..."):
                                    # Prepare processing options
                                    options = {
                                        "whisper_model": get_whisper_model(),
                                        "diarization_model": get_diarization_model(),
                                        "enable_diarization": st.session_state.enable_diarization,
                                        "enable_summarization": st.session_state.enable_summarization,
                                    }

                                    # Add HuggingFace token if available
                                    hf_token = get_effective_config("HUGGINGFACE_TOKEN")
                                    if hf_token:
                                        options["huggingface_token"] = hf_token

                                    # Add LLM settings if summarization is enabled
                                    if st.session_state.enable_summarization:
                                        llm_config = get_llm_config()
                                        if llm_config["base_url"]:
                                            options["llm_api_base_url"] = llm_config["base_url"]
                                        if llm_config["api_key"]:
                                            options["llm_api_key"] = llm_config["api_key"]
                                        if llm_config["model"]:
                                            options["llm_model"] = llm_config["model"]

                                    # Upload to server
                                    result = st.session_state.api_client.upload_audio_file(recording["path"], options)

                                    job_id = result["job_id"]
                                    st.success(f"✅ Uploaded! Job ID: {job_id}")
                                    time.sleep(2)
                                    st.session_state.current_page = "Jobs"
                                    st.rerun()

                            except Exception as e:
                                st.error(f"❌ Upload failed: {e}")
                    else:
                        st.button(
                            "🚀 Upload",
                            key=f"upload_{idx}",
                            use_container_width=True,
                            disabled=True,
                            help="Server offline",
                        )


def jobs_page():
    """Jobs listing and monitoring interface."""
    st.title("📋 Transcription Jobs")

    # Check API connection
    if not check_api_connection():
        api_url = get_api_base_url()
        st.error(f"⚠️ Cannot connect to API server at {api_url}")
        return

    # Refresh controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.jobs_cache = []
            st.session_state.jobs_last_refresh = None
            st.rerun()

    with col2:
        auto_refresh = st.checkbox("Auto Refresh", value=False)

    with col3:
        status_filter = st.selectbox("Filter by Status", options=["All", "queued", "processing", "completed", "failed"])
        if status_filter == "All":
            status_filter = None

    # Fetch jobs (with caching)
    should_refresh = (
        not st.session_state.jobs_cache
        or not st.session_state.jobs_last_refresh
        or (datetime.now() - st.session_state.jobs_last_refresh).seconds > 30
    )

    if should_refresh or auto_refresh:
        try:
            with st.spinner("Loading jobs..."):
                jobs_data = st.session_state.api_client.list_jobs(status_filter=status_filter, limit=50)
                st.session_state.jobs_cache = jobs_data["jobs"]
                st.session_state.jobs_last_refresh = datetime.now()
        except Exception as e:
            st.error(f"❌ Failed to load jobs: {e}")
            return

    jobs = st.session_state.jobs_cache

    if not jobs:
        st.info("📭 No jobs found. Start by recording and uploading audio on the Recording page.")
        return

    # Jobs summary
    status_counts = {}
    for job in jobs:
        status = job.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Jobs", len(jobs))
    with col2:
        st.metric("Completed", status_counts.get("completed", 0))
    with col3:
        st.metric("Processing", status_counts.get("processing", 0))
    with col4:
        st.metric("Queued", status_counts.get("queued", 0))

    # Jobs table
    st.subheader("Jobs List")

    for job in jobs:
        status = job.get("status", "unknown")

        # Status emoji
        status_emoji = {"queued": "⏳", "processing": "⚙️", "completed": "✅", "failed": "❌"}.get(status, "❓")

        # Create expandable job card
        with st.expander(
            f"{status_emoji} {job.get('original_filename', 'Unknown')} - {status.upper()}", expanded=False
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.text(f"Job ID: {job.get('id', 'Unknown')}")
                st.text(f"Created: {job.get('created_at', 'Unknown')}")
                st.text(f"Status: {status}")

                if "file_size" in job:
                    file_size_mb = int(job["file_size"]) / (1024 * 1024)
                    st.text(f"File Size: {file_size_mb:.1f} MB")

                if status == "failed" and "error" in job:
                    st.error(f"Error: {job['error']}")

            with col2:
                button_col1, button_col2, button_col3 = st.columns(3)

                with button_col1:
                    if status == "completed":
                        if st.button("👁️ View", key=f"view_{job['id']}"):
                            st.session_state.selected_job_id = job["id"]
                            st.session_state.current_page = "Transcript"
                            st.rerun()

                with button_col2:
                    if status == "completed":
                        try:
                            # Get the result data for download
                            result_data = st.session_state.api_client.get_job_result(job["id"])
                            json_data = json.dumps(result_data, ensure_ascii=False, indent=2)

                            # Determine filename
                            original_filename = result_data.get("metadata", {}).get("original_filename", "transcript")
                            download_filename = f"{original_filename}_transcript.json"

                            st.download_button(
                                label="⬇️ Download",
                                data=json_data,
                                file_name=download_filename,
                                mime="application/json",
                                key=f"download_{job['id']}",
                            )
                        except Exception as e:
                            st.error(f"❌ Failed to prepare download: {e}")

                with button_col3:
                    if st.button("🗑️ Delete", key=f"delete_{job['id']}"):
                        try:
                            with st.spinner("Deleting..."):
                                st.session_state.api_client.delete_job(job["id"])
                                st.success("✅ Job deleted")
                                # Clear cache to refresh
                                st.session_state.jobs_cache = []
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Delete failed: {e}")

    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()


def transcript_page():
    """Detailed transcript view."""
    st.title("📄 Transcript Details")

    if not st.session_state.selected_job_id:
        st.info("📭 No transcript selected. Go to the Jobs page and click 'View' on a completed job.")
        return

    # Check API connection
    if not check_api_connection():
        api_url = get_api_base_url()
        st.error(f"⚠️ Cannot connect to API server at {api_url}")
        return

    job_id = st.session_state.selected_job_id

    # Fetch job result
    try:
        with st.spinner("Loading transcript..."):
            result = st.session_state.api_client.get_job_result(job_id)
    except Exception as e:
        st.error(f"❌ Failed to load transcript: {e}")
        return

    # Header with job info
    metadata = result.get("metadata", {})
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Job ID", job_id[:8] + "...")
    with col2:
        st.metric("File", metadata.get("original_filename", "Unknown"))
    with col3:
        if "processing_time" in metadata:
            processing_time = float(metadata["processing_time"])
            st.metric("Processing Time", f"{processing_time:.1f}s")

    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Transcript", "👥 Segments", "📋 Summary", "ℹ️ Details"])

    with tab1:
        st.subheader("Full Transcript")
        transcript_text = result.get("transcript", "")

        if transcript_text:
            st.text_area("Transcript", transcript_text, height=400, label_visibility="collapsed")

            # Copy button
            if st.button("📋 Copy to Clipboard"):
                st.code(transcript_text, language=None)
                st.info("Select and copy the text above")
        else:
            st.info("No transcript available.")

    with tab2:
        st.subheader("Timestamped Segments")
        segments = result.get("segments", [])

        if segments:
            for i, segment in enumerate(segments):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "")
                speaker = segment.get("speaker")

                # Format timestamp
                start_time = f"{int(start // 60):02d}:{int(start % 60):02d}"
                end_time = f"{int(end // 60):02d}:{int(end % 60):02d}"

                # Speaker label
                speaker_label = f"[{speaker}]" if speaker else "[Unknown]"

                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.text(f"{start_time} - {end_time}")
                        speaker_label = f"[{speaker}]" if speaker else "[Unknown]"
                        st.caption(speaker_label)
                    with col2:
                        st.text(text.strip() if text else "(no text)")

                if i < len(segments) - 1:
                    st.divider()
        else:
            st.info("No segments available.")

    with tab3:
        st.subheader("Meeting Summary")
        summary = result.get("summary")

        if summary:
            st.markdown(summary)

            # Copy button for summary
            if st.button("📋 Copy Summary", key="copy_summary"):
                st.code(summary, language=None)
                st.info("Select and copy the summary above")
        else:
            st.info("No summary available. Make sure summarization was enabled during processing.")

    with tab4:
        st.subheader("Processing Details")

        # Job metadata
        st.json(metadata)

        # Processing metadata
        if "processing_metadata" in result:
            st.subheader("Processing Information")
            st.json(result["processing_metadata"])

    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅️ Back to Jobs", use_container_width=True):
            st.session_state.current_page = "Jobs"
            st.rerun()

    with col2:
        try:
            # Prepare download data
            json_data = json.dumps(result, ensure_ascii=False, indent=2)
            original_filename = result.get("metadata", {}).get("original_filename", "transcript")
            download_filename = f"{original_filename}_transcript.json"

            st.download_button(
                label="⬇️ Download JSON",
                data=json_data,
                file_name=download_filename,
                mime="application/json",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"❌ Failed to prepare download: {e}")

    with col3:
        # Initialize delete confirmation state
        if "confirm_delete" not in st.session_state:
            st.session_state.confirm_delete = False

        if not st.session_state.confirm_delete:
            if st.button("🗑️ Delete Job", use_container_width=True, type="secondary"):
                st.session_state.confirm_delete = True
                st.rerun()
        else:
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("⚠️ Confirm", use_container_width=True, type="primary"):
                    try:
                        with st.spinner("Deleting..."):
                            st.session_state.api_client.delete_job(job_id)
                            st.session_state.confirm_delete = False
                            st.success("✅ Job deleted")
                            st.session_state.selected_job_id = None
                            st.session_state.current_page = "Jobs"
                            st.rerun()
                    except Exception as e:
                        st.session_state.confirm_delete = False
                        st.error(f"❌ Delete failed: {e}")
            with col_cancel:
                if st.button("✖️ Cancel", use_container_width=True):
                    st.session_state.confirm_delete = False
                    st.rerun()


def settings_page():
    """Settings page for API and LLM configuration with three-tier precedence."""
    st.title("⚙️ Settings")

    st.markdown(
        """
    Configure the API endpoints and LLM settings for the application.
    
    **Configuration Precedence:**
    1. 🔵 **Default** - Built-in codebase defaults
    2. 🟢 **Environment** - Values from `.env` file
    3. 🟡 **UI Override** - Custom values set here (highest priority)
    
    Leave fields empty to use environment or default values.
    """
    )

    # API Server Settings
    st.header("🌐 API Server Settings")

    # Get current effective value and source
    current_api_url, api_source = ConfigManager.get_display_value("API_BASE_URL", st.session_state.ui_api_base_url)

    # Show current effective configuration
    source_icon = {"default": "🔵", "env": "🟢", "ui": "🟡"}[api_source]
    source_label = {"default": "Default", "env": "Environment (.env)", "ui": "UI Override"}[api_source]

    st.info(f"{source_icon} **Currently using:** `{current_api_url}` (from {source_label})")

    # Show environment variable name
    st.caption("💡 **Environment variable:** `API_BASE_URL` (set in `.env` file)")

    api_url_input = st.text_input(
        "Server API Endpoint Override",
        value=st.session_state.ui_api_base_url,
        help="Leave empty to use environment variable or default. Set a value to override.",
        placeholder=f"Empty = use {'env' if api_source == 'env' else 'default'}: {current_api_url}",
    )

    if api_url_input != st.session_state.ui_api_base_url:
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("✅ Apply", use_container_width=True, type="primary"):
                st.session_state.ui_api_base_url = api_url_input
                # Recreate API client with new effective URL
                new_url = ConfigManager.get("API_BASE_URL", api_url_input)
                st.session_state.api_client = APIClient(new_url)
                st.success("✅ API server endpoint updated!")
                st.rerun()
        with col2:
            if st.button("↩️ Cancel", use_container_width=True):
                st.rerun()

    # Test connection
    if st.button("🔍 Test Connection", use_container_width=False):
        with st.spinner("Testing connection..."):
            if check_api_connection():
                st.success("✅ Successfully connected to API server!")
            else:
                st.error("❌ Failed to connect to API server. Please check the URL and server status.")

    st.markdown("---")

    # LLM Settings
    st.header("🤖 LLM Settings for Summarization")

    st.markdown("Configure the LLM endpoint used for meeting summarization.")

    # LLM Base URL
    current_llm_url, llm_url_source = ConfigManager.get_display_value(
        "LLM_API_BASE_URL", st.session_state.ui_llm_api_base_url
    )
    source_icon = {"default": "🔵", "env": "🟢", "ui": "🟡"}[llm_url_source]
    source_label = {"default": "Default", "env": "Environment (.env)", "ui": "UI Override"}[llm_url_source]
    st.info(f"{source_icon} **LLM Base URL:** `{current_llm_url or '(using OpenAI default)'}` (from {source_label})")
    st.caption("💡 **Environment variable:** `LLM_API_BASE_URL`")

    llm_base_url = st.text_input(
        "LLM API Base URL Override",
        value=st.session_state.ui_llm_api_base_url,
        help="Custom API base URL for LLM provider. Leave empty to use environment/default.",
        placeholder=f"Empty = use {'env' if llm_url_source == 'env' else 'default'}",
    )

    # LLM Model
    current_llm_model, llm_model_source = ConfigManager.get_display_value("LLM_MODEL", st.session_state.ui_llm_model)
    source_icon = {"default": "🔵", "env": "🟢", "ui": "🟡"}[llm_model_source]
    source_label = {"default": "Default", "env": "Environment (.env)", "ui": "UI Override"}[llm_model_source]
    st.info(f"{source_icon} **LLM Model:** `{current_llm_model}` (from {source_label})")
    st.caption("💡 **Environment variable:** `LLM_MODEL`")

    llm_model = st.text_input(
        "LLM Model Override",
        value=st.session_state.ui_llm_model,
        help="Model name to use for summarization. Leave empty to use environment/default.",
        placeholder=f"Empty = use {'env' if llm_model_source == 'env' else 'default'}: {current_llm_model}",
    )

    # API Key
    current_api_key, api_key_source = ConfigManager.get_display_value("OPENAI_API_KEY", st.session_state.ui_llm_api_key)
    source_icon = {"default": "🔵", "env": "🟢", "ui": "🟡"}[api_key_source]
    source_label = {"default": "Default", "env": "Environment (.env)", "ui": "UI Override"}[api_key_source]
    key_display = "***" + current_api_key[-4:] if current_api_key else "(not set)"
    st.info(f"{source_icon} **API Key:** `{key_display}` (from {source_label})")
    st.caption("💡 **Environment variable:** `OPENAI_API_KEY`")

    llm_api_key = st.text_input(
        "LLM API Key Override",
        value=st.session_state.ui_llm_api_key,
        type="password",
        help="API key for authentication. Leave empty to use environment/default.",
        placeholder="Empty = use environment variable",
    )

    # Check if any LLM settings changed
    llm_changed = (
        llm_base_url != st.session_state.ui_llm_api_base_url
        or llm_api_key != st.session_state.ui_llm_api_key
        or llm_model != st.session_state.ui_llm_model
    )

    if llm_changed:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("✅ Save LLM Settings", use_container_width=True, type="primary", key="save_llm"):
                st.session_state.ui_llm_api_base_url = llm_base_url
                st.session_state.ui_llm_api_key = llm_api_key
                st.session_state.ui_llm_model = llm_model
                st.success("✅ LLM settings saved! These will be used for new summarization jobs.")
                st.rerun()
        with col2:
            if st.button("↩️ Cancel", use_container_width=True, key="reset_llm"):
                st.rerun()

    st.markdown("---")

    # Model Settings
    st.header("🤖 Model Configuration")

    st.markdown("Configure the models used for transcription and speaker diarization.")

    # Whisper Model
    current_whisper, whisper_source = ConfigManager.get_display_value(
        "WHISPER_MODEL", st.session_state.ui_whisper_model
    )
    source_icon = {"default": "🔵", "env": "🟢", "ui": "🟡"}[whisper_source]
    source_label = {"default": "Default", "env": "Environment (.env)", "ui": "UI Override"}[whisper_source]
    st.info(f"{source_icon} **Whisper Model:** `{current_whisper}` (from {source_label})")
    st.caption("💡 **Environment variable:** `WHISPER_MODEL`")

    whisper_model_input = st.text_input(
        "Whisper Model Override",
        value=st.session_state.ui_whisper_model,
        help="OpenAI model (tiny/base/small/medium/large), HuggingFace ID (openai/whisper-large-v3), or local path. Leave empty for env/default.",
        placeholder=f"Empty = use {'env' if whisper_source == 'env' else 'default'}: {current_whisper}",
    )

    # Diarization Model
    current_diarization, diarization_source = ConfigManager.get_display_value(
        "DIARIZATION_MODEL", st.session_state.ui_diarization_model
    )
    source_icon = {"default": "🔵", "env": "🟢", "ui": "🟡"}[diarization_source]
    source_label = {"default": "Default", "env": "Environment (.env)", "ui": "UI Override"}[diarization_source]
    st.info(f"{source_icon} **Diarization Model:** `{current_diarization}` (from {source_label})")
    st.caption("💡 **Environment variable:** `DIARIZATION_MODEL`")

    diarization_model_input = st.text_input(
        "Diarization Model Override",
        value=st.session_state.ui_diarization_model,
        help="HuggingFace model ID or local path for PyAnnote diarization. Leave empty for env/default.",
        placeholder=f"Empty = use {'env' if diarization_source == 'env' else 'default'}: {current_diarization}",
    )

    # HuggingFace Token
    st.markdown("---")
    st.subheader("🔑 HuggingFace Token")
    st.markdown("Required for downloading models from HuggingFace (diarization, custom Whisper models).")

    current_hf_token, hf_token_source = ConfigManager.get_display_value(
        "HUGGINGFACE_TOKEN", st.session_state.ui_huggingface_token
    )
    source_icon = {"default": "🔵", "env": "🟢", "ui": "🟡"}[hf_token_source]
    source_label = {"default": "Default", "env": "Environment (.env)", "ui": "UI Override"}[hf_token_source]
    token_display = "***" + current_hf_token[-4:] if current_hf_token else "(not set)"
    st.info(f"{source_icon} **HuggingFace Token:** `{token_display}` (from {source_label})")
    st.caption("💡 **Environment variable:** `HUGGINGFACE_TOKEN`")

    hf_token_input = st.text_input(
        "HuggingFace Token Override",
        value=st.session_state.ui_huggingface_token,
        type="password",
        help="HuggingFace API token for accessing gated models. Leave empty to use environment/default.",
        placeholder="Empty = use environment variable",
    )

    # Check if any model settings changed
    models_changed = (
        whisper_model_input != st.session_state.ui_whisper_model
        or diarization_model_input != st.session_state.ui_diarization_model
        or hf_token_input != st.session_state.ui_huggingface_token
    )

    if models_changed:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("✅ Save Model Settings", use_container_width=True, type="primary", key="save_models"):
                st.session_state.ui_whisper_model = whisper_model_input
                st.session_state.ui_diarization_model = diarization_model_input
                st.session_state.ui_huggingface_token = hf_token_input
                st.success("✅ Model settings saved! These will be used for new processing jobs.")
                st.rerun()
        with col2:
            if st.button("↩️ Cancel", use_container_width=True, key="reset_models"):
                st.rerun()

    st.markdown("---")

    # Show all effective configuration
    st.header("📊 Current Effective Configuration")

    config_data = {
        "API Server URL": get_effective_config("API_BASE_URL"),
        "LLM Base URL": get_effective_config("LLM_API_BASE_URL") or "(OpenAI default)",
        "LLM Model": get_effective_config("LLM_MODEL"),
        "Whisper Model": get_effective_config("WHISPER_MODEL"),
        "Diarization Model": get_effective_config("DIARIZATION_MODEL"),
        "OpenAI API Key": "***" + get_effective_config("OPENAI_API_KEY")[-4:]
        if get_effective_config("OPENAI_API_KEY")
        else "(not set)",
        "HuggingFace Token": "***" + get_effective_config("HUGGINGFACE_TOKEN")[-4:]
        if get_effective_config("HUGGINGFACE_TOKEN")
        else "(not set)",
    }

    for key, value in config_data.items():
        st.text(f"{key}: {value}")

    st.markdown("---")

    # Current Settings Summary
    st.header("📋 Current Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("API Server")
        st.code(get_api_base_url())

    with col2:
        st.subheader("LLM Configuration")
        llm_config = get_llm_config()
        if llm_config["base_url"]:
            st.code(f"Base URL: {llm_config['base_url']}")
        else:
            st.code("Base URL: Default (OpenAI)")
        st.code(f"Model: {llm_config['model']}")
        if llm_config["api_key"]:
            st.code(f"API Key: {'*' * 8}{llm_config['api_key'][-4:]}")
        else:
            st.warning("⚠️ No API key configured")

    # Info box
    st.info(
        """
    **💡 Note:** 
    - The Server API Endpoint points to your Flask backend server
    - LLM settings are used when you enable summarization in recording options
    - Changes to LLM settings will only affect new transcription jobs
    - API keys are stored in session state and not persisted
    """
    )


def main():
    """Main application entry point."""
    initialize_session_state()

    # Sidebar navigation
    with st.sidebar:
        st.title("🎙️ Speech-to-Text")

        # Page selection
        page = st.radio(
            "Navigation",
            options=["Recording", "Jobs", "Transcript", "Settings"],
            index=["Recording", "Jobs", "Transcript", "Settings"].index(st.session_state.current_page),
        )

        if page != st.session_state.current_page:
            st.session_state.current_page = page
            st.rerun()

        st.markdown("---")

        # API server status
        st.subheader("Server Status")
        if check_api_connection():
            st.success("🟢 API Server Connected")
        else:
            st.error("🔴 API Server Disconnected")

        st.caption(f"API URL: {get_api_base_url()}")

    # Main content area
    if st.session_state.current_page == "Recording":
        recording_page()
    elif st.session_state.current_page == "Jobs":
        jobs_page()
    elif st.session_state.current_page == "Transcript":
        transcript_page()
    elif st.session_state.current_page == "Settings":
        settings_page()


if __name__ == "__main__":
    main()
