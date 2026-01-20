# Speech to Text POC

A meeting capture solution that records both system audio (from Teams, Zoom, Youtube etc.) and microphone input simultaneously. The application uses OpenAI Whisper for speech-to-text transcription and PyAnnote for speaker diarization, enabling detailed meeting transcripts with speaker identification and timestamps. The transcript is then used to generate a meeting minutes using OpenAI API with GPT-4.

## Table of Contents
- [Speech to Text POC](#speech-to-text-poc)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Configure Speaker Diarization](#configure-speaker-diarization)
    - [Configure Meeting Summarization](#configure-meeting-summarization)
    - [Running the Application](#running-the-application)
  - [Speech to Text Workflow](#speech-to-text-workflow)
    - [How It Works](#how-it-works)
  - [Testing \& Development](#testing--development)
    - [Available Test Scripts](#available-test-scripts)
    - [Code Formatting](#code-formatting)
  - [Whisper Model Comparison](#whisper-model-comparison)
  - [Limitations](#limitations)
  - [Development Challenges](#development-challenges)
    - [Audio Device Detection](#audio-device-detection)
    - [ASR Methodology](#asr-methodology)
    - [PyAnnote Dependencies](#pyannote-dependencies)
    - [Hallucination Filtering](#hallucination-filtering)
    - [Combining Transcription and Diarization](#combining-transcription-and-diarization)
    - [Audio Corruption when adding Diarization](#audio-corruption-when-adding-diarization)
  - [Future Improvements](#future-improvements)
    - [Core: spch2txt](#core-spch2txt)
    - [UI/UX Enhancements](#uiux-enhancements)
    - [Post-Processing with LLM](#post-processing-with-llm)
    - [Packaging and Deployment](#packaging-and-deployment)
    - [Audio and Timestamp Logic](#audio-and-timestamp-logic)
    - [Transcription Quality](#transcription-quality)

## Features

- **Client-Server Architecture**: Separate Flask API server for processing and Streamlit UI for interaction
- **Three-Tier Configuration**: Flexible endpoint configuration (Defaults → Environment → UI Settings)
- **Configurable Models**: Support for OpenAI Whisper, HuggingFace models, and local models for both transcription and diarization
- **System Audio Recording**: Capture audio from Teams, Zoom, Youtube, or any system audio playing on your device using WASAPI loopback
- **Microphone Recording**: Simultaneous microphone and system audio capture
- **Transcription**: Transcription from speech to text using OpenAI Whisper or custom models
- **Speaker Diarization**: Identify different speakers using pyannote.audio (configurable models)
- **Meeting Summarization**: Automatic generation of meeting minutes using OpenAI GPT-4 or custom LLM endpoints
- **Multiple Whisper Models**: Support for tiny, base, small, medium, large, plus HuggingFace and local models
- **Export Transcripts**: Save transcriptions as JSON files with metadata and summaries as text files
- **Simple UI**: Built with Streamlit UI for easy use

## Quick Start
Warning: This app is only meant for Windows operating systems.

### Prerequisites

1. **Python 3.13** (recommended)
2. **Poetry** for dependency management
3. **Conda** for environment management

### Installation

```bash
# Create conda environment
conda create -n spch2txt "python=3.13"
conda activate spch2txt

# Install Poetry
pip install poetry

# Install dependencies
poetry install --no-root
```

### Configuration System

The application uses a **three-tier configuration precedence system**:

1. 🔵 **Defaults** - Built-in codebase defaults (lowest priority)
2. 🟢 **Environment Variables** - Values from `.env` file (medium priority)
3. 🟡 **UI Settings** - Custom values set in the app's Settings page (highest priority)

This allows you to configure the application at different levels based on your needs.

### Essential Configuration

**Quick Start:**
Copy the example configuration file and add your API keys:
```bash
cp .env.example .env
```

Then edit `.env` and add your keys (see `.env.example` for all available options).

**Minimal Required Configuration:**

**1. HuggingFace Token (Required for Speaker Diarization)**
   - Accept user conditions for pyannote models:
     - https://huggingface.co/pyannote/segmentation-3.0
     - https://huggingface.co/pyannote/speaker-diarization-3.1
   - Create a Hugging Face access token (READ permissions) at https://hf.co/settings/tokens
   - Add to `.env`:
     ```bash
     HUGGINGFACE_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234567890
     ```

**2. OpenAI API Key (Required for Meeting Summarization)**
   - Create an OpenAI API key at https://platform.openai.com/api-keys
   - Add to `.env`:
     ```bash
     OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz1234567890
     ```

### Advanced Configuration (Optional)

All settings can be configured via `.env` file or UI Settings page.

**Complete `.env` Example:**

```bash
# ============================================================================
# API ENDPOINTS
# ============================================================================

# Server API endpoint (where the Flask backend runs)
API_BASE_URL=http://localhost:5001

# LLM API endpoint for summarization
# - Use default OpenAI: https://api.openai.com/v1
# - Or custom endpoint: http://your-server:8000/v1 (vLLM, Ollama, etc.)
LLM_API_BASE_URL=https://api.openai.com/v1

# ============================================================================
# API KEYS & TOKENS
# ============================================================================

# OpenAI API key (get from: https://platform.openai.com/api-keys)
OPENAI_API_KEY=sk-proj-your_openai_api_key_here

# HuggingFace token (get from: https://hf.co/settings/tokens)
# Required for diarization and HuggingFace models
HUGGINGFACE_TOKEN=hf_your_huggingface_token_here

# ============================================================================
# LLM MODEL CONFIGURATION
# ============================================================================

# Model to use for meeting summarization
# Examples:
#   - gpt-4o (recommended)
#   - gpt-4-turbo
#   - gpt-3.5-turbo
#   - claude-3-opus-20240229 (if using Anthropic API)
#   - llama-3-70b-instruct (if using vLLM/Ollama)
LLM_MODEL=gpt-4o

# ============================================================================
# WHISPER MODEL CONFIGURATION
# ============================================================================

# Whisper model for speech-to-text transcription
# 
# Three-tier precedence (highest to lowest):
#   1. Client request (whisper_model in API options) - per-request override
#   2. Environment variable (WHISPER_MODEL below) - server default
#   3. Hardcoded default ("base") - fallback if not specified
#
# OpenAI Whisper models (built-in, no prefix needed):
#   - tiny    (fastest, least accurate, ~39MB)
#   - base    (default, good balance, ~74MB)
#   - small   (better quality, ~244MB)
#   - medium  (high quality, ~769MB)
#   - large   (best quality, ~1550MB)
#
# HuggingFace models (requires transformers library):
#   - openai/whisper-tiny
#   - openai/whisper-base
#   - openai/whisper-small
#   - openai/whisper-medium
#   - openai/whisper-large-v3
#   - openai/whisper-large-v3-turbo
#
# Local model (absolute path):
#   - /Users/username/models/whisper-large
#   - /home/user/ml-models/custom-whisper
#
WHISPER_MODEL=base

# ============================================================================
# DIARIZATION MODEL CONFIGURATION
# ============================================================================

# PyAnnote model for speaker diarization (who spoke when)
# Requires HUGGINGFACE_TOKEN to be set
#
# IMPORTANT: The diarization pipeline uses multiple models internally:
#   - Main pipeline: pyannote/speaker-diarization-3.1
#   - Segmentation: pyannote/segmentation-3.0 (auto-downloaded)
#   - Embedding: pyannote/wespeaker-voxceleb-resnet34-LM (auto-downloaded)
#
# You only configure the main pipeline - dependencies are handled automatically.
#
# HuggingFace PyAnnote models (recommended):
#   - pyannote/speaker-diarization-3.1 (default, uses cache)
#   - pyannote/speaker-diarization-2.1 (older version)
#
# For offline use, models are cached in ~/.cache/huggingface/
# Download while online: huggingface-cli download pyannote/speaker-diarization-3.1
#
DIARIZATION_MODEL=pyannote/speaker-diarization-3.1
```

### Configuration Notes

**Loading Local Models:**
- **Whisper**: Path to a directory containing Whisper model files
- **Diarization**: See "Offline Model Setup" below
- Use absolute paths (e.g., `/Users/yourname/models/...` on macOS/Linux, `C:\Users\yourname\models\...` on Windows)

**Using HuggingFace Models:**
- **Whisper**: Install transformers library: `poetry add transformers`
- **Diarization**: Requires `HUGGINGFACE_TOKEN` for authentication
- Model format: `organization/model-name` (e.g., `openai/whisper-large-v3`, `pyannote/speaker-diarization-3.1`)
- Models are cached automatically in `~/.cache/huggingface/hub/`

**Offline Model Setup (Running Without Internet):**

PyAnnote diarization uses multiple models internally. To run completely offline:

1. **While connected to internet**, accept model conditions on HuggingFace:
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM

2. **Download all required models** (they go to `~/.cache/huggingface/`):
   ```bash
   # Install HuggingFace CLI
   pip install huggingface-hub
   
   # Login with your token
   huggingface-cli login
   
   # Download main pipeline (downloads dependencies automatically)
   huggingface-cli download pyannote/speaker-diarization-3.1
   huggingface-cli download pyannote/segmentation-3.0
   huggingface-cli download pyannote/wespeaker-voxceleb-resnet34-LM
   ```

3. **Enable offline mode** in your `.env` file:
   ```bash
   HF_HUB_OFFLINE=1
   HF_HUB_CACHE=/path/to/models  # Optional: custom cache location
   ```
   This forces HuggingFace libraries to only use cached models (no internet requests).

4. **Specify models** using either approach:
   
   **Option A: Use model names** (if cache structure is intact):
   ```bash
   WHISPER_MODEL=openai/whisper-large-v3
   DIARIZATION_MODEL=pyannote/speaker-diarization-3.1
   ```
   
   **Option B: Use full snapshot paths** (for isolated/custom cache):
   ```bash
   # Find snapshot hash first
   ls ~/.cache/huggingface/hub/models--openai--whisper-large-v3/snapshots/
   # Output example: abc123def456
   
   # Use full path including hash
   WHISPER_MODEL=/path/to/cache/models--openai--whisper-large-v3/snapshots/abc123def456/
   DIARIZATION_MODEL=/path/to/cache/models--pyannote--speaker-diarization-3.1/snapshots/xyz789/
   ```
   
   **Cache structure:** `HF_HUB_CACHE/models--org--model/snapshots/HASH/config.json`

5. **For deployment**, copy the entire `~/.cache/huggingface/` folder to your offline machine:
   ```bash
   scp -r ~/.cache/huggingface/ user@offline-server:/path/to/models/
   ```

**Note:** You only configure one model (`DIARIZATION_MODEL`), but the pipeline automatically loads its dependencies from cache.

**Custom LLM Endpoints:**
- Set `LLM_API_BASE_URL` to your server (e.g., `http://localhost:8000/v1` for local vLLM)
- Set `LLM_MODEL` to match the model name your server expects
- Works with OpenAI-compatible APIs (vLLM, Ollama, LM Studio, etc.)

**UI Overrides:**
- All settings can be temporarily overridden in Settings page
- UI overrides don't modify the `.env` file
- Leave UI fields empty to use `.env` or default values

### Running the Application

The application uses a **client-server architecture**:
- **Server**: Flask API that handles audio processing (transcription, diarization, summarization)
- **Client**: Streamlit UI for user interaction

**Start Both Components:**

**Option 1: Using Docker (Easiest)**

1. **Create `.env` file** with your API keys:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY and HUGGINGFACE_TOKEN
   ```

2. **Start the server with Docker Compose**:
   ```bash
   docker-compose up -d
   ```
   Server will be available at `http://localhost:5001`

3. **Start the Streamlit UI** (in a separate terminal):
   ```bash
   poetry run streamlit run src/ui/app_new.py
   ```
   App will be available at `http://localhost:8501`

**Option 2: Using Shell Scripts (Local Development)**

1. **Start the Flask API Server**
   ```bash
   ./start_server.sh
   ```
   Server will be available at `http://localhost:5001`

2. **Start the Streamlit UI** (in a separate terminal)
   ```bash
   ./start_client.sh
   ```
   App will be available at `http://localhost:8501`

**Option 3: Manual Start**

1. **Start the Flask API Server**
   ```bash
   poetry run python -m src.server.app
   ```

2. **Start the Streamlit UI** (in a separate terminal)
   ```bash
   poetry run streamlit run src/ui/app_new.py
   ```

The UI will automatically connect to the API server. You can customize the API endpoint in Settings if needed.

**Docker Commands:**

```bash
# Start server in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop server
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Run with custom .env file
docker-compose --env-file .env.production up -d
```

**Using the Application:**

1. **User Mode (Default)**
   - Automatically detects default microphone and all loopback devices
   - Click "Start Recording" to begin unlimited recording
   - Click "Stop Recording" when finished
   - Transcription and summarization start automatically after recording stops
   - View timestamped transcripts with speaker labels and meeting summaries

2. **Dev Mode**
   - Enable "Dev Mode" in the sidebar
   - Manually select specific microphone and loopback devices
   - Set fixed recording duration
   - Useful for testing specific device configurations

## Speech to Text Workflow

<img src="src/image/spch2txt-diagram-2025-10-22-094333.png" alt="Speech-to-Text Flow" width="60%">

### How It Works

1. **Audio Capture**
   - Captures audio from microphone and system loopback devices simultaneously
   - Uses WASAPI loopback to record system audio (Teams, Zoom, etc.)
   - Records each device to separate WAV files

2. **Transcription**
   - Loads Whisper model (cached at initiation for performance)
   - Transcribes each audio stream separately with timestamps
   - Filters out Whisper hallucinations (like false transcriptions from silence)

3. **Speaker Diarization**
   - Analyzes audio to identify different speakers
   - Assigns speaker labels to transcription segments
   - Combines diarization with transcription using timestamp overlap

4. **Meeting Summarization**
   - Generates meeting minutes using OpenAI GPT-4
   - Creates structured summaries organized by topic
   - Highlights key decisions and action items
   - Uses lazy loading to avoid unnecessary API calls

5. **Output**
   - Combines transcripts from all devices in chronological order
   - Formats output with timestamps and speaker labels
   - Saves transcript to JSON file with metadata in `src/saved_transcripts/`
   - Saves summary to text file in `src/saved_summary/`
   - Example transcript output:
     ```
     [00:05] [Microphone SPEAKER_00]: Hello everyone
     [00:08] [System Audio SPEAKER_01]: Hi, thanks for joining
     ```
   - Example summary output:
     ```
     **Project Update**
     - Beta release scheduled for next Monday
     - Data pipeline refactor completed, reducing latency by 30%
     
     **Action Items**
     - Finalize event schema by end of week
     - Demo drift dashboard during stakeholder sync
     ```

## Testing & Development

### Available Test Scripts

| Test Script | Purpose | Command |
|------------|---------|----------|
| `test_full_workflow.py` | Complete record + transcribe workflow | `poetry run python tests/test_full_workflow.py` |
| `test_audio_devices.py` | List all audio devices | `poetry run python tests/test_audio_devices.py` |
| `test_record.py` | Simple 10-second recording | `poetry run python tests/test_record.py` |
| `test_transcribe.py` | Transcribe existing WAV file | `poetry run python tests/test_transcribe.py` |
| `test_teams_audio.py` | Interactive Teams audio testing | `poetry run python tests/test_teams_audio.py` |
| `test_summarizer.py` | Summarize from a test transcript JSON | `poetry run python tests/test_summarizer.py` |


### Code Formatting
```bash
poetry run ruff check . --fix
poetry run ruff format .
```

## Whisper Model Comparison

| Model  | Size     | Speed     | Accuracy | Recommended For |
| ------ | -------- | --------- | -------- | --------------- |
| tiny   | ~39 MB   | Very Fast | Basic    | Quick tests     |
| base   | ~74 MB   | Fast      | Good     | General use     |
| small  | ~244 MB  | Medium    | Better   | Quality results |
| medium | ~769 MB  | Slow      | Great    | High accuracy   |
| large  | ~1550 MB | Very Slow | Best     | Maximum quality |

## Limitations

- ⚠️ **Windows Only** ⚠️: Uses `pyaudiowpatch` for WASAPI support
- **No Docker**: Runs directly on Windows
- **System Audio**: Captures all system audio, not isolated to specific apps
- **Permissions**: May require admin rights depending on audio device configuration

## Development Challenges

### Audio Device Detection
- Windows provides a long list of audio devices (microphones, loopback, outputs)
- Challenge: Identifying which loopback device is actually in use
- Solution: Detect all loopback devices and filter by checking for actual audio data during recording

### ASR Methodology
- Understanding different Automatic Speech Recognition approaches
- Learning Whisper's capabilities and limitations
- Balancing model size vs. accuracy vs. speed

### PyAnnote Dependencies
- Complex dependency chain with unclear documentation
- Gated models requiring Hugging Face authentication
- Solution: Implemented lazy loading to avoid conflicts

### Hallucination Filtering
- Microphone transcribes false text ("1.5%", "...", etc.) during silence
- Caused by low audio levels triggering Whisper's pattern recognition
- Solution: Filter segments based on `no_speech_prob` threshold and known hallucination patterns
- Example filtered output:
  ```
  ⚠ Filtered: '1.5%' (no_speech_prob=0.58)
  ⚠ Filtered 14 hallucination(s)
  ✓ Kept 0 valid segment(s)
  ```

### Combining Transcription and Diarization
- Challenge: Merging speaker labels from diarization with text from transcription
- Both systems produce time-based segments with different boundaries
- Solution: Match segments using timestamp interval overlap, assigning speakers based on maximum overlap duration

### Audio Corruption when adding Diarization
- Initial implementation caused corrupted/cut WAV files during recording
- Symptoms: Audio quality degraded, making transcripts unusable
- **Root cause**: `from pyannote.audio import Pipeline` at module import time loaded torchaudio and set a global audio backend that interfered with PyAudio's recording
- **Solution**: Moved all pyannote imports inside methods (`_load_pipeline()` and `diarize()`), ensuring they only load after recording completes, eliminating the conflict

## Future Improvements

### Core: spch2txt
- Improve signal processing and merging logic when multiple speakers overlap  
- Analyze impact of audio volume on transcription quality  
- Optimize performance (CPU usage, latency, I/O)  
- Add unit tests and basic security checks
- Add an evaluation system and log every evaluation to compare the results after optimizing and improving the app

### UI/UX Enhancements
- Add pause/resume button for recording to handle meeting breaks
- Implement visual indicators for recording status (recording, paused, processing)
- Add progress bars for transcription and summarization steps

### Post-Processing with LLM
- ✅ **Implemented**: Flexible endpoint selection (OpenAI API or custom endpoints via Settings)
- ✅ **Implemented**: Three-tier configuration system for easy endpoint switching
- Implement structured prompt templates for meeting minutes with customizable sections:
  - Meeting metadata (date, attendees, duration)
  - Executive summary
  - Discussion topics with timestamps
  - Decisions made
  - Action items with assigned owners
  - Follow-up items and next steps
- Add prompt engineering options for different meeting types (standup, planning, retrospective, etc.)
- Allow users to customize summary format and detail level
- Implement transcript text cleanup using LLM to:
  - Fix word repetitions and stutters
  - Correct grammar and sentence structure
  - Remove filler words ("um", "uh", etc.)
  - Maintain context while improving readability

### Packaging and Deployment
- Package the entire application and dependencies in a portable ZIP  

### Audio and Timestamp Logic
- Fix: audio devices are detected only at application startup (connecting another device after starting the app won't show in the loopback device list)
- Adjust timestamp format to show ranges (e.g., `[00:00 → 00:23] [Speaker 01] [System Audio]`)  

### Transcription Quality
- Test WhisperX instead of Whisper for improved alignment and reduced computation time  
- Experiment with larger Whisper models (from `base` to `medium`)
- Apply Whisper optimization parameters for better accuracy and speed
- Allow users to optionally specify the recording language for higher precision in single-language sessions