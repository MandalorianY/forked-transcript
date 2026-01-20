# Speech-to-Text Application - Refactored Architecture

This project has been refactored to use a client-server architecture that separates audio recording from processing.

## New Architecture Overview

### Client Side (Streamlit App)
- **Audio Recording**: Records audio locally using the device's microphone and system audio
- **File Upload**: Uploads recorded audio files to the API server for processing
- **Job Management**: View and manage transcription jobs
- **Results Display**: View detailed transcripts, speaker segments, and summaries

### Server Side (Flask API)
- **File Processing**: Handles audio transcription, diarization, and summarization
- **Queue Management**: Uses Python's built-in Queue with ThreadPoolExecutor for async processing
- **State Management**: Filesystem-based state tracking with dedicated job directories
- **No External Dependencies**: No Redis, Celery, or other external services required

## Quick Start

### 1. Install System Dependencies

#### macOS (Required)
```bash
# Install PortAudio (required for pyaudio)
brew install portaudio

# Install ffmpeg (required for Whisper audio processing)
brew install ffmpeg
```

#### Windows
No additional system dependencies required - pyaudiowpatch handles Windows audio.

#### Linux
```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev ffmpeg

# CentOS/RHEL/Fedora
sudo yum install portaudio-devel ffmpeg
```

### 2. Install Python Dependencies
```bash
# Install/update dependencies
pip install -e .
```

**Note**: This project uses platform-dynamic dependencies:
- **Windows**: Uses `pyaudiowpatch` for WASAPI loopback support
- **macOS/Linux**: Uses standard `pyaudio` (requires PortAudio system library)
- **All platforms**: Requires `ffmpeg` for Whisper audio processing

### 3. Set Up Environment Variables (Optional)
Create a `.env` file in the project root:
```bash
# For speaker diarization (optional)
HUGGINGFACE_TOKEN=your_huggingface_token_here

# For meeting summarization (optional)  
OPENAI_API_KEY=your_openai_api_key_here

# API server URL (default: http://localhost:5001)
API_BASE_URL=http://localhost:5001
```

### 4. Start the API Server
```bash
# Option 1: Use the startup script
./start_server.sh

# Option 2: Run directly
python -m src.server.app_new
```

The API server will start on `http://localhost:5001`

### 5. Start the Client App
In a new terminal:
```bash
# Option 1: Use the startup script
./start_client.sh

# Option 2: Run directly
streamlit run src/ui/app_new.py
```

The Streamlit app will open in your browser at `http://localhost:8501`

## How It Works

### Job Processing Pipeline

1. **Upload**: Audio file is uploaded to the API server
2. **Queued**: Job is added to the processing queue
3. **Transcription**: Audio is transcribed using OpenAI Whisper
4. **Diarization** (optional): Speakers are identified using pyannote.audio
5. **Summarization** (optional): Meeting summary is generated using OpenAI GPT
6. **Complete**: Results are available for download

### Filesystem State Management

Each job gets a dedicated directory structure:
```
server_jobs/
├── {job_id}/
│   ├── metadata.json      # Job metadata and status
│   ├── audio.wav         # Original audio file
│   ├── transcription.json # Whisper transcription results
│   ├── diarization.json  # Speaker diarization results (if enabled)
│   ├── summary.txt       # Meeting summary (if enabled)
│   ├── progress.json     # Processing progress updates
│   └── error.txt         # Error information (if failed)
```

Job status is determined by which files exist:
- **Queued**: Only `metadata.json` and `audio.wav`
- **Processing**: `progress.json` exists
- **Completed**: `transcription.json` and optionally `summary.txt` exist
- **Failed**: `error.txt` exists

## Platform Compatibility

### Supported Platforms
- **macOS**: ✅ Full support (requires PortAudio and ffmpeg)
- **Windows**: ✅ Full support with WASAPI loopback audio
- **Linux**: ✅ Full support (requires PortAudio and ffmpeg)

### Audio Dependencies
The project uses dynamic platform-specific dependencies defined in `pyproject.toml`:

```toml
# Windows: Advanced audio support with loopback
"pyaudiowpatch>=0.2.12.7,<0.3.0.0; sys_platform == 'win32'"

# macOS/Linux: Standard audio support  
"pyaudio>=0.2.11; sys_platform != 'win32'"
```

### Audio Capabilities by Platform

| Feature | Windows | macOS | Linux |
|---------|---------|-------|-------|
| Microphone Recording | ✅ | ✅ | ✅ |
| System Audio (Loopback) | ✅ Full WASAPI | ⚠️ Limited¹ | ⚠️ Limited¹ |
| Multi-device Recording | ✅ | ✅ | ✅ |
| Real-time Processing | ✅ | ✅ | ✅ |

¹ System audio capture on macOS/Linux may require additional setup or third-party tools

## API Endpoints

### Core Endpoints
- `GET /health` - Health check and queue status
- `POST /upload` - Upload audio file for processing
- `GET /status/{job_id}` - Get job status and progress
- `GET /jobs` - List all jobs with filtering
- `GET /result/{job_id}` - Get complete job results
- `GET /result/{job_id}/download` - Download results as JSON
- `DELETE /delete/{job_id}` - Delete job and all files

### Queue Management
- `GET /queue/status` - Get detailed queue information

## Streamlit App Features

### 📹 Recording Page
- Record audio from microphone and system audio simultaneously
- Real-time recording with duration display
- Configurable processing options (Whisper model, diarization, summarization)
- Local save option or upload for processing

### 📋 Jobs Page  
- View all transcription jobs
- Filter by status (queued, processing, completed, failed)
- Auto-refresh for real-time updates
- Download and delete individual jobs

### 📄 Transcript Page
- Detailed view of completed transcriptions
- Tabbed interface: Full transcript, timestamped segments, summary, metadata
- Speaker labels (if diarization was enabled)
- Copy to clipboard functionality

## Processing Options

### Whisper Models
- `tiny`: Fastest, least accurate
- `base`: Good balance of speed and accuracy
- `small`: Better accuracy, slower
- `medium`: High accuracy, much slower
- `large`: Best accuracy, slowest

### Speaker Diarization
- Requires HuggingFace token
- Uses pyannote.audio for speaker identification
- Assigns speaker labels to transcript segments

### Meeting Summarization  
- Requires OpenAI API key
- Uses GPT-4o-mini by default
- Generates structured meeting minutes

## Advantages of New Architecture

1. **No External Dependencies**: Pure Python solution using built-in Queue and ThreadPoolExecutor
2. **Filesystem State**: Easy to debug and understand job states
3. **Self-Contained**: No Redis, Celery, or database setup required  
4. **Scalable**: Easy to adjust worker count and queue size
5. **Robust**: Jobs persist across server restarts
6. **Transparent**: All intermediate files are accessible

## Migration from Old Version

The old version processed everything in the Streamlit app. The new version:
- Moves all heavy processing to the API server
- Keeps only recording in the client
- Provides better job management and monitoring
- Enables multiple concurrent processing jobs
- Separates concerns for better maintainability

## Troubleshooting

### Installation Issues

#### macOS: "No such file or directory: 'ffmpeg'"
```bash
# Install ffmpeg using Homebrew
brew install ffmpeg
```

#### macOS: "Invalid number of channels" or pyaudio build fails
```bash
# Install PortAudio system library
brew install portaudio

# Then reinstall the project
pip install -e .
```

#### Windows: Audio recording not working
- Ensure `pyaudiowpatch` is installed (should be automatic)
- Check Windows audio permissions
- Try running as administrator if needed

### Server Issues

#### Server Won't Start
- Check that port 5001 is available (macOS AirPlay uses port 5000)
- Ensure all dependencies are installed: `pip install -e .`
- Check the terminal for error messages
- Verify ffmpeg is installed: `ffmpeg -version`

#### Processing Fails with ffmpeg Error
```bash
# Install ffmpeg (see Installation Issues above)
ffmpeg -version  # Verify installation
```

### Client Issues

#### Client Can't Connect to Server
- Verify the API server is running on `http://localhost:5001`
- Check the `API_BASE_URL` in the Streamlit sidebar
- Ensure no firewall is blocking the connection

#### Recording Fails
- Check system audio permissions
- On macOS: System Preferences → Security & Privacy → Microphone
- Verify audio devices are detected in the Recording page

### Processing Issues

#### Jobs Fail During Processing
- Check the API server logs for detailed error messages
- Verify environment variables are set correctly (HF token, OpenAI key)
- Ensure audio file format is supported (.wav, .mp3, .m4a, etc.)
- Check disk space in the `server_jobs` directory
- Check microphone permissions
- Verify audio devices are detected in the Streamlit app
- Try different audio file formats if upload fails

## Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_workflow.py
```

### Project Structure
```
src/
├── audio/           # Audio processing modules (transcription, diarization, etc.)
├── client/          # API client for Streamlit app
├── server/          # Flask API server
│   ├── app_new.py   # Main Flask application
│   ├── job_manager.py    # Filesystem state management
│   ├── processing_queue.py # Queue and worker management
│   └── processor.py      # Audio processing logic
└── ui/
    └── app_new.py   # Refactored Streamlit client
```