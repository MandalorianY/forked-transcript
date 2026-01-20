"""
Flask API server for audio processing.

This server provides endpoints for:
- Uploading audio files for processing
- Checking processing status
- Retrieving completed transcripts and summaries

The server uses Python Queue with ThreadPoolExecutor for async processing.
"""

import atexit
import json
import logging
import os
import tempfile
import warnings
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from .job_manager import JobManager, JobStatus
from .processing_queue import ProcessingQueue

# Suppress common third-party warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.set_audio_backend.*")
warnings.filterwarnings("ignore", message=".*invalid escape sequence.*")


# Initialize Flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max file size
CORS(app)

# Configure logging to reduce verbosity
load_dotenv()
log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(getattr(logging, log_level, logging.WARNING))

# Initialize job manager and processing queue
job_manager = JobManager("server_jobs")
processing_queue = ProcessingQueue(job_manager, max_workers=2)

# Start the processing queue
processing_queue.start()

# Ensure cleanup on shutdown


def cleanup():
    processing_queue.stop()


atexit.register(cleanup)

ALLOWED_EXTENSIONS = {"wav", "mp3", "mp4", "m4a", "flac", "aac", "ogg", "wma"}


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    queue_status = processing_queue.get_queue_status()
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "queue_running": queue_status["is_running"],
            "queue_size": queue_status["queue_size"],
            "running_jobs": len(queue_status["running_jobs"]),
        }
    )


@app.route("/upload", methods=["POST"])
def upload_audio():
    """
    Upload an audio file for processing.

    Expected form data:
    - file: Audio file to process
    - filename: Optional custom filename
    - options: Optional JSON string with processing options
        - whisper_model: Whisper model to use (optional)
            Precedence: 1) This option 2) WHISPER_MODEL env var 3) "base" default
            Examples: "tiny", "base", "small", "medium", "large",
                     "openai/whisper-large-v3", "/path/to/model"
        - enable_diarization: Boolean to enable speaker diarization (optional)
        - enable_summary: Boolean to enable meeting summarization (optional)

    Returns:
    - job_id: Unique identifier for tracking the processing job
    - status: Initial status (queued)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        allowed_types = ", ".join(ALLOWED_EXTENSIONS)
        return jsonify({"error": f"File type not allowed. Allowed types: {allowed_types}"}), 400

    # Validate filename is not empty after sanitization
    original_filename = secure_filename(file.filename)
    if not original_filename:
        return jsonify({"error": "Invalid filename"}), 400

    # Get processing options
    options = {}
    if "options" in request.form:
        try:
            options = json.loads(request.form["options"])
        except json.JSONDecodeError:
            pass

    # Save file temporarily
    original_filename = secure_filename(file.filename)
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{original_filename.rsplit('.', 1)[1].lower()}"
    ) as tmp_file:
        file.save(tmp_file.name)
        temp_file_path = tmp_file.name

    try:
        # Get file size and validate
        file_size = os.path.getsize(temp_file_path)

        # Validate file size (additional check beyond Flask's MAX_CONTENT_LENGTH)
        if file_size == 0:
            os.unlink(temp_file_path)
            return jsonify({"error": "Empty file not allowed"}), 400

        # Reasonable minimum size for audio files (1KB)
        if file_size < 1024:
            os.unlink(temp_file_path)
            return jsonify({"error": "File too small to be a valid audio file"}), 400

        # Create job
        job_id = job_manager.create_job(original_filename=original_filename, file_size=file_size, options=options)

        # Save audio file to job directory
        job_manager.save_audio_file(job_id, temp_file_path)

        # Enqueue for processing
        processing_queue.enqueue_job(job_id)

        return jsonify(
            {
                "job_id": job_id,
                "status": JobStatus.QUEUED.value,
                "message": "File uploaded successfully and queued for processing",
            }
        ), 201

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.route("/status/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """
    Get the status of a processing job.

    Returns job metadata including:
    - status: Current processing status
    - progress: Processing progress (if available)
    - created_at: Job creation timestamp
    - updated_at: Last update timestamp
    - error: Error message (if failed)
    """
    metadata = job_manager.get_metadata(job_id)
    if not metadata:
        return jsonify({"error": "Job not found"}), 404

    # Add progress information if available
    progress_info = job_manager.get_progress(job_id)
    if progress_info:
        metadata["progress"] = progress_info.get("progress", 0)
        metadata["progress_message"] = progress_info.get("message", "")

    return jsonify(metadata)


@app.route("/jobs", methods=["GET"])
def list_jobs():
    """
    List all processing jobs.

    Query parameters:
    - status: Filter by status (queued, processing, completed, failed)
    - limit: Limit number of results (default: 100)
    - offset: Offset for pagination (default: 0)

    Returns list of job metadata sorted by creation time (newest first).
    """
    status_filter = request.args.get("status")
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))

    jobs = job_manager.list_jobs(status_filter=status_filter, limit=limit + offset)

    # Apply pagination
    total_jobs = len(jobs)
    jobs = jobs[offset : offset + limit]

    return jsonify({"jobs": jobs, "total": total_jobs, "limit": limit, "offset": offset})


@app.route("/result/<job_id>", methods=["GET"])
def get_job_result(job_id: str):
    """
    Get the result of a completed processing job.

    Returns:
    - transcript: Full transcript text
    - segments: Timestamped segments with speaker labels
    - summary: Meeting summary (if generated)
    - metadata: Job metadata
    """
    metadata = job_manager.get_metadata(job_id)
    if not metadata:
        return jsonify({"error": "Job not found"}), 404

    if metadata.get("status") != JobStatus.COMPLETED.value:
        return jsonify({"error": "Job not completed yet"}), 400

    # Get complete result
    result = job_manager.get_complete_result(job_id)
    if not result:
        return jsonify({"error": "Result not found"}), 404

    return jsonify(result)


@app.route("/delete/<job_id>", methods=["DELETE"])
def delete_job(job_id: str):
    """
    Delete a job and its associated files.

    This will:
    - Remove the job directory and all files
    - Cancel the job if it's currently running
    """
    if not job_manager.job_exists(job_id):
        return jsonify({"error": "Job not found"}), 404

    # Try to cancel the job if it's running
    processing_queue.cancel_job(job_id)

    # Delete the job
    if job_manager.delete_job(job_id):
        return jsonify({"message": "Job deleted successfully"})
    else:
        return jsonify({"error": "Failed to delete job"}), 500


@app.route("/queue/status", methods=["GET"])
def get_queue_status():
    """Get detailed queue status information."""
    return jsonify(processing_queue.get_queue_status())


if __name__ == "__main__":
    try:
        app.run(debug=True, host="0.0.0.0", port=5001)
    finally:
        processing_queue.stop()
