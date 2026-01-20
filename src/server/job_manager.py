"""
Filesystem-based state management for audio processing jobs.

This module manages job state by tracking files in dedicated job directories:
- Each job gets a unique directory
- State is determined by which files exist
- All intermediate assets are stored in the job directory
"""

import json
import shutil
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class JobStage(Enum):
    """Processing stages for a job."""

    NOT_STARTED = "not_started"
    TRANSCRIBING = "transcribing"
    TRANSCRIPTION_COMPLETE = "transcription_complete"
    DIARIZING = "diarizing"
    DIARIZATION_COMPLETE = "diarization_complete"
    SUMMARIZING = "summarizing"
    COMPLETE = "complete"
    FAILED = "failed"


class JobStatus(Enum):
    """Overall job status."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobManager:
    """Manages audio processing jobs using filesystem-based state."""

    def __init__(self, jobs_dir: str = "server_jobs"):
        """
        Initialize the job manager.

        Args:
            jobs_dir: Directory to store all job directories
        """
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(exist_ok=True)

        # File names for different assets
        self.FILES = {
            "metadata": "metadata.json",
            "audio": "audio.wav",
            "transcription": "transcription.json",
            "diarization": "diarization.json",
            "summary": "summary.txt",
            "error": "error.txt",
            "progress": "progress.json",
        }

    def create_job(self, original_filename: str, file_size: int, options: Dict[str, Any] = None) -> str:
        """
        Create a new job.

        Args:
            original_filename: Original name of the uploaded file
            file_size: Size of the uploaded file in bytes
            options: Processing options

        Returns:
            Job ID (UUID string)
        """
        job_id = str(uuid.uuid4())
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(exist_ok=True)

        # Create metadata
        metadata = {
            "id": job_id,
            "original_filename": original_filename,
            "file_size": file_size,
            "options": options or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "stage": JobStage.NOT_STARTED.value,
            "status": JobStatus.QUEUED.value,
        }

        self._save_metadata(job_id, metadata)
        return job_id

    def get_job_dir(self, job_id: str) -> Path:
        """Get the directory path for a job."""
        return self.jobs_dir / job_id

    def job_exists(self, job_id: str) -> bool:
        """Check if a job exists."""
        return self.get_job_dir(job_id).exists()

    def save_audio_file(self, job_id: str, audio_file_path: str) -> None:
        """
        Save the uploaded audio file to the job directory.

        Args:
            job_id: Job identifier
            audio_file_path: Path to the uploaded audio file
        """
        if not self.job_exists(job_id):
            raise ValueError(f"Job {job_id} does not exist")

        job_dir = self.get_job_dir(job_id)
        target_path = job_dir / self.FILES["audio"]

        # Copy the audio file
        shutil.copy2(audio_file_path, target_path)

        # Update metadata
        self._update_timestamp(job_id)

    def get_audio_file_path(self, job_id: str) -> Optional[Path]:
        """Get the path to the audio file for a job."""
        if not self.job_exists(job_id):
            return None

        audio_path = self.get_job_dir(job_id) / self.FILES["audio"]
        return audio_path if audio_path.exists() else None

    def save_transcription(self, job_id: str, transcription_data: Dict[str, Any]) -> None:
        """Save transcription results."""
        self._save_json_file(job_id, self.FILES["transcription"], transcription_data)
        self._update_stage(job_id, JobStage.TRANSCRIPTION_COMPLETE)

    def save_diarization(self, job_id: str, diarization_data: Dict[str, Any]) -> None:
        """Save diarization results."""
        self._save_json_file(job_id, self.FILES["diarization"], diarization_data)
        self._update_stage(job_id, JobStage.DIARIZATION_COMPLETE)

    def save_summary(self, job_id: str, summary_text: str) -> None:
        """Save summary text."""
        self._save_text_file(job_id, self.FILES["summary"], summary_text)
        self._update_stage(job_id, JobStage.COMPLETE)
        self._update_status(job_id, JobStatus.COMPLETED)

    def save_error(self, job_id: str, error_message: str) -> None:
        """Save error information and mark job as failed."""
        self._save_text_file(job_id, self.FILES["error"], error_message)
        self._update_stage(job_id, JobStage.FAILED)
        self._update_status(job_id, JobStatus.FAILED)

    def update_progress(self, job_id: str, progress: float, message: str = "") -> None:
        """Update job progress."""
        progress_data = {"progress": progress, "message": message, "updated_at": datetime.now().isoformat()}
        self._save_json_file(job_id, self.FILES["progress"], progress_data)
        self._update_timestamp(job_id)

    def update_stage(self, job_id: str, stage: JobStage) -> None:
        """Update the processing stage."""
        self._update_stage(job_id, stage)

        # Update status based on stage
        if stage == JobStage.FAILED:
            self._update_status(job_id, JobStatus.FAILED)
        elif stage == JobStage.COMPLETE:
            self._update_status(job_id, JobStatus.COMPLETED)
        elif stage != JobStage.NOT_STARTED:
            self._update_status(job_id, JobStatus.PROCESSING)

    def get_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job metadata."""
        if not self.job_exists(job_id):
            return None

        metadata_path = self.get_job_dir(job_id) / self.FILES["metadata"]
        if not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_transcription(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get transcription data."""
        return self._load_json_file(job_id, self.FILES["transcription"])

    def get_diarization(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get diarization data."""
        return self._load_json_file(job_id, self.FILES["diarization"])

    def get_summary(self, job_id: str) -> Optional[str]:
        """Get summary text."""
        return self._load_text_file(job_id, self.FILES["summary"])

    def get_error(self, job_id: str) -> Optional[str]:
        """Get error message."""
        return self._load_text_file(job_id, self.FILES["error"])

    def get_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get progress information."""
        return self._load_json_file(job_id, self.FILES["progress"])

    def get_complete_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the complete result for a job.

        Returns:
            Dictionary containing all available data for the job
        """
        if not self.job_exists(job_id):
            return None

        metadata = self.get_metadata(job_id)
        if not metadata:
            return None

        result = {
            "job_id": job_id,
            "metadata": metadata,
            "transcript": "",
            "segments": [],
            "summary": None,
            "language": None,
            "processing_metadata": {},
        }

        # Add transcription data
        transcription = self.get_transcription(job_id)
        if transcription:
            result["transcript"] = transcription.get("text", "")
            result["segments"] = transcription.get("segments", [])
            result["language"] = transcription.get("language")
            result["processing_metadata"]["transcription"] = transcription.get("metadata", {})

        # Add diarization data (merge with segments)
        diarization = self.get_diarization(job_id)
        if diarization:
            result["segments"] = diarization.get("segments_with_speakers", result["segments"])
            result["processing_metadata"]["diarization"] = diarization.get("metadata", {})

        # Add summary
        summary = self.get_summary(job_id)
        if summary:
            result["summary"] = summary

        # Add error if exists
        error = self.get_error(job_id)
        if error:
            result["error"] = error

        return result

    def list_jobs(self, status_filter: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all jobs.

        Args:
            status_filter: Filter by status (queued, processing, completed, failed)
            limit: Maximum number of jobs to return

        Returns:
            List of job metadata dictionaries
        """
        jobs = []

        # Iterate through job directories
        for job_dir in self.jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue

            job_id = job_dir.name
            metadata = self.get_metadata(job_id)

            if not metadata:
                continue

            # Update status based on current filesystem state
            current_status = self._determine_current_status(job_id)
            if current_status != metadata.get("status"):
                metadata["status"] = current_status.value
                self._save_metadata(job_id, metadata)

            # Apply status filter
            if status_filter and metadata.get("status") != status_filter:
                continue

            jobs.append(metadata)

        # Sort by created_at (newest first)
        jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return jobs[:limit]

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job and all its files.

        Args:
            job_id: Job identifier

        Returns:
            True if job was deleted, False if job didn't exist
        """
        job_dir = self.get_job_dir(job_id)
        if not job_dir.exists():
            return False

        shutil.rmtree(job_dir)
        return True

    def _determine_current_status(self, job_id: str) -> JobStatus:
        """Determine current job status based on filesystem state."""
        if not self.job_exists(job_id):
            return JobStatus.FAILED

        job_dir = self.get_job_dir(job_id)

        # Check for error
        if (job_dir / self.FILES["error"]).exists():
            return JobStatus.FAILED

        # Check for completion
        if (job_dir / self.FILES["summary"]).exists() or (
            (job_dir / self.FILES["transcription"]).exists()
            and not self._should_process_diarization(job_id)
            and not self._should_process_summary(job_id)
        ):
            return JobStatus.COMPLETED

        # Check if processing has started
        if (job_dir / self.FILES["transcription"]).exists() or (job_dir / self.FILES["progress"]).exists():
            return JobStatus.PROCESSING

        # Default to queued
        return JobStatus.QUEUED

    def _should_process_diarization(self, job_id: str) -> bool:
        """Check if diarization should be processed for this job."""
        metadata = self.get_metadata(job_id)
        if not metadata:
            return False

        options = metadata.get("options", {})
        return options.get("enable_diarization", True)

    def _should_process_summary(self, job_id: str) -> bool:
        """Check if summary should be processed for this job."""
        metadata = self.get_metadata(job_id)
        if not metadata:
            return False

        options = metadata.get("options", {})
        return options.get("enable_summarization", True)

    def _save_metadata(self, job_id: str, metadata: Dict[str, Any]) -> None:
        """Save job metadata."""
        metadata["updated_at"] = datetime.now().isoformat()
        self._save_json_file(job_id, self.FILES["metadata"], metadata)

    def _update_timestamp(self, job_id: str) -> None:
        """Update the job's timestamp."""
        metadata = self.get_metadata(job_id)
        if metadata:
            metadata["updated_at"] = datetime.now().isoformat()
            self._save_json_file(job_id, self.FILES["metadata"], metadata)

    def _update_stage(self, job_id: str, stage: JobStage) -> None:
        """Update the job's processing stage."""
        metadata = self.get_metadata(job_id)
        if metadata:
            metadata["stage"] = stage.value
            metadata["updated_at"] = datetime.now().isoformat()
            self._save_json_file(job_id, self.FILES["metadata"], metadata)

    def _update_status(self, job_id: str, status: JobStatus) -> None:
        """Update the job's status."""
        metadata = self.get_metadata(job_id)
        if metadata:
            metadata["status"] = status.value
            metadata["updated_at"] = datetime.now().isoformat()
            if status == JobStatus.COMPLETED:
                metadata["completed_at"] = datetime.now().isoformat()
            elif status == JobStatus.FAILED:
                metadata["failed_at"] = datetime.now().isoformat()
            self._save_json_file(job_id, self.FILES["metadata"], metadata)

    def _save_json_file(self, job_id: str, filename: str, data: Any) -> None:
        """Save data as JSON to a job file."""
        if not self.job_exists(job_id):
            raise ValueError(f"Job {job_id} does not exist")

        file_path = self.get_job_dir(job_id) / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_text_file(self, job_id: str, filename: str, text: str) -> None:
        """Save text to a job file."""
        if not self.job_exists(job_id):
            raise ValueError(f"Job {job_id} does not exist")

        file_path = self.get_job_dir(job_id) / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    def _load_json_file(self, job_id: str, filename: str) -> Optional[Any]:
        """Load JSON data from a job file."""
        if not self.job_exists(job_id):
            return None

        file_path = self.get_job_dir(job_id) / filename
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _load_text_file(self, job_id: str, filename: str) -> Optional[str]:
        """Load text from a job file."""
        if not self.job_exists(job_id):
            return None

        file_path = self.get_job_dir(job_id) / filename
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except IOError:
            return None
