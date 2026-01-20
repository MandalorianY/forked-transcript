"""
Data models for the audio processing server.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Re-export enums from job_manager for backward compatibility
from .job_manager import JobStage, JobStatus


@dataclass
class TranscriptSegment:
    """A single segment of transcribed audio with timing and speaker info."""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None


@dataclass
class TranscriptResult:
    """Complete result of audio processing."""

    job_id: str
    transcript: str
    segments: List[TranscriptSegment]
    summary: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    processing_time: Optional[float] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class JobMetadata:
    """Metadata for a processing job."""

    id: str
    original_filename: str
    file_path: str
    status: JobStatus
    stage: JobStage
    created_at: datetime
    updated_at: datetime
    file_size: int
    options: Dict[str, Any]
    error: Optional[str] = None
    progress: Optional[float] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None


# Legacy aliases for backward compatibility
TranscriptStatus = JobStatus
