"""
Audio processing server package.

This package provides a Flask API server with Python Queue-based asynchronous
processing for audio transcription, diarization, and summarization.
"""

from .job_manager import JobManager, JobStage, JobStatus, JobFailureHandler
from .models import TranscriptResult, TranscriptSegment


def __getattr__(name):
    """Lazy import to avoid circular dependencies during module initialization."""
    if name == "app":
        from .app import app

        return app
    if name == "ProcessingQueue":
        from .processing_queue import ProcessingQueue

        return ProcessingQueue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "app",
    "JobManager",
    "JobStatus",
    "JobStage",
    "TranscriptResult",
    "TranscriptSegment",
    "JobFailureHandler",
    "ProcessingQueue",
]
