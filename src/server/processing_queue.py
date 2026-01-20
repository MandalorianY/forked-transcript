"""
Queue-based audio processing system using ThreadPoolExecutor.

This module manages a queue of audio processing jobs and processes them
asynchronously using Python's built-in threading capabilities.
"""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Empty, Queue
from typing import Any, Dict, Optional

from .job_manager import JobManager, JobStage
from .processor import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingQueue:
    """Manages a queue of audio processing jobs using ThreadPoolExecutor."""

    def __init__(self, job_manager: JobManager, max_workers: int = 2, queue_check_interval: float = 1.0):
        """
        Initialize the processing queue.

        Args:
            job_manager: JobManager instance for state management
            max_workers: Maximum number of concurrent processing threads
            queue_check_interval: How often to check for new jobs (seconds)
        """
        self.job_manager = job_manager
        self.max_workers = max_workers
        self.queue_check_interval = queue_check_interval

        # Threading components
        self.job_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_jobs: Dict[str, Future] = {}
        self.is_running = False
        self.queue_thread: Optional[threading.Thread] = None

        # Processing components
        self.processor = AudioProcessor(job_manager)

        # Lock for thread safety
        self._lock = threading.Lock()

    def start(self):
        """Start the processing queue."""
        if self.is_running:
            logger.warning("Processing queue is already running")
            return

        self.is_running = True
        self.queue_thread = threading.Thread(target=self._queue_worker, daemon=True)
        self.queue_thread.start()
        logger.info(f"Processing queue started with {self.max_workers} workers")

    def stop(self):
        """Stop the processing queue."""
        if not self.is_running:
            return

        logger.info("Stopping processing queue...")
        self.is_running = False

        # Wait for queue thread to finish
        if self.queue_thread:
            self.queue_thread.join(timeout=5.0)

        # Cancel running jobs
        with self._lock:
            for job_id, future in self.running_jobs.items():
                if not future.done():
                    logger.info(f"Cancelling job {job_id}")
                    future.cancel()

        # Shutdown executor
        self.executor.shutdown(wait=True)
        logger.info("Processing queue stopped")

    def enqueue_job(self, job_id: str, priority: int = 0) -> bool:
        """
        Add a job to the processing queue.

        Args:
            job_id: Job identifier
            priority: Job priority (lower numbers = higher priority)

        Returns:
            True if job was enqueued, False if job is already running
        """
        if not self.is_running:
            logger.error("Cannot enqueue job: processing queue is not running")
            return False

        # Check if job is already running
        with self._lock:
            if job_id in self.running_jobs:
                logger.warning(f"Job {job_id} is already running")
                return False

        # Add to queue
        self.job_queue.put((priority, job_id))
        logger.info(f"Job {job_id} enqueued with priority {priority}")
        return True

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled, False if job wasn't running
        """
        with self._lock:
            if job_id not in self.running_jobs:
                return False

            future = self.running_jobs[job_id]
            if future.cancel():
                del self.running_jobs[job_id]
                self.job_manager.save_error(job_id, "Job cancelled by user")
                logger.info(f"Job {job_id} cancelled")
                return True
            else:
                logger.warning(f"Could not cancel job {job_id} (may have already started)")
                return False

    def get_queue_status(self) -> Dict[str, Any]:
        """Get status information about the processing queue."""
        with self._lock:
            running_jobs = list(self.running_jobs.keys())

        return {
            "is_running": self.is_running,
            "queue_size": self.job_queue.qsize(),
            "running_jobs": running_jobs,
            "max_workers": self.max_workers,
        }

    def _queue_worker(self):
        """Main queue worker thread that processes jobs."""
        logger.info("Queue worker thread started")

        while self.is_running:
            try:
                # Get next job from queue (with timeout)
                try:
                    priority, job_id = self.job_queue.get(timeout=self.queue_check_interval)
                except Empty:
                    continue

                # Submit job to executor (ThreadPoolExecutor handles worker management)
                logger.info(f"Starting processing for job {job_id}")
                future = self.executor.submit(self._process_job, job_id)

                with self._lock:
                    self.running_jobs[job_id] = future

                # Add callback to clean up when job completes
                future.add_done_callback(lambda f, jid=job_id: self._job_completed(jid, f))

            except Exception as e:
                logger.error(f"Error in queue worker: {e}")
                time.sleep(1.0)

        logger.info("Queue worker thread stopped")

    def _job_completed(self, job_id: str, future: Future):
        """Callback called when a job completes."""
        with self._lock:
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]

        if future.cancelled():
            logger.info(f"Job {job_id} was cancelled")
        elif future.exception():
            error = future.exception()
            logger.error(f"Job {job_id} failed with error: {error}")
            self.job_manager.save_error(job_id, str(error))
        else:
            logger.info(f"Job {job_id} completed successfully")

    def _process_job(self, job_id: str):
        """
        Process a single job.

        Args:
            job_id: Job identifier
        """
        try:
            # Update status to processing
            self.job_manager.update_stage(job_id, JobStage.NOT_STARTED)

            # Get job metadata and options
            metadata = self.job_manager.get_metadata(job_id)
            if not metadata:
                raise ValueError(f"Job {job_id} metadata not found")

            options = metadata.get("options", {})
            audio_file_path = self.job_manager.get_audio_file_path(job_id)

            if not audio_file_path or not audio_file_path.exists():
                raise ValueError(f"Audio file not found for job {job_id}")

            # Process the audio file
            self.processor.process_audio_file(job_id, str(audio_file_path), options)

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            self.job_manager.save_error(job_id, str(e))
            raise
