"""
Client module for communicating with the audio processing API server.

This module provides a simple interface for the Streamlit app to:
- Upload audio files for processing
- Check processing status
- Retrieve completed results
- List all jobs
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.exceptions import ConnectionError, RequestException


class APIClient:
    """Client for communicating with the audio processing API server."""

    def __init__(self, base_url: str = "http://localhost:5001"):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.timeout = 30  # Default timeout

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the API server is healthy.

        Returns:
            Dictionary containing health status information

        Raises:
            ConnectionError: If unable to connect to the server
        """
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            raise ConnectionError(f"Unable to connect to API server: {e}")

    def upload_audio_file(
        self, file_path: str, options: Optional[Dict[str, Any]] = None, timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Upload an audio file for processing.

        Args:
            file_path: Path to the audio file to upload
            options: Processing options dictionary with optional keys:
                - whisper_model: Whisper model to use (str, optional)
                    Precedence: 1) This option 2) WHISPER_MODEL env 3) "base" default
                    Examples: "tiny", "base", "small", "medium", "large",
                             "openai/whisper-large-v3", "/path/to/model"
                - enable_diarization: Enable speaker diarization (bool, optional)
                - enable_summary: Enable meeting summarization (bool, optional)
            timeout: Request timeout in seconds

        Returns:
            Dictionary containing job_id and initial status

        Raises:
            FileNotFoundError: If the file doesn't exist
            RequestException: If the upload fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Prepare data for upload
        data = {}
        if options:
            data["options"] = json.dumps(options)

        try:
            # Use context manager to ensure file is properly closed
            with open(file_path, "rb") as audio_file:
                files = {"file": (file_path.name, audio_file)}
                response = self.session.post(f"{self.base_url}/upload", files=files, data=data, timeout=timeout)
                response.raise_for_status()
                return response.json()
        except RequestException as e:
            raise RequestException(f"Upload failed: {e}")
        finally:
            # Close the file
            files["file"][1].close()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a processing job.

        Args:
            job_id: Unique identifier for the job

        Returns:
            Dictionary containing job status and metadata

        Raises:
            RequestException: If the request fails
        """
        try:
            response = self.session.get(f"{self.base_url}/status/{job_id}")
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            raise RequestException(f"Failed to get job status: {e}")

    def list_jobs(self, status_filter: Optional[str] = None, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """
        List all processing jobs.

        Args:
            status_filter: Filter by status (queued, processing, completed, failed)
            limit: Maximum number of jobs to return
            offset: Offset for pagination

        Returns:
            Dictionary containing list of jobs and pagination info

        Raises:
            RequestException: If the request fails
        """
        params = {"limit": limit, "offset": offset}
        if status_filter:
            params["status"] = status_filter

        try:
            response = self.session.get(f"{self.base_url}/jobs", params=params)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            raise RequestException(f"Failed to list jobs: {e}")

    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """
        Get the result of a completed processing job.

        Args:
            job_id: Unique identifier for the job

        Returns:
            Dictionary containing transcript, segments, summary, and metadata

        Raises:
            RequestException: If the request fails
        """
        try:
            response = self.session.get(f"{self.base_url}/result/{job_id}")
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            raise RequestException(f"Failed to get job result: {e}")

    def delete_job(self, job_id: str) -> Dict[str, Any]:
        """
        Delete a job and its associated files.

        Args:
            job_id: Unique identifier for the job

        Returns:
            Dictionary containing success message

        Raises:
            RequestException: If the deletion fails
        """
        try:
            response = self.session.delete(f"{self.base_url}/delete/{job_id}")
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            raise RequestException(f"Failed to delete job: {e}")

    def wait_for_completion(self, job_id: str, poll_interval: int = 5, timeout: int = 3600) -> Dict[str, Any]:
        """
        Wait for a job to complete and return the result.

        Args:
            job_id: Unique identifier for the job
            poll_interval: Time to wait between status checks (seconds)
            timeout: Maximum time to wait (seconds)

        Returns:
            Dictionary containing the final result

        Raises:
            TimeoutError: If the job doesn't complete within the timeout
            RequestException: If any API call fails
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status_info = self.get_job_status(job_id)
            status = status_info.get("status")

            if status == "completed":
                return self.get_job_result(job_id)
            elif status == "failed":
                error = status_info.get("error", "Unknown error")
                raise RequestException(f"Job failed: {error}")

            time.sleep(poll_interval)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")


# Convenience function for quick uploads
def upload_and_process(
    file_path: str,
    options: Optional[Dict[str, Any]] = None,
    api_url: str = "http://localhost:5001",
    wait_for_result: bool = True,
    poll_interval: int = 5,
    timeout: int = 3600,
) -> Dict[str, Any]:
    """
    Upload a file and optionally wait for processing to complete.

    Args:
        file_path: Path to the audio file
        options: Processing options
        api_url: API server URL
        wait_for_result: Whether to wait for completion
        poll_interval: Polling interval in seconds
        timeout: Maximum wait time in seconds

    Returns:
        Dictionary containing either upload info or final result
    """
    client = APIClient(api_url)

    # Upload the file
    upload_result = client.upload_audio_file(file_path, options)
    job_id = upload_result["job_id"]

    if wait_for_result:
        # Wait for completion and return the result
        return client.wait_for_completion(job_id, poll_interval, timeout)
    else:
        # Return just the upload result
        return upload_result
