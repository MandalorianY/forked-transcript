from typing import Optional
from src.server.job_manager import JobManager, JobFailureHandler


class MeetingSummarizer:
    """
    Handle meeting summarization using OpenAI API.

    Generates concise meeting minutes from transcripts, organizing content
    by topic and highlighting decisions and action items. Uses lazy loading
    to avoid unnecessary API initialization.
    """

    def __init__(
        self, api_key: str, job_id: str, model: str = "gpt-4", base_url: str = None, job_manager: JobManager = None
    ):
        """
        Initialize summarizer with OpenAI API key.

        Args:
            api_key: OpenAI API authentication key
            job_id: The job ID to update on failure
            model: OpenAI model to use (default: "gpt-4")
            base_url: Custom API base URL (optional, for OpenAI-compatible APIs)
            job_manager: The job manager used to track job status (must be passed)
        """
        self.api_key = api_key
        self.job_id = job_id
        self.model = model
        self.base_url = base_url
        self.job_manager = job_manager
        self.client = None
        self._client_loaded = False

    def _load_client(self):
        """
        Lazy load the OpenAI client.

        Imports OpenAI client only when needed to avoid conflicts during
        module import. The import at module level would initialize the client
        immediately, which may not be desired.
        """
        if self._client_loaded:
            return

        try:
            # Import OpenAI ONLY when loading client (not at module import time)
            import openai

            print("Loading OpenAI client...")
            # Initialize with custom base_url if provided
            if self.base_url:
                self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
                print(f"✓ OpenAI client loaded with custom base URL: {self.base_url} (model: {self.model})")
            else:
                self.client = openai.OpenAI(api_key=self.api_key)
                print(f"✓ OpenAI client loaded (model: {self.model})")
            self._client_loaded = True
        except Exception as e:
            self.client = None
            self._client_loaded = False
            JobFailureHandler.handle_failure(self.job_id, f"Failed to load OpenAI client: {e}", self.job_manager)

    def summarize(self, transcript_text: str) -> Optional[str]:
        """
        Generate meeting summary from transcript text.

        Creates structured meeting minutes organized by topic, including
        key decisions and next actions. Uses OpenAI's chat completion API
        to generate natural, concise summaries.

        Args:
            transcript_text: Full transcript text to summarize

        Returns:
            Formatted meeting summary string, or None if summarization fails.
        """
        if not self._client_loaded:
            self._load_client()

        if not self.client:
            error_message = "OpenAI client not available"
            JobFailureHandler.handle_failure(self.job_id, error_message, self.job_manager)
            return None

        if not transcript_text or transcript_text.strip() == "":
            error_message = "Empty transcript provided"
            JobFailureHandler.handle_failure(self.job_id, error_message, self.job_manager)
            return None

        try:
            print(f"Generating summary using {self.model}...")

            # Construct prompt for meeting summarization
            prompt = f"""You are an assistant creating concise meeting minutes.
Summarize this transcript clearly by topic, including decisions and next actions.

Transcript:
{transcript_text}"""

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a meeting summarizer."},
                    {"role": "user", "content": prompt},
                ],
            )

            summary = response.choices[0].message.content
            print("✓ Summary generated successfully")
            return summary

        except Exception as e:
            error_message = f"Summarization failed: {e}"
            JobFailureHandler.handle_failure(self.job_id, error_message, self.job_manager)
            return None


def summarize_transcript(transcript_text: str, api_key: str, job_id: str, model: str = "gpt-4") -> Optional[str]:
    """
    Convenience function to summarize a transcript.

    Creates a MeetingSummarizer instance and generates a summary in one call.
    Useful for simple use cases where you don't need to reuse the summarizer.

    Args:
        transcript_text: Full transcript text to summarize
        api_key: OpenAI API authentication key
        job_id: The job ID to track summarization status
        model: OpenAI model to use (default: "gpt-4")

    Returns:
        Formatted meeting summary string, or None if summarization fails.
    """
    job_manager = JobManager("server_jobs")
    summarizer = MeetingSummarizer(api_key=api_key, job_id=job_id, model=model, job_manager=job_manager)
    return summarizer.summarize(transcript_text)
