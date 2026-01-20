"""
Meeting summarization functionality using OpenAI API.

This module provides meeting summarization capabilities using OpenAI's GPT models.
Uses lazy loading to avoid unnecessary API initialization during module import.

Key features:
- Lazy loading of OpenAI client to avoid conflicts during initialization
- Configurable model selection (GPT-4, GPT-4o-mini, etc.)
- Structured meeting minutes generation from transcripts
- Topic-based summarization with decisions and action items

Important: OpenAI client is initialized only when first needed to prevent
unnecessary API calls and allow for flexible configuration.
"""

from typing import Optional


class MeetingSummarizer:
    """
    Handle meeting summarization using OpenAI API.

    Generates concise meeting minutes from transcripts, organizing content
    by topic and highlighting decisions and action items. Uses lazy loading
    to avoid unnecessary API initialization.
    """

    def __init__(self, api_key: str, model: str = "gpt-4", base_url: str = None):
        """
        Initialize summarizer with OpenAI API key.

        Args:
            api_key: OpenAI API authentication key
            model: OpenAI model to use (default: "gpt-4")
            base_url: Custom API base URL (optional, for OpenAI-compatible APIs)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
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
            from openai import OpenAI

            print("Loading OpenAI client...")
            # Initialize with custom base_url if provided
            if self.base_url:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                print(f"✓ OpenAI client loaded with custom base URL: {self.base_url} (model: {self.model})")
            else:
                self.client = OpenAI(api_key=self.api_key)
                print(f"✓ OpenAI client loaded (model: {self.model})")
            self._client_loaded = True
        except Exception as e:
            self.client = None
            self._client_loaded = False
            print(f"⚠ OpenAI client failed to load: {e}")

    def summarize(self, transcript_text: str) -> Optional[str]:
        """
        Generate meeting summary from transcript text.

        Creates structured meeting minutes organized by topic, including
        key decisions and action items. Uses OpenAI's chat completion API
        to generate natural, concise summaries.

        Args:
            transcript_text: Full transcript text to summarize

        Returns:
            Formatted meeting summary string, or None if summarization fails.
        """
        if not self._client_loaded:
            self._load_client()

        if not self.client:
            print("⚠ OpenAI client not available")
            return None

        if not transcript_text or transcript_text.strip() == "":
            print("⚠ Empty transcript provided")
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
            print(f"⚠ Summarization failed: {e}")
            return None


def summarize_transcript(transcript_text: str, api_key: str, model: str = "gpt-4") -> Optional[str]:
    """
    Convenience function to summarize a transcript.

    Creates a MeetingSummarizer instance and generates a summary in one call.
    Useful for simple use cases where you don't need to reuse the summarizer.

    Args:
        transcript_text: Full transcript text to summarize
        api_key: OpenAI API authentication key
        model: OpenAI model to use (default: "gpt-4")

    Returns:
        Formatted meeting summary string, or None if summarization fails.
    """
    summarizer = MeetingSummarizer(api_key=api_key, model=model)
    return summarizer.summarize(transcript_text)
