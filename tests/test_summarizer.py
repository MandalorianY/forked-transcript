import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

"""
poetry run python tests/test_summarizer.py
"""

load_dotenv()
# Get OpenAI API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

transcript = json.loads(Path("src/saved_transcripts/test_transcript_20251022.json").read_text())

prompt = f"""
You are an assistant creating concise meeting minutes.
Summarize this transcript clearly by topic, including decisions and next actions.

Transcript:
{transcript["combined_text"]}
"""


class MeetingSummarizer:
    def __init__(self, token, model="gpt-4"):
        self.client = OpenAI()
        self.client.api_key = token
        self.model = model

    def summarize(self, text):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a meeting summarizer."},
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content


# Save the summary to a text file
summary = MeetingSummarizer(api_key).summarize(prompt)
Path("src/saved_summary/test_summary_20251022.txt").write_text(summary)
