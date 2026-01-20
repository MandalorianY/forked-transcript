"""
Audio processing module for transcription, diarization, and summarization.

This module contains the logic for processing audio files through the
various stages: transcription, diarization, and summarization.
"""

import logging
import os
import time
import warnings
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from ..audio import AudioTranscriber, MeetingSummarizer, PyannoteDiarizer
from .job_manager import JobManager, JobStage

# Suppress warnings from third-party libraries
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pyannote")

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class AudioProcessor:
    """Handles the actual audio processing tasks."""

    def __init__(self, job_manager: JobManager):
        """
        Initialize the audio processor.

        Args:
            job_manager: JobManager instance for state management
        """
        self.job_manager = job_manager

        # Initialize components (lazy loading)
        self.transcriber: Optional[AudioTranscriber] = None
        self.transcriber_model_name: Optional[str] = None
        self.diarizer: Optional[PyannoteDiarizer] = None
        self.summarizer: Optional[MeetingSummarizer] = None

    def process_audio_file(self, job_id: str, audio_file_path: str, options: Dict[str, Any]) -> None:
        """
        Process an audio file through all stages.

        Args:
            job_id: Job identifier
            audio_file_path: Path to the audio file
            options: Processing options
        """
        start_time = time.time()

        try:
            logger.info(f"Starting processing for job {job_id}")

            # Stage 1: Transcription
            self._process_transcription(job_id, audio_file_path, options)

            # Stage 2: Diarization (if enabled)
            if options.get("enable_diarization", True):
                self._process_diarization(job_id, audio_file_path, options)

            # Stage 3: Summarization (if enabled)
            if options.get("enable_summarization", True):
                self._process_summarization(job_id, options)

            # Mark as complete if no summary was requested
            if not options.get("enable_summarization", True):
                self.job_manager.update_stage(job_id, JobStage.COMPLETE)

            processing_time = time.time() - start_time
            logger.info(f"Job {job_id} completed in {processing_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Processing failed for job {job_id}: {e}")
            self.job_manager.save_error(job_id, str(e))
            raise

    def _process_transcription(self, job_id: str, audio_file_path: str, options: Dict[str, Any]) -> None:
        """Process transcription stage."""
        logger.info(f"Starting transcription for job {job_id}")

        # Update stage
        self.job_manager.update_stage(job_id, JobStage.TRANSCRIBING)
        self.job_manager.update_progress(job_id, 10.0, "Starting transcription...")

        requested_model = options.get("whisper_model", os.getenv("WHISPER_MODEL", "base"))

        # Initialize or refresh transcriber when model changes
        if self.transcriber is None or self.transcriber_model_name != requested_model:
            self.transcriber = AudioTranscriber(model_name=requested_model)
            self.transcriber.load_model()
            self.transcriber_model_name = requested_model
            logger.info(f"Loaded Whisper model: {requested_model}")

        self.job_manager.update_progress(job_id, 20.0, "Transcribing audio...")

        # Perform transcription
        result = self.transcriber.transcribe(audio_file_path)

        # Extract data
        transcript_text = result.get("text", "")
        segments = result.get("segments", [])
        language = result.get("language", "unknown")

        # Save transcription data
        transcription_data = {
            "text": transcript_text,
            "segments": segments,
            "language": language,
            "metadata": {
                "whisper_model": requested_model,
                "duration": result.get("duration", 0),
                "processing_stage": "transcription",
            },
        }

        self.job_manager.save_transcription(job_id, transcription_data)
        self.job_manager.update_progress(job_id, 40.0, "Transcription completed")

        logger.info(f"Transcription completed for job {job_id}")

    def _process_diarization(self, job_id: str, audio_file_path: str, options: Dict[str, Any]) -> None:
        """Process diarization stage."""
        logger.info(f"Starting diarization for job {job_id}")

        # Update stage
        self.job_manager.update_stage(job_id, JobStage.DIARIZING)
        self.job_manager.update_progress(job_id, 50.0, "Starting diarization...")

        # Check for HuggingFace token (prioritize options, fallback to env)
        hf_token = options.get("huggingface_token") or os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            logger.warning(f"No HuggingFace token found, skipping diarization for job {job_id}")
            self.job_manager.update_progress(job_id, 70.0, "Diarization skipped (no token)")
            return

        # Initialize diarizer if needed
        if self.diarizer is None:
            diarization_model = options.get("diarization_model", "pyannote/speaker-diarization-3.1")
            self.diarizer = PyannoteDiarizer(hf_token=hf_token, model_name=diarization_model)
            logger.info(f"Initialized speaker diarization with model: {diarization_model}")

        self.job_manager.update_progress(job_id, 60.0, "Performing speaker diarization...")

        try:
            # Perform diarization
            diarization_result = self.diarizer.diarize(audio_file_path)

            if diarization_result:
                # Get original transcription segments
                transcription_data = self.job_manager.get_transcription(job_id)
                if transcription_data:
                    segments = transcription_data.get("segments", [])

                    # Assign speakers to segments
                    segments_with_speakers = self.diarizer.assign_speakers_to_segments(segments, diarization_result)

                    # Save diarization data
                    diarization_data = {
                        "segments_with_speakers": segments_with_speakers,
                        "speaker_segments": diarization_result,
                        "metadata": {
                            "num_speakers": len(set(seg[2] for seg in diarization_result)),
                            "num_segments": len(diarization_result),
                            "processing_stage": "diarization",
                        },
                    }

                    self.job_manager.save_diarization(job_id, diarization_data)
                    self.job_manager.update_progress(job_id, 70.0, "Diarization completed")

                    logger.info(f"Diarization completed for job {job_id}: {len(diarization_result)} speaker segments")
                else:
                    logger.error(f"No transcription data found for diarization in job {job_id}")
            else:
                logger.warning(f"Diarization returned no results for job {job_id}")
                self.job_manager.update_progress(job_id, 70.0, "Diarization completed (no speakers found)")

        except Exception as e:
            logger.error(f"Diarization failed for job {job_id}: {e}")
            self.job_manager.update_progress(job_id, 70.0, f"Diarization failed: {str(e)}")
            # Don't fail the entire job, just skip diarization

    def _process_summarization(self, job_id: str, options: Dict[str, Any]) -> None:
        """Process summarization stage."""
        logger.info(f"Starting summarization for job {job_id}")

        # Update stage
        self.job_manager.update_stage(job_id, JobStage.SUMMARIZING)
        self.job_manager.update_progress(job_id, 80.0, "Starting summarization...")

        # Get transcript text
        transcription_data = self.job_manager.get_transcription(job_id)
        if not transcription_data:
            logger.error(f"No transcription data found for summarization in job {job_id}")
            return

        transcript_text = transcription_data.get("text", "")
        if not transcript_text.strip():
            logger.warning(f"Empty transcript, skipping summarization for job {job_id}")
            self.job_manager.update_progress(job_id, 100.0, "Summarization skipped (empty transcript)")
            self.job_manager.update_stage(job_id, JobStage.COMPLETE)
            return

        # Check for OpenAI API key - prioritize options, then environment
        openai_key = options.get("llm_api_key") or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            logger.warning(f"No OpenAI API key found, skipping summarization for job {job_id}")
            self.job_manager.update_progress(job_id, 100.0, "Summarization skipped (no API key)")
            self.job_manager.update_stage(job_id, JobStage.COMPLETE)
            return

        # Get LLM configuration from options (with fallbacks to defaults)
        llm_model = options.get("llm_model") or options.get("summary_model") or "gpt-4o-mini"
        llm_base_url = options.get("llm_api_base_url")  # Optional custom endpoint

        # Initialize summarizer if needed or if settings changed
        summarizer_key = f"{llm_model}_{llm_base_url or 'default'}"
        if self.summarizer is None or getattr(self, "_summarizer_key", None) != summarizer_key:
            if llm_base_url:
                logger.info(f"Initializing summarizer with custom endpoint: {llm_base_url}, model: {llm_model}")
                self.summarizer = MeetingSummarizer(api_key=openai_key, model=llm_model, base_url=llm_base_url)
            else:
                logger.info(f"Initializing summarizer with model: {llm_model}")
                self.summarizer = MeetingSummarizer(api_key=openai_key, model=llm_model)
            self._summarizer_key = summarizer_key

        self.job_manager.update_progress(job_id, 90.0, "Generating summary...")

        try:
            # Generate summary
            summary = self.summarizer.summarize(transcript_text)

            if summary:
                self.job_manager.save_summary(job_id, summary)
                self.job_manager.update_progress(job_id, 100.0, "Processing complete")
                logger.info(f"Summarization completed for job {job_id}")
            else:
                logger.warning(f"Summarization returned empty result for job {job_id}")
                self.job_manager.update_progress(job_id, 100.0, "Summarization completed (empty result)")
                self.job_manager.update_stage(job_id, JobStage.COMPLETE)

        except Exception as e:
            logger.error(f"Summarization failed for job {job_id}: {e}")
            self.job_manager.update_progress(job_id, 100.0, f"Summarization failed: {str(e)}")
            # Don't fail the entire job, just skip summarization
            self.job_manager.update_stage(job_id, JobStage.COMPLETE)
