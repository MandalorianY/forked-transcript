from src.server.job_manager import JobManager, JobFailureHandler
import os
from typing import Dict, Optional
import numpy as np
import whisper


class AudioTranscriber:
    """
    Handle audio transcription using OpenAI Whisper.

    Provides methods for transcribing audio files or arrays, extracting
    timestamped segments, filtering hallucinations, and combining multiple
    audio streams with speaker diarization.
    """

    def __init__(self, model_name: str = "base", job_manager: JobManager = None):
        """
        Initialize transcriber with a Whisper model.

        Args:
            model_name: Whisper model identifier. Can be:
                - OpenAI model size: tiny, base, small, medium, large
                - HuggingFace model ID: e.g., "openai/whisper-large-v3"
                - Local model path: e.g., "/path/to/model"
            job_manager: The job manager used to track job status (must be passed).
        """
        self.model_name = model_name
        self.model = None
        self.job_manager = job_manager

    def load_model(self):
        """
        Load the Whisper model.

        Supports multiple model sources:
        - OpenAI models (tiny, base, small, medium, large)
        - HuggingFace models (model IDs like "openai/whisper-large-v3")
        - Local models (file paths)
        """
        if self.model is None:
            try:
                # Check if it's a HuggingFace model ID or local path
                if "/" in self.model_name or os.path.exists(self.model_name):
                    # Use transformers library for HuggingFace models
                    try:
                        from transformers import pipeline

                        print(f"Loading Whisper model from HuggingFace: {self.model_name}")
                        self.model = pipeline("automatic-speech-recognition", model=self.model_name)
                        self._is_hf_model = True
                    except ImportError:
                        print("⚠ transformers library not installed. Install with: pip install transformers")
                        print(f"Falling back to OpenAI Whisper for: {self.model_name}")
                        self.model = whisper.load_model("base")
                        self._is_hf_model = False
                else:
                    # Standard OpenAI Whisper model
                    print(f"Loading OpenAI Whisper model: {self.model_name}")
                    self.model = whisper.load_model(self.model_name)
                    self._is_hf_model = False
            except Exception as e:
                print(f"⚠ Failed to load model {self.model_name}: {e}")
                print("Falling back to base model")
                self.model = whisper.load_model("base")
                self._is_hf_model = False

    def transcribe(self, job_id: str, audio_input, language: str | None = None, **kwargs) -> Dict:
        """
        Transcribe audio to text using Whisper.

        Args:
            job_id: The job ID to update on error.
            audio_input: Audio source - can be file path (str), numpy array, or bytes
            language: Language code for transcription (default: auto-detect)
            **kwargs: Additional arguments passed to whisper.transcribe()

        Returns:
            Dictionary containing transcription results with keys:
            - 'text': Full transcribed text
            - 'segments': List of timestamped segments
            - 'language': Detected or specified language

        Raises:
            ValueError: If audio_input type is not supported
        """
        self.load_model()

        try:
            # Handle HuggingFace models differently
            if getattr(self, "_is_hf_model", False):
                return self._transcribe_hf(audio_input, language, **kwargs)

            # Handle different input types for OpenAI Whisper
            if isinstance(audio_input, str):
                # Pass file path directly to Whisper - it handles all preprocessing
                result = self.model.transcribe(audio_input, language=language, **kwargs)
            elif isinstance(audio_input, np.ndarray):
                # Ensure audio data is float32
                audio = audio_input.astype(np.float32)
                result = self.model.transcribe(audio, language=language, **kwargs)
            elif isinstance(audio_input, bytes):
                # Convert bytes to numpy array
                audio = np.frombuffer(audio_input, dtype=np.int16).astype(np.float32) / 32768.0
                result = self.model.transcribe(audio, language=language, **kwargs)
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

            return result
        except Exception as e:
            JobFailureHandler.handle_failure(job_id, f"Transcription failed: {str(e)}", self.job_manager)
            raise  # Reraise the exception to stop further processing

    def _transcribe_hf(self, audio_input, language: str | None = None, **kwargs) -> Dict:
        """
        Transcribe using HuggingFace transformers pipeline.

        Converts HuggingFace output format to match OpenAI Whisper format.
        """
        try:
            # HuggingFace pipeline expects file path or numpy array
            if isinstance(audio_input, str):
                hf_result = self.model(audio_input, return_timestamps=True)
            elif isinstance(audio_input, np.ndarray):
                audio = audio_input.astype(np.float32)
                hf_result = self.model(audio, return_timestamps=True)
            elif isinstance(audio_input, bytes):
                audio = np.frombuffer(audio_input, dtype=np.int16).astype(np.float32) / 32768.0
                hf_result = self.model(audio, return_timestamps=True)
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

            # Convert HuggingFace format to OpenAI Whisper format
            result = {"text": hf_result.get("text", ""), "language": language or "unknown", "segments": []}

            # Convert chunks to segments if available
            if "chunks" in hf_result:
                for chunk in hf_result["chunks"]:
                    result["segments"].append(
                        {
                            "start": chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0.0,
                            "end": chunk["timestamp"][1] if chunk["timestamp"][1] is not None else 0.0,
                            "text": chunk["text"],
                            "no_speech_prob": 0.0,  # HuggingFace doesn't provide this
                        }
                    )

            return result
        except Exception as e:
            raise ValueError(f"Error in HuggingFace transcription: {e}")

    def transcribe_multiple(
        self, job_id: str, audio_files: list, device_names: list, diarizer: Optional[object] = None
    ) -> Dict:
        """
        Transcribe multiple audio files separately and combine results chronologically.

        Args:
            job_id: The job ID to update on error.
            audio_files: List of audio file paths to transcribe
            device_names: List of device names for labeling (e.g., "Microphone", "Loopback")
            diarizer: Optional PyannoteDiarizer instance for speaker identification

        Returns:
            Dictionary containing:
            - 'transcripts': List of per-device transcription results
            - 'combined_text': Chronologically ordered text with timestamps and speakers
            - 'segments': List of all segments sorted by start time
            - 'num_devices': Number of devices transcribed
        """
        transcripts = []
        all_segments = []

        for i, (audio_file, device_name) in enumerate(zip(audio_files, device_names)):
            print(f"\nTranscribing device {i + 1}: {device_name}")

            try:
                result = self.transcribe(job_id, audio_file, verbose=False)
                text = result["text"].strip()

                # Perform diarization if diarizer is provided
                diarization_segments = None
                if diarizer is not None and text:
                    print("  Running diarization...")
                    try:
                        diarization_segments = diarizer.diarize(audio_file)
                        if diarization_segments:
                            print(f"  ✓ Found {len(diarization_segments)} speaker segment(s)")
                    except Exception as e:
                        print(f"  ⚠ Diarization error: {e}")

                if text:
                    transcripts.append(
                        {
                            "device": device_name,
                            "speaker": "Microphone" if "loopback" not in device_name.lower() else "System Audio",
                            "text": text,
                            "language": result.get("language", "unknown"),
                            "audio_file": os.path.basename(audio_file),
                        }
                    )

                    # Extract segments with timestamps
                    if "segments" in result:
                        all_segments.extend(result["segments"])

                else:
                    print("  ⚠ No speech detected")

            except Exception as e:
                JobFailureHandler.handle_failure(
                    job_id, f"Error during transcription for {audio_file}: {str(e)}", self.job_manager
                )

        # Sort segments by start time
        all_segments.sort(key=lambda x: x["start"])

        combined_text = "\n\n".join(
            [f"[{int(seg['start'] // 60):02d}:{int(seg['start'] % 60):02d}] {seg['text']}" for seg in all_segments]
        )

        return {
            "transcripts": transcripts,
            "combined_text": combined_text,
            "segments": all_segments,
            "num_devices": len(audio_files),
        }
