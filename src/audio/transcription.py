"""
Audio transcription functionality using OpenAI Whisper.

This module handles speech-to-text transcription using OpenAI's Whisper model.
Supports multiple audio input formats, hallucination filtering, and integration
with speaker diarization for multi-speaker transcription.

Key features:
- Multiple Whisper model sizes (tiny to large)
- Hallucination detection and filtering
- Multi-device transcription with speaker labels
- Timestamped segment extraction
- Integration with pyannote.audio for speaker diarization
"""

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

    def __init__(self, model_name: str = "base"):
        """
        Initialize transcriber with a Whisper model.

        Args:
            model_name: Whisper model identifier. Can be:
                - OpenAI model size: tiny, base, small, medium, large
                - HuggingFace model ID: e.g., "openai/whisper-large-v3"
                - Local model path: e.g., "/path/to/model"
        """
        self.model_name = model_name
        self.model = None

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

    def transcribe(self, audio_input, language: str | None = None, **kwargs) -> Dict:
        """
        Transcribe audio to text using Whisper.

        Accepts multiple input formats and passes them to Whisper for transcription.
        Whisper handles all audio preprocessing internally.

        Args:
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

    def _transcribe_hf(self, audio_input, language: str | None = None, **kwargs) -> Dict:
        """
        Transcribe using HuggingFace transformers pipeline.

        Converts HuggingFace output format to match OpenAI Whisper format.
        """
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

    def get_segments(self, result: Dict) -> list:
        """
        Extract timestamped segments from transcription result.

        Args:
            result: Whisper transcription result

        Returns:
            List of segments with start, end, and text
        """
        if "segments" not in result:
            return []

        return [{"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()} for seg in result["segments"]]

    def _is_valid_transcription(self, text: str, no_speech_prob: float = 0.0) -> bool:
        """
        Check if a transcription segment is valid (not a hallucination).

        Filters out common Whisper hallucinations that occur during silence or
        low audio levels. Uses no_speech_prob threshold and pattern matching.

        Args:
            text: Transcribed text to validate
            no_speech_prob: Whisper's probability that segment contains no speech (0-1)

        Returns:
            True if segment appears to be valid speech, False if likely hallucination
        """
        if no_speech_prob > 0.6:
            return False

        hallucinations = ["1.5%", "2.5%", "3.5%", "subscribe", ".", "...", "♪", "[BLANK_AUDIO]", "(blank)"]

        text_lower = text.lower().strip()

        if text_lower in hallucinations:
            return False

        if len(text_lower) <= 3 and not any(c.isalpha() for c in text_lower):
            return False

        if len(set(text_lower.replace(" ", ""))) <= 2 and len(text_lower) < 10:
            return False

        return True

    def transcribe_multiple(self, audio_files: list, device_names: list, diarizer: Optional[object] = None) -> Dict:
        """
        Transcribe multiple audio files separately and combine results chronologically.

        Transcribes each audio stream independently, filters hallucinations, optionally
        performs speaker diarization, then combines all segments in chronological order
        using timestamps. Useful for multi-device recordings (e.g., microphone + loopback).

        Args:
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
                result = self.transcribe(audio_file, verbose=False)
                text = result["text"].strip()

                # Determine speaker label
                is_loopback = "loopback" in device_name.lower()
                device_label = "System Audio" if is_loopback else "Microphone"

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
                            "speaker": device_label,
                            "text": text,
                            "language": result.get("language", "unknown"),
                            "audio_file": os.path.basename(audio_file),
                        }
                    )

                    # Extract segments with timestamps
                    if "segments" in result:
                        filtered_count = 0
                        kept_count = 0
                        device_segments = []

                        for segment in result["segments"]:
                            segment_text = segment["text"].strip()
                            no_speech_prob = segment.get("no_speech_prob", 0.0)

                            is_valid = self._is_valid_transcription(segment_text, no_speech_prob)

                            if segment_text and is_valid:
                                device_segments.append(
                                    {"start": segment["start"], "end": segment["end"], "text": segment_text}
                                )
                                kept_count += 1
                            else:
                                filtered_count += 1

                        # Assign speakers using diarization if available
                        if diarization_segments:
                            from .diarization import assign_speakers_to_segments

                            device_segments = assign_speakers_to_segments(device_segments, diarization_segments)
                            # Add device label to speaker
                            for seg in device_segments:
                                seg["speaker"] = f"{device_label} {seg['speaker']}"
                        else:
                            # No diarization, use device label only
                            for seg in device_segments:
                                seg["speaker"] = device_label

                        all_segments.extend(device_segments)

                        if filtered_count > 0:
                            print(f"  ⚠ Filtered {filtered_count} hallucination(s), kept {kept_count} segment(s)")
                        else:
                            print(f"  ✓ Kept {kept_count} segment(s)")
                else:
                    print("  ⚠ No speech detected")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback

                traceback.print_exc()

        # Sort segments by start time to get chronological order
        all_segments.sort(key=lambda x: x["start"])

        # Build combined text with timestamps
        combined_lines = []
        for segment in all_segments:
            # Format timestamp as MM:SS
            minutes = int(segment["start"] // 60)
            seconds = int(segment["start"] % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"

            combined_lines.append(f"[{timestamp}] [{segment['speaker']}]: {segment['text']}")

        combined_text = "\n\n".join(combined_lines) if combined_lines else "(No speech detected)"

        return {
            "transcripts": transcripts,
            "combined_text": combined_text,
            "segments": all_segments,
            "num_devices": len(audio_files),
        }
