"""
Speaker diarization functionality using pyannote.audio.

This module provides speaker diarization (identifying who spoke when) using
pyannote.audio models. Uses lazy loading to avoid conflicts with PyAudio during
audio recording. The pipeline is only loaded when first needed.

Key features:
- Lazy loading of pyannote pipeline to avoid audio backend conflicts
- Speaker identification with timestamps
- Integration with Whisper transcription segments
- Overlap-based speaker assignment to transcription segments

Important: All pyannote imports are done inside methods, not at module level,
to prevent torchaudio from interfering with PyAudio's recording functionality.
"""

import os
from typing import List, Optional, Tuple


class PyannoteDiarizer:
    """
    Handle speaker diarization using pyannote.audio.

    Identifies different speakers in audio recordings and provides timestamps
    for when each speaker is active. Uses lazy loading to avoid conflicts with
    audio recording libraries.
    """

    def __init__(self, hf_token: str, model_name: str = "pyannote/speaker-diarization-3.1"):
        """
        Initialize diarizer with Hugging Face token and model.

        Args:
            hf_token: Hugging Face authentication token
            model_name: HuggingFace model ID or local path for diarization model
                       Default: "pyannote/speaker-diarization-3.1"
        """
        self.hf_token = hf_token
        self.model_name = model_name
        self.pipeline = None
        self._pipeline_loaded = False

    def _load_pipeline(self):
        """
        Load the pyannote diarization pipeline (lazy loading).

        Imports pyannote modules only when needed to avoid conflicts with PyAudio.
        The import at module level would load torchaudio and set a global audio
        backend that interferes with PyAudio's recording.
        """
        if self._pipeline_loaded:
            return

        try:
            # Import pyannote ONLY when loading pipeline (not at module import time)
            import torch
            from pyannote.audio import Pipeline
            from pyannote.audio.core.task import Problem, Resolution, Specifications

            # Fix PyTorch 2.6+ weights_only security issue
            # Allow safe globals needed by PyAnnote models
            torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
            torch.serialization.add_safe_globals([Specifications])
            torch.serialization.add_safe_globals([Problem])
            torch.serialization.add_safe_globals([Resolution])

            print(f"Loading diarization pipeline: {self.model_name}")
            self.pipeline = Pipeline.from_pretrained(self.model_name, use_auth_token=self.hf_token)
            self._pipeline_loaded = True
            print(f"✓ Diarization pipeline loaded: {self.model_name}")
        except Exception as e:
            self.pipeline = None
            self._pipeline_loaded = False
            print(f"⚠ Diarization pipeline failed to load ({self.model_name}): {e}")

    def diarize(self, audio_path: str) -> Optional[List[Tuple[float, float, str]]]:
        """
        Perform speaker diarization on an audio file.

        Analyzes the audio to identify different speakers and when they speak.
        Returns segments with speaker labels that can be matched to transcription
        segments using timestamps.

        Args:
            audio_path: Path to WAV audio file to analyze

        Returns:
            List of (start_time, end_time, speaker_label) tuples, where speaker_label
            is typically "SPEAKER_00", "SPEAKER_01", etc. Returns None if diarization
            fails or empty list if no speakers detected.
        """
        # Lazy load the pipeline on first use
        if not self._pipeline_loaded:
            self._load_pipeline()

        if not self.pipeline:
            print("⚠ Diarization pipeline not available")
            return None

        if not os.path.exists(audio_path):
            print(f"⚠ Audio file not found: {audio_path}")
            return None

        try:
            # Import ProgressHook only when needed
            from pyannote.audio.pipelines.utils.hook import ProgressHook

            # Check if audio file is empty or too small
            file_size = os.path.getsize(audio_path)
            if file_size < 1000:
                print(f"⚠ Audio file too small for diarization: {audio_path}")
                return []

            with ProgressHook() as hook:
                diarization = self.pipeline(audio_path, hook=hook)

                segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segments.append((turn.start, turn.end, speaker))

                return segments

        except Exception as e:
            print(f"⚠ Diarization failed for {audio_path}: {e}")
            return None

    @staticmethod
    def assign_speakers_to_segments(
        whisper_segments: List[dict], diarization_segments: List[Tuple[float, float, str]]
    ) -> List[dict]:
        """
        Assign speaker labels to Whisper transcription segments using timestamp overlap.

        Matches each transcription segment to a speaker by finding the diarization segment
        with the most temporal overlap. This combines the text from Whisper with the
        speaker identification from pyannote.

        Args:
            whisper_segments: List of Whisper segments with 'start', 'end', 'text' keys
            diarization_segments: List of (start_time, end_time, speaker_label) tuples

        Returns:
            List of segments with added 'speaker' field containing the assigned speaker label.
            If no overlap is found, speaker is set to "UNKNOWN".
        """
        if not diarization_segments:
            return whisper_segments

        result_segments = []

        for segment in whisper_segments:
            seg_start = segment["start"]
            seg_end = segment["end"]

            # Find overlapping diarization segments
            overlaps = []
            for dia_start, dia_end, speaker in diarization_segments:
                # Calculate overlap
                overlap_start = max(seg_start, dia_start)
                overlap_end = min(seg_end, dia_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > 0:
                    overlaps.append((overlap_duration, speaker))

            # Assign speaker with most overlap
            if overlaps:
                overlaps.sort(reverse=True, key=lambda x: x[0])
                assigned_speaker = overlaps[0][1]
            else:
                assigned_speaker = "UNKNOWN"

            result_segment = segment.copy()
            result_segment["speaker"] = assigned_speaker
            result_segments.append(result_segment)

        return result_segments


def assign_speakers_to_segments(
    whisper_segments: List[dict], diarization_segments: List[Tuple[float, float, str]]
) -> List[dict]:
    """
    Assign speaker labels to Whisper transcription segments using timestamp overlap.

    Wrapper function for module-level import compatibility.
    See PyannoteDiarizer.assign_speakers_to_segments for details.
    """
    return PyannoteDiarizer.assign_speakers_to_segments(whisper_segments, diarization_segments)
