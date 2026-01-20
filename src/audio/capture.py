"""
Audio capture functionality using pyaudiowpatch for Windows WASAPI support.

This module handles recording audio from multiple devices simultaneously, including
microphones and system loopback devices. It uses threading to capture from multiple
sources in parallel and saves each stream to separate WAV files.

Key features:
- Multi-device simultaneous recording
- WASAPI loopback support for system audio capture
- Audio level detection for device testing
- Thread-safe recording with error handling
"""

import math
import os
import queue
import struct
import sys
import threading
import time
import wave
from datetime import datetime
from typing import Dict, List

# Platform-specific audio library import
if sys.platform == "win32":
    import pyaudiowpatch as pyaudio
else:
    import pyaudio


class AudioCapture:
    """
    Handle audio recording from various input devices.

    Supports recording from microphones, speakers, and WASAPI loopback devices.
    Can record from multiple devices simultaneously using threading.
    """

    def __init__(self, frames_per_buffer: int = 1024):
        """
        Initialize audio capture.

        Args:
            frames_per_buffer: Buffer size for audio chunks
        """
        self.frames_per_buffer = frames_per_buffer
        self.pa = None
        self.stream = None

    def list_devices(self) -> List[Dict]:
        """
        List all available audio devices.

        Returns:
            List of device information dictionaries
        """
        pa = pyaudio.PyAudio()
        devices = []

        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            devices.append(
                {
                    "index": i,
                    "name": info["name"],
                    "hostApi": info["hostApi"],
                    "maxInputChannels": info["maxInputChannels"],
                    "maxOutputChannels": info["maxOutputChannels"],
                    "defaultSampleRate": info["defaultSampleRate"],
                    "isLoopback": "loopback" in info["name"].lower(),
                }
            )

        pa.terminate()
        return devices

    def get_default_input_device(self) -> Dict:
        """Get the default input device information."""
        pa = pyaudio.PyAudio()
        device_info = pa.get_default_input_device_info()
        pa.terminate()
        return device_info

    def get_supported_sample_rate(self, device_index: int, channels: int, preferred_rate: int = 44100) -> int:
        """
        Find a supported sample rate for the device.

        Args:
            device_index: Audio device index
            channels: Number of channels to use
            preferred_rate: Preferred sample rate (default: 44100)

        Returns:
            Supported sample rate, or None if device is invalid
        """
        pa = pyaudio.PyAudio()

        try:
            # Verify device exists
            device_info = pa.get_device_info_by_index(device_index)
            if device_info["maxInputChannels"] == 0:
                pa.terminate()
                return None  # Not an input device
        except Exception:
            pa.terminate()
            return None  # Invalid device

        # Try common sample rates in order of preference
        rates_to_try = [preferred_rate, 44100, 48000, 22050, 16000, 8000]
        # Remove duplicates while preserving order
        rates_to_try = list(dict.fromkeys(rates_to_try))

        for rate in rates_to_try:
            try:
                # Test if this rate is supported
                supported = pa.is_format_supported(
                    rate, input_device=device_index, input_channels=channels, input_format=pyaudio.paInt16
                )
                if supported:
                    pa.terminate()
                    return int(rate)
            except Exception:
                continue

        pa.terminate()
        return 44100  # Fallback

    def get_audio_level(self, device_index: int, duration: float = 0.5) -> float:
        """
        Get the current audio level for a device.

        Args:
            device_index: Audio device index
            duration: Duration to sample in seconds

        Returns:
            Audio level (RMS) as float between 0 and 1
        """
        pa = pyaudio.PyAudio()

        try:
            device_info = pa.get_device_info_by_index(device_index)
            max_channels = device_info["maxInputChannels"]
            # Ensure we have at least 1 channel, maximum 2 (stereo)
            channels = max(1, min(max_channels, 2)) if max_channels > 0 else 1
            rate = int(device_info["defaultSampleRate"])

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=self.frames_per_buffer,
                input_device_index=device_index,
            )

            num_chunks = int(rate / self.frames_per_buffer * duration)
            frames = []

            for _ in range(num_chunks):
                try:
                    data = stream.read(self.frames_per_buffer, exception_on_overflow=False)
                    frames.append(data)
                except Exception:
                    break

            stream.close()

            if frames:
                audio_data = b"".join(frames)
                # Convert bytes to int16 values using struct
                num_samples = len(audio_data) // 2
                audio_samples = struct.unpack(f"<{num_samples}h", audio_data)
                # Calculate RMS: square root of mean of squares
                sum_squares = sum((sample / 32768.0) ** 2 for sample in audio_samples)
                level = math.sqrt(sum_squares / num_samples) if num_samples > 0 else 0.0
                return float(level)

            return 0.0

        except Exception:
            return 0.0
        finally:
            pa.terminate()

    def record_multi_device(
        self,
        device_indices: List[int],
        device_names: List[str],
        channels_list: List[int],
        rates: List[int],
        duration: int,
        output_dir: str = "src/saved_audio",
    ) -> List[str]:
        """
        Record audio from multiple devices simultaneously using threading.

        Each device is recorded in a separate thread to ensure synchronization.
        Handles loopback devices specially to capture system audio. Saves each
        stream to a separate WAV file with timestamp and device name.

        Args:
            device_indices: List of device indices to record from
            device_names: List of device names for file naming
            channels_list: List of channel counts for each device
            rates: List of sample rates for each device
            duration: Recording duration in seconds
            output_dir: Directory to save recordings (default: "src/saved_audio")

        Returns:
            List of output file paths for recorded WAV files

        Raises:
            RuntimeError: If stream opening fails or recording error occurs
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = []

        for i, name in enumerate(device_names):
            clean_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name)
            clean_name = clean_name.replace(" ", "_")
            if len(clean_name) > 50:
                clean_name = clean_name[:50]
            filename = f"recording_{timestamp}_dev{i + 1}_{clean_name}.wav"
            output_files.append(os.path.join(output_dir, filename))

        pa = pyaudio.PyAudio()

        try:
            streams = []

            for i, (device_index, channels, rate) in enumerate(zip(device_indices, channels_list, rates)):
                try:
                    stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=self.frames_per_buffer,
                        input_device_index=device_index,
                    )
                    streams.append(stream)
                except Exception as e:
                    for s in streams:
                        s.close()
                    pa.terminate()
                    raise RuntimeError(f"Failed to open stream for device {i + 1}: {e}")

            num_chunks_per_device = [int(rate * duration / self.frames_per_buffer) for rate in rates]

            queues = [queue.Queue() for _ in device_indices]
            stop_event = threading.Event()
            threads = []

            is_loopback = ["loopback" in name.lower() for name in device_names]

            for i, (stream, name, is_loop, channels, num_chunks) in enumerate(
                zip(streams, device_names, is_loopback, channels_list, num_chunks_per_device)
            ):
                thread = threading.Thread(
                    target=self._record_stream_thread,
                    args=(stream, name, is_loop, channels, num_chunks, queues[i], stop_event),
                )
                thread.daemon = True
                thread.start()
                threads.append(thread)

            start_time = time.time()

            try:
                while any(t.is_alive() for t in threads):
                    elapsed = time.time() - start_time
                    if elapsed >= duration:
                        stop_event.set()
                        break
                    time.sleep(0.1)
            except KeyboardInterrupt:
                stop_event.set()

            for thread in threads:
                thread.join(timeout=2.0)

            stop_event.set()

            for i, (output_file, q) in enumerate(zip(output_files, queues)):
                frames = []
                while not q.empty():
                    try:
                        frames.append(q.get_nowait())
                    except Exception:
                        break

                if len(frames) == 0:
                    num_samples = int(rates[i] * duration * channels_list[i])
                    silence_chunk = b"\x00" * (self.frames_per_buffer * 2 * channels_list[i])
                    num_silence_chunks = int(num_samples / (self.frames_per_buffer * channels_list[i]))
                    frames = [silence_chunk] * num_silence_chunks

                with wave.open(output_file, "wb") as wf:
                    wf.setnchannels(channels_list[i])
                    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(rates[i])
                    wf.writeframes(b"".join(frames))

            for stream in streams:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

            return output_files

        except Exception as e:
            raise RuntimeError(f"Error recording: {e}")
        finally:
            pa.terminate()

    def _record_stream_thread(
        self,
        stream,
        device_name: str,
        is_loopback: bool,
        channels: int,
        num_chunks: int,
        frames_queue: queue.Queue,
        stop_event: threading.Event,
    ):
        """
        Thread function to record from a single audio stream.

        Continuously reads audio chunks from the stream and adds them to a queue.
        Handles read errors gracefully by inserting silence. Stops when the
        specified number of chunks is reached or stop event is set.

        Args:
            stream: PyAudio stream object to read from
            device_name: Name of the device (for logging)
            is_loopback: Whether device is a loopback device
            channels: Number of audio channels
            num_chunks: Maximum number of chunks to record
            frames_queue: Queue to store audio frame data
            stop_event: Threading event to signal early stop
        """
        chunk_count = 0
        silence = b"\x00" * (self.frames_per_buffer * 2 * channels)
        consecutive_errors = 0
        max_consecutive_errors = 100

        while chunk_count < num_chunks and not stop_event.is_set():
            try:
                data = stream.read(self.frames_per_buffer, exception_on_overflow=False)
                frames_queue.put(data)
                chunk_count += 1
                consecutive_errors = 0

            except Exception:
                consecutive_errors += 1
                frames_queue.put(silence)
                chunk_count += 1

                if consecutive_errors >= max_consecutive_errors:
                    break

                time.sleep(0.01)

    def record_multiple_unlimited(self, devices: List[Dict], stop_callback) -> List[str]:
        """
        Record audio from multiple devices until stop_callback returns True.

        Args:
            devices: List of device dictionaries with device info
            stop_callback: Function that returns True when recording should stop

        Returns:
            List of output file paths for recorded WAV files
        """
        if not devices:
            return []

        # Extract device information
        device_indices = []
        device_names = []
        channels_list = []
        rates = []

        for device in devices:
            device_indices.append(device["index"])
            device_names.append(device["name"])

            # Safely get number of channels with fallbacks
            max_channels = device.get("max_input_channels", 0)
            if max_channels <= 0:
                # Fallback to mono for devices with no input channels specified
                channels = 1
            elif max_channels == 1:
                channels = 1
            else:
                # Use stereo for devices that support it
                channels = 2

            # Safely get sample rate
            sample_rate = device.get("default_sample_rate", 44100)
            if sample_rate <= 0:
                sample_rate = 44100

            # Verify the sample rate is actually supported
            device_idx = device["index"]
            sample_rate = self.get_supported_sample_rate(device_idx, channels, int(sample_rate))
            if sample_rate is None:
                # Device is invalid, skip it
                print(f"Skipping invalid device: {device.get('name', 'Unknown')}")
                continue

            channels_list.append(channels)
            rates.append(int(sample_rate))

        # Create output directory
        output_dir = "src/saved_audio"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize PyAudio
        pa = pyaudio.PyAudio()
        streams = []
        threads = []
        successful_output_files = []

        try:
            # Set up streams and queues for each device
            for i, (device_idx, device_name, channels, rate) in enumerate(
                zip(device_indices, device_names, channels_list, rates)
            ):
                # Create filename
                allowed_chars = (" ", "-", "_")
                safe_name = "".join(c for c in device_name if c.isalnum() or c in allowed_chars)
                safe_name = safe_name.rstrip()
                filename = f"{timestamp}_{safe_name}_{i}.wav"
                output_path = os.path.join(output_dir, filename)

                # Create stream with error handling
                stream = None
                actual_channels = channels

                try:
                    stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        input_device_index=device_idx,
                        frames_per_buffer=self.frames_per_buffer,
                    )
                except Exception as e:
                    print(f"Failed to open stream for {device_name}: {e}")
                    # Try with mono if stereo failed
                    if channels > 1:
                        try:
                            print(f"Retrying with mono for {device_name}")
                            stream = pa.open(
                                format=pyaudio.paInt16,
                                channels=1,
                                rate=rate,
                                input=True,
                                input_device_index=device_idx,
                                frames_per_buffer=self.frames_per_buffer,
                            )
                            actual_channels = 1
                        except Exception as e2:
                            print(f"Failed mono for {device_name}: {e2}")
                            continue  # Skip this device entirely
                    else:
                        continue  # Skip this device entirely

                if stream:
                    streams.append(stream)
                    successful_output_files.append(output_path)

                    # Start recording thread
                    thread = threading.Thread(
                        target=self._record_unlimited_thread,
                        args=(stream, output_path, rate, actual_channels, None, stop_callback),
                    )
                    threads.append(thread)
                    thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

        finally:
            # Clean up streams
            for stream in streams:
                if stream:
                    stream.close()
            pa.terminate()

        return successful_output_files

    def _record_unlimited_thread(self, stream, output_path: str, rate: int, channels: int, audio_queue, stop_callback):
        """Record from a stream until stop_callback returns True."""
        frames = []

        try:
            while not stop_callback():
                try:
                    data = stream.read(self.frames_per_buffer, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    print(f"Recording error: {e}")
                    break

                time.sleep(0.01)  # Small delay to prevent excessive CPU usage

        finally:
            # Save the recorded audio
            if frames:
                with wave.open(output_path, "wb") as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(rate)
                    wf.writeframes(b"".join(frames))

    def cleanup(self):
        """Clean up audio resources."""
        if self.stream:
            self.stream.close()
        if self.pa:
            self.pa.terminate()
