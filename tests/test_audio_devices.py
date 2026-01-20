import sys

# Platform-specific audio library import
if sys.platform == "win32":
    import pyaudiowpatch as pyaudio
else:
    import pyaudio

"""
List all the audio devices detected in the Windows machine.
Should detect WASAPI loopback devices.

How to run:
    poetry run python tests/test_audio_devices.py
"""


def list_devices():
    """List all audio devices."""
    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"{i}: {info['name']}  (hostApi={info['hostApi']})")
    pa.terminate()


if __name__ == "__main__":
    list_devices()
