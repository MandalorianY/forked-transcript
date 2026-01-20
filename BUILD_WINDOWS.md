# Building Windows Recorder Client

This guide explains how to create a standalone Windows executable for the recorder client.

## Prerequisites

- Windows 10/11
- Python 3.10 or higher installed
- Git (to clone the repository)
- At least 2GB free disk space

## Build Methods Comparison

| Feature | PyInstaller | Nuitka |
|---------|-------------|--------|
| **Build Speed** | Fast (5-10 min) | Slow (10-30 min) |
| **Antivirus Detection** | Often flagged | Rarely flagged ✅ |
| **Performance** | Good | Better ✅ |
| **Startup Time** | Slower (unpacking) | Fast ✅ |
| **Build Complexity** | Simple | Requires C compiler |
| **Recommendation** | Quick testing | Production release |

**Choose Nuitka if:** You're getting antivirus false positives or need better performance  
**Choose PyInstaller if:** You want faster builds for testing

## Quick Start

### Method A: PyInstaller (Fast Build)

1. **Open Command Prompt** in the project directory
2. **Run the build script:**
   ```cmd
   build_windows.bat
   ```
3. **Wait for completion** (5-10 minutes)
4. **Find your executable** in `dist\SpchToText_Recorder\`

### Method B: Nuitka (Better AV Compatibility) ⭐

1. **Install a C compiler** (required):
   - **Option 1 - Visual Studio Build Tools** (recommended):
     - Download: https://visualstudio.microsoft.com/downloads/
     - Install "Desktop development with C++"
     - Restart your computer
   - **Option 2 - MinGW64** (simpler):
     - Nuitka will auto-download if no compiler found

2. **Run the Nuitka build script:**
   ```cmd
   build_windows_nuitka.bat
   ```

3. **Wait for completion** (10-30 minutes - compiling to C is slower)
4. **Find your executable** in `dist\SpchToText_Recorder_Nuitka\`

### Method C: Manual PyInstaller Build

1. **Install dependencies:**
   ```cmd
   pip install pyinstaller streamlit sounddevice requests python-dotenv numpy pyaudiowpatch
   ```

2. **Build the executable:**
   ```cmd
   pyinstaller recorder_client.spec
   ```

3. **Executable location:**
   ```
   dist\SpchToText_Recorder\SpchToText_Recorder.exe
   ```

## What Gets Included

The recorder client bundle includes:
- ✅ Audio recording functionality
- ✅ Streamlit UI
- ✅ API client for uploading to server
- ✅ Local file management
- ✅ Windows audio drivers (PyAudioWPatch)

**Excluded** (server-only components):
- ❌ Whisper models
- ❌ PyAnnote diarization
- ❌ PyTorch/TorchAudio
- ❌ Flask server
- ❌ Transformers library

This keeps the bundle size small (~100-200MB vs 2GB+ with models).

## Running the Executable

### First Time Setup

1. **Copy the entire folder** `dist\SpchToText_Recorder\` to your desired location
   
2. **Create a `.env` file** in the same folder as the .exe:
   ```env
   # API Server URL (REQUIRED)
   API_BASE_URL=http://your-server-ip:5001
   
   # Optional: Custom output directory
   # OUTPUT_DIR=C:\Users\YourName\Recordings
   ```

3. **Double-click** `SpchToText_Recorder.exe`

4. **Browser will open** automatically to `http://localhost:8501`

### Configuration

The executable looks for `.env` file in these locations (in order):
1. Same directory as the .exe
2. User's home directory
3. Falls back to defaults

**Default Settings:**
- API Server: `http://localhost:5001`
- Output Directory: `saved_audio\` (relative to exe)

### Connecting to Remote Server

To connect to a remote server:

```env
# .env file
API_BASE_URL=http://192.168.1.100:5001
```

Or use a domain:
```env
API_BASE_URL=https://transcription-server.company.com
```

## Troubleshooting

### Build Errors

**Problem:** "PyInstaller not found"
```cmd
pip install pyinstaller
```

**Problem:** "Module not found: pyaudiowpatch"
```cmd
pip install pyaudiowpatch
```

**Problem:** Build succeeds but exe crashes
- Check `dist\SpchToText_Recorder\SpchToText_Recorder.exe.log` for errors
- Try building with console window: Edit `recorder_client.spec`, change `console=False` to `console=True`

### Runtime Errors

**Problem:** "Cannot connect to server"
- Check `.env` file has correct `API_BASE_URL`
- Verify server is running: Open browser to `http://your-server:5001/health`
- Check Windows Firewall isn't blocking the connection

**Problem:** "No audio devices found"
- Ensure microphone is connected
- Check Windows sound settings
- Run as Administrator if needed

**Problem:** "Streamlit won't start"
- Check if port 8501 is already in use
- Close other Streamlit instances
- Restart the application

### Antivirus Warnings

Some antivirus software may flag PyInstaller executables as suspicious. **This is a false positive**.

#### Why This Happens:
- **PyInstaller signature:** The bootloader pattern is commonly used by malware
- **UPX compression:** File packing is associated with malware obfuscation
- **Behavioral heuristics:** Self-extracting executables trigger warnings

#### Solutions:

**1. Use Nuitka Instead (Recommended)** ✅
```cmd
build_windows_nuitka.bat
```
- Compiles to native C code (no PyInstaller signature)
- Much better antivirus compatibility
- See "Method B" above for full instructions

**2. Submit False Positive Reports**
- Bkav: https://www.bkav.com/support/submit-virus-sample
- Yandex: https://yandex.com/support/antivirus/false-positive.html
- VirusTotal: Upload and mark as safe

**3. Code Signing Certificate**
- Get an EV (Extended Validation) certificate (~$400/year)
- Signs the executable with your identity
- Builds reputation with antivirus vendors over time
- Best for commercial distribution

**4. Temporary Workaround**
- Add exception in Windows Security for `SpchToText_Recorder.exe`
- Right-click → Properties → "Unblock" checkbox
- Run as Administrator if needed

**Known Detections (PyInstaller builds):**
- Bkav: W32.AIDetectMalware
- Yandex: Riskware.PyInstaller
- Skyhigh: BehavesLike.Win64.Generic

These are **false positives** - the application is safe. Switching to Nuitka resolves most issues.

## Advanced Configuration

### Custom Icon

Add your icon to the spec file:
```python
# In recorder_client.spec
exe = EXE(
    ...
    icon='path/to/icon.ico',
)
```

### One-File Mode

To create a single .exe instead of folder:

```python
# In recorder_client.spec, replace COLLECT with:
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SpchToText_Recorder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
)
```

Note: Single file mode is slower to start.

### Reducing Size

To reduce executable size:

1. **Enable UPX compression** (already enabled in spec)
2. **Exclude unnecessary Streamlit features:**
   ```python
   excludes=['streamlit.runtime.caching', ...]
   ```
3. **Use virtual environment** for clean build

## Distribution

### Packaging for Distribution

1. **Compress the folder:**
   ```cmd
   cd dist
   tar -czf SpchToText_Recorder_v0.1.0.zip SpchToText_Recorder
   ```

2. **Include documentation:**
   - Copy `.env.example` to package
   - Include `README.md` with setup instructions

3. **Create installer** (optional):
   - Use Inno Setup or NSIS
   - Auto-creates desktop shortcut
   - Registers .env template

## Security Considerations

### Safe Distribution

- **Don't include `.env` with secrets** in the package
- Include `.env.example` template only
- Users create their own `.env` with their API URL

### Network Security

- Use HTTPS for production API URLs
- Consider VPN for corporate networks
- Firewall rules for port 5001

## Support

For issues:
1. Check logs in exe directory
2. Test with console mode enabled
3. Verify dependencies with `pip list`
4. Report issues with full error logs

## Building on Other Platforms

This guide is Windows-specific. For other platforms:
- **macOS:** Use `.app` bundle with `py2app`
- **Linux:** Use `.AppImage` or native package managers
