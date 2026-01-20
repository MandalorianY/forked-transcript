@echo off
REM ============================================================================
REM Nuitka Build Script for spch2txt Recorder Client on Windows
REM ============================================================================
REM This creates a native compiled executable using Nuitka instead of PyInstaller.
REM Nuitka compiles Python to C code, resulting in better antivirus compatibility.

echo ========================================
echo Building spch2txt Recorder Client (Nuitka)
echo ========================================
echo.

REM Set Python command to use Python 3.12
set PYTHON_CMD=py -3.12

REM Verify Python 3.12 is available
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python 3.12 not found!
    echo.
    echo Please install Python 3.12 from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Found Python 3.12:
%PYTHON_CMD% --version

:python_found
echo.

REM Check Python version
echo Checking Python version...
%PYTHON_CMD% -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo.
    echo ERROR: Python 3.10 or higher is required!
    echo Your version:
    %PYTHON_CMD% --version
    pause
    exit /b 1
)
echo Python version OK
echo.

REM Install/upgrade pip
echo Ensuring pip is up to date...
%PYTHON_CMD% -m pip install --upgrade pip --quiet
echo.

REM Check if Nuitka is installed
echo Checking for Nuitka...
%PYTHON_CMD% -m nuitka --version >nul 2>&1
if errorlevel 1 (
    echo Nuitka not found. Installing...
    %PYTHON_CMD% -m pip install nuitka ordered-set --quiet
    echo Nuitka installed!
) else (
    echo Nuitka found:
    %PYTHON_CMD% -m nuitka --version
)
echo.

REM Using MinGW64 compiler
echo Using MinGW64 compiler (Nuitka will download if needed)...
echo.

REM Install runtime dependencies
echo ========================================
echo Installing Runtime Dependencies
echo ========================================
echo.
echo Installing Streamlit...
%PYTHON_CMD% -m pip install streamlit --quiet
echo Installing SoundDevice...
%PYTHON_CMD% -m pip install sounddevice --quiet
echo Installing PyAudioWPatch (Windows audio)...
%PYTHON_CMD% -m pip install pyaudiowpatch --quiet
echo Installing Requests (HTTP client)...
%PYTHON_CMD% -m pip install requests --quiet
echo Installing Python-dotenv (config)...
%PYTHON_CMD% -m pip install python-dotenv --quiet
echo.
echo All dependencies installed!
echo.

REM Clean previous build
if exist "dist\launcher.dist" (
    echo Cleaning previous build...
    rmdir /s /q "launcher.dist"
)
if exist "dist\launcher.build" (
    rmdir /s /q "launcher.build"
)

REM Build with Nuitka
echo.
echo ========================================
echo Building with Nuitka (using MinGW64)...
echo This will take 10-30 minutes - please be patient
echo ========================================
echo.

%PYTHON_CMD% -m nuitka ^
    --mingw64 ^
    --standalone ^
    --windows-console-mode=force ^
    --disable-plugin=anti-bloat ^
    --noinclude-pytest-mode=nofollow ^
    --noinclude-setuptools-mode=nofollow ^
    --noinclude-IPython-mode=nofollow ^
    --include-package=streamlit ^
    --include-package=sounddevice ^
    --include-package=pyaudiowpatch ^
    --include-package=requests ^
    --include-package=dotenv ^
    --include-package-data=streamlit ^
    --include-data-dir=src=src ^
    --include-data-file=.env.example=.env.example ^
    --output-dir=dist ^
    --output-filename=SpchToText_Recorder.exe ^
    --company-name="spch2txt" ^
    --product-name="Speech to Text Recorder" ^
    --file-version=0.1.0 ^
    --product-version=0.1.0 ^
    --file-description="Audio recording client for speech-to-text transcription" ^
    --assume-yes-for-downloads ^
    --show-progress ^
    launcher.py

set BUILD_ERROR=%errorlevel%

if %BUILD_ERROR% neq 0 (
    echo.
    echo ========================================
    echo ERROR: Nuitka build failed! (Error code: %BUILD_ERROR%)
    echo ========================================
    echo.
    echo Common issues:
    echo 1. Missing C compiler - see instructions above
    echo 2. Out of memory - close other applications
    echo 3. Antivirus blocking - temporarily disable
    echo 4. Missing dependencies - check output above
    echo.
    echo Full command that failed:
    echo %PYTHON_CMD% -m nuitka --show-progress launcher.py
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b %BUILD_ERROR%
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================

pause
