@echo off
REM ============================================================================
REM Build script for spch2txt Recorder Client on Windows
REM ============================================================================
REM This creates a standalone executable that only includes the recording
REM client without server components or ML models.

echo ========================================
echo Building spch2txt Recorder Client
echo ========================================
echo.

REM Try to find Python using multiple methods
set PYTHON_CMD=

REM Method 1: Try py launcher (recommended on Windows)
py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    echo Found Python via launcher: 
    py --version
    goto :python_found
)

REM Method 2: Try python command
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    echo Found Python: 
    python --version
    goto :python_found
)

REM Method 3: Try python3 command
python3 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3
    echo Found Python3: 
    python3 --version
    goto :python_found
)

REM Method 4: Check common installation paths
if exist "C:\Python313\python.exe" (
    set PYTHON_CMD=C:\Python313\python.exe
    echo Found Python at: C:\Python313\python.exe
    goto :python_found
)
if exist "C:\Python312\python.exe" (
    set PYTHON_CMD=C:\Python312\python.exe
    echo Found Python at: C:\Python312\python.exe
    goto :python_found
)
if exist "C:\Python311\python.exe" (
    set PYTHON_CMD=C:\Python311\python.exe
    echo Found Python at: C:\Python311\python.exe
    goto :python_found
)
if exist "C:\Python310\python.exe" (
    set PYTHON_CMD=C:\Python310\python.exe
    echo Found Python at: C:\Python310\python.exe
    goto :python_found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python313\python.exe
    echo Found Python at: %LOCALAPPDATA%\Programs\Python\Python313\python.exe
    goto :python_found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
    echo Found Python at: %LOCALAPPDATA%\Programs\Python\Python312\python.exe
    goto :python_found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
    echo Found Python at: %LOCALAPPDATA%\Programs\Python\Python311\python.exe
    goto :python_found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe
    echo Found Python at: %LOCALAPPDATA%\Programs\Python\Python310\python.exe
    goto :python_found
)

REM Python not found
echo.
echo ERROR: Python not found!
echo.
echo Please do one of the following:
echo 1. Add Python to PATH:
echo    - Search "Environment Variables" in Windows
echo    - Edit PATH and add Python installation directory
echo    - Common locations:
echo      * C:\Python310
echo      * C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310
echo.
echo 2. Or reinstall Python from https://www.python.org/downloads/
echo    Make sure to check "Add Python to PATH" during installation
echo.
echo 3. Or manually set PYTHON_CMD in this script to your Python path
echo.
pause
exit /b 1

:python_found
echo.

REM Add Python Scripts directory to PATH (where pip.exe lives)
for /f "delims=" %%i in ('%PYTHON_CMD% -c "import sys; import os; print(os.path.join(sys.prefix, 'Scripts'))"') do set SCRIPTS_DIR=%%i
if exist "%SCRIPTS_DIR%" (
    echo Adding Scripts directory to PATH: %SCRIPTS_DIR%
    set PATH=%SCRIPTS_DIR%;%PATH%
) else (
    echo Warning: Scripts directory not found at %SCRIPTS_DIR%
)

REM Verify pip is working
echo Checking pip...
%PYTHON_CMD% -m pip --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: pip not found or not working properly.
    echo Attempting to install/upgrade pip...
    %PYTHON_CMD% -m ensurepip --default-pip
    %PYTHON_CMD% -m pip install --upgrade pip
)

REM Check if PyInstaller is installed
echo Checking PyInstaller...
%PYTHON_CMD% -m pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    %PYTHON_CMD% -m pip install pyinstaller
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install PyInstaller!
        echo Try running this command manually:
        echo %PYTHON_CMD% -m pip install pyinstaller
        pause
        exit /b 1
    )
    echo PyInstaller installed successfully!
) else (
    echo PyInstaller is already installed.
)

REM Verify PyInstaller can be imported
echo Verifying PyInstaller installation...
%PYTHON_CMD% -c "import PyInstaller; print('PyInstaller version:', PyInstaller.__version__)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo PyInstaller import failed. Reinstalling...
    %PYTHON_CMD% -m pip uninstall pyinstaller -y
    %PYTHON_CMD% -m pip install pyinstaller
)

REM Clean previous builds
if exist "build" (
    echo Cleaning previous build...
    rmdir /s /q build
)
if exist "dist" (
    echo Cleaning previous dist...
    rmdir /s /q dist
)

REM Install minimal dependencies (client only)
echo.
echo ========================================
echo Installing client dependencies...
echo This may take 5-10 minutes on first run
echo ========================================
echo.
echo Upgrading pip...
%PYTHON_CMD% -m pip install --upgrade pip --quiet
echo.
echo Installing Streamlit (UI framework)...
%PYTHON_CMD% -m pip install streamlit --quiet
echo Installing SoundDevice (audio capture)...
%PYTHON_CMD% -m pip install sounddevice --quiet
echo Installing PyAudioWPatch (Windows audio)...
%PYTHON_CMD% -m pip install pyaudiowpatch --quiet
echo Installing Requests (HTTP client)...
%PYTHON_CMD% -m pip install requests --quiet
echo Installing Python-dotenv (config)...
%PYTHON_CMD% -m pip install python-dotenv --quiet
echo Installing NumPy (arrays)...
%PYTHON_CMD% -m pip install numpy --quiet
echo.
echo All dependencies installed!
echo.

REM Build with PyInstaller
echo.
echo ========================================
echo Building executable...
echo This will take 5-15 minutes - please wait
echo ========================================
echo.
%PYTHON_CMD% -m PyInstaller recorder_client.spec

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    echo.
    echo Try running manually to see detailed error:
    echo %PYTHON_CMD% -m PyInstaller recorder_client.spec
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Executable location: dist\SpchToText_Recorder\SpchToText_Recorder.exe
echo.
echo To run the application:
echo 1. Copy the entire 'dist\SpchToText_Recorder' folder to your desired location
echo 2. Create a .env file in that folder with your API server URL:
echo    API_BASE_URL=http://your-server:5001
echo 3. Run SpchToText_Recorder.exe
echo.

pause
