@echo off
REM Launcher script for Speech to Text POC on Windows
REM Run this file to start the Streamlit application

echo ========================================
echo   Speech to Text POC - Streamlit App
echo ========================================
echo.

REM Check if poetry is available
where poetry >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Poetry is not installed or not in PATH
    echo Please install Poetry first: https://python-poetry.org/docs/#installation
    pause
    exit /b 1
)

echo Starting Streamlit application...
echo.
echo Choose which version to run:
echo 1. Basic version (streamlit_app.py)
echo 2. Advanced version with real-time transcription (streamlit_advanced.py)
echo.

set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Starting basic version...
    poetry run streamlit run src/streamlit_app.py
) else if "%choice%"=="2" (
    echo.
    echo Starting advanced version...
    poetry run streamlit run src/streamlit_advanced.py
) else (
    echo Invalid choice. Starting advanced version by default...
    poetry run streamlit run src/streamlit_advanced.py
)

pause
