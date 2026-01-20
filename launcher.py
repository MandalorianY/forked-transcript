"""
Launcher script for spch2txt Recorder Client (Nuitka build).

This script launches the Streamlit app with proper configuration for Nuitka standalone builds.
"""

import os
import sys
from pathlib import Path


def main():
    """Launch the Streamlit application."""

    # Determine application path
    if getattr(sys, "frozen", False):
        # Running as Nuitka compiled executable
        # Nuitka standalone places everything in the same directory as the executable
        application_path = Path(sys.executable).parent
    else:
        # Running as script (development mode)
        application_path = Path(__file__).parent

    # Path to the Streamlit app file
    # In Nuitka standalone, src package is compiled alongside the executable
    app_file = application_path / "src" / "ui" / "app_new.py"

    # Verify the app file exists
    if not app_file.exists():
        print(f"ERROR: Cannot find Streamlit app at: {app_file}")
        print(f"Application path: {application_path}")
        print("Directory contents:")
        for item in application_path.iterdir():
            print(f"  - {item.name}")
        input("\nPress Enter to exit...")
        sys.exit(1)

    # Set working directory
    os.chdir(application_path)

    # Add application path to Python path for imports
    sys.path.insert(0, str(application_path))

    # Import and run Streamlit
    from streamlit.web import cli as stcli

    # Configure Streamlit command line arguments
    sys.argv = [
        "streamlit",
        "run",
        str(app_file),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.port=8501",
        "--server.address=localhost",
        "--global.developmentMode=false",
    ]

    # Launch Streamlit
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
