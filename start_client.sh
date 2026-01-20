#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Start the Streamlit client app
echo "🎙️ Starting Streamlit Client App..."
echo "App will be available at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

# Start the Streamlit app using Poetry
poetry run streamlit run src/ui/app_new.py