#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Start the Flask API server
echo "🚀 Starting Flask API Server..."
echo "Server will be available at http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask server using Poetry
poetry run python -m src.server.app