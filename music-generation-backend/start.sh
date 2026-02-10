#!/bin/bash

# Music Generation Backend - Startup Script

echo "=================================="
echo "Music Generation Backend"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created. Please review and update if needed."
fi

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/uploads
mkdir -p data/processed
mkdir -p data/generated

# Start the server
echo ""
echo "=================================="
echo "Starting server..."
echo "=================================="
echo ""
echo "API will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
