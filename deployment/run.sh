#!/bin/bash

# Water Segmentation Flask App Runner
# This script sets up and runs the Flask application

echo "🚀 Starting Water Segmentation Flask App..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create a virtual environment first:"
    echo "python3 -m venv .venv"
    echo "source .venv/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import flask, torch, torchvision, tifffile, matplotlib, numpy, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies! Installing from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies!"
        exit 1
    fi
fi

# Check if model file exists
echo "🔍 Checking for trained model..."
if [ -f "pretrained_segmentation.pth" ]; then
    echo "✅ Found trained model: pretrained_segmentation.pth"
elif [ -f "Pretrained_segmentation.pth" ]; then
    echo "✅ Found trained model: Pretrained_segmentation.pth"
else
    echo "⚠️  No trained model found. App will use base architecture."
    echo "Expected file: pretrained_segmentation.pth"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads results templates

# Check if templates exist
if [ ! -f "templates/index.html" ]; then
    echo "⚠️  Warning: templates/index.html not found!"
fi

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

# Run the Flask application
echo "🌐 Starting Flask server..."
echo "Server will be available at:"
echo "  - Local: http://127.0.0.1:5000"
echo "  - Network: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
