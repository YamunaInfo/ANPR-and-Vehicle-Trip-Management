#!/bin/bash
# Start PaddleOCR microservice on port 5001

echo "========================================="
echo "GateSense ANPR - PaddleOCR Service Launcher"
echo "========================================="
echo ""

# Check if Python 3.11+ is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python: $PYTHON_VERSION"

# Check and install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q fastapi uvicorn paddlepaddle paddlex paddleocr opencv-python numpy

# Check if port 5001 is already in use
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "ERROR: Port 5001 is already in use"
    echo "Stop the existing service and try again."
    exit 1
fi

echo ""
echo "========================================="
echo "Starting OCR Service on http://0.0.0.0:5001"
echo "Health Check: http://localhost:5001/healthz"
echo "OCR API: POST http://localhost:5001/api/ocr"
echo "========================================="
echo ""

# Run the service
cd "$(dirname "$0")"
uvicorn ocr_service:app --host 0.0.0.0 --port 5001 --log-level info
