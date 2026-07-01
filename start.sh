#!/bin/bash
# Start the Pillow Deformation Simulator
# Dependencies: Python 3.9+, pip packages (see backend/requirements.txt)

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  枕头形变模拟器 - Pillow Deformation Simulator"
echo "========================================"
echo ""

# Check Python
python3 --version 2>/dev/null || { echo "ERROR: python3 not found"; exit 1; }

# Check dependencies
echo "Checking dependencies..."
python3 -c "import fastapi, numpy, scipy, trimesh" 2>/dev/null || {
    echo "Installing dependencies..."
    python3 -m pip install fastapi uvicorn python-multipart numpy scipy trimesh rtree
}

echo ""
echo "Starting server on http://localhost:8000"
echo "Open your browser and navigate to http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

cd "$DIR/backend"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
