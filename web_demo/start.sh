#!/bin/bash
# Startup script for the web demo

echo "Starting Real-time 3D Motion Generation Demo..."
echo "Activating motion_gen environment..."

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate motion_gen

# Start Flask server
cd "$(dirname "$0")"
python app.py

