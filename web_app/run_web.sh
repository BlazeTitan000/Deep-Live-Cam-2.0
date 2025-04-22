#!/bin/bash

echo "Starting Deep-Live-Cam Web Version..."

# Activate virtual environment
source ../web_venv/bin/activate

# Check for models
if [ ! -f "../models/inswapper_128_fp16.onnx" ]; then
    echo "Error: Model file inswapper_128_fp16.onnx not found in models directory!"
    echo "Please download it from: https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
    exit 1
fi

if [ ! -f "../models/GFPGANv1.4.pth" ]; then
    echo "Error: Model file GFPGANv1.4.pth not found in models directory!"
    echo "Please download it from: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    exit 1
fi

# Run the web app
echo "Starting web application..."
echo
echo "Access the web interface at: http://localhost:5000"
echo
echo "Press Ctrl+C to stop the server"
echo

python app.py --execution-provider cuda