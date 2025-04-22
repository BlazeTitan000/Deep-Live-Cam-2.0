@echo off
echo Starting Deep-Live-Cam Web Version...

REM Activate virtual environment
call web_venv\Scripts\activate

REM Check for models
if not exist "..\models\inswapper_128_fp16.onnx" (
    echo Error: Model file inswapper_128_fp16.onnx not found in models directory!
    echo Please download it from: https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx
    pause
    exit /b 1
)

if not exist "..\models\GFPGANv1.4.pth" (
    echo Error: Model file GFPGANv1.4.pth not found in models directory!
    echo Please download it from: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
    pause
    exit /b 1
)

REM Run the web app
echo Starting web application...
echo.
echo Access the web interface at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py --execution-provider cuda

pause 