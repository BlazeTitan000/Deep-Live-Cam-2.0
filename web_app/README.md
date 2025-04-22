# Deep-Live-Cam Web

This is the web version of Deep-Live-Cam, a real-time face swapping application. It provides the same functionality as the desktop version but through a web interface.

## Features

- Real-time face swapping using webcam
- Source and target image upload
- Multiple face processing
- Face enhancement
- Mouth masking
- NSFW filtering
- FPS and audio control

## Requirements

- Python 3.9 or higher
- Webcam
- Modern web browser

## Installation

1. Navigate to the web_app directory:
```bash
cd web_app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure you have the required models in the `models` directory:
   - `inswapper_128_fp16.onnx`
   - `GFPGANv1.4.pth`

2. Start the Flask server:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Upload a source image (the face you want to swap)
2. Upload a target image (optional)
3. Adjust the settings as needed
4. Click "Start Camera" to begin face swapping
5. Use the switches to enable/disable various features

## Notes

- The web version uses WebSocket for real-time video streaming
- All processing is done on the server side
- The UI is responsive and works on both desktop and mobile devices
- Make sure your webcam is properly connected and accessible

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Check that your webcam is working and accessible
3. Ensure the models are in the correct directory
4. Check the console for any error messages

## License

This project is licensed under the same license as the original Deep-Live-Cam project. 