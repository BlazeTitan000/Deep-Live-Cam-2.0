import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"Python path: {sys.path}")
print(f"Project root: {project_root}")

# Import required modules
import modules.globals
from face_swapper import get_face_swapper, swap_face, process_frame, process_image

# Import web-specific modules
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from PIL import Image
import argparse
import onnxruntime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deep-live-cam-secret'
socketio = SocketIO(app)

# Global variables
source_image = None
target_image = None
face_swapper = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution-provider', 
                       help='execution provider',
                       dest='execution_provider',
                       default=['cpu'],
                       choices=['cpu', 'cuda', 'dml', 'rocm', 'coreml'],
                       nargs='+')
    args = parser.parse_args()
    
    # Set execution providers
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    return args

def decode_execution_providers(execution_providers: list) -> list:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]

def encode_execution_providers(execution_providers: list) -> list:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]

# Initialize face swapper at startup
def initialize_face_swapper():
    global face_swapper
    print("Initializing face swapper...")
    try:
        face_swapper = get_face_swapper()
        print("Successfully initialized face swapper")
    except Exception as e:
        print(f"Error initializing face swapper: {str(e)}")
        face_swapper = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_source', methods=['POST'])
def upload_source():
    global source_image
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read and process the image
    img = Image.open(file.stream)
    source_image = np.array(img)
    return jsonify({'success': True})

@app.route('/upload_target', methods=['POST'])
def upload_target():
    global target_image
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read and process the image
    img = Image.open(file.stream)
    target_image = np.array(img)
    return jsonify({'success': True})

@app.route('/swap_faces', methods=['POST'])
def swap_faces():
    global source_image, target_image, face_swapper
    
    if source_image is None or target_image is None:
        return jsonify({'error': 'Both source and target images are required'}), 400
    
    if face_swapper is None:
        return jsonify({'error': 'Face swapper not initialized'}), 500
    
    # Process the target image with face swapping
    result_image = process_image(source_image, target_image)
    
    # Convert the result to base64 with high quality
    result_pil = Image.fromarray(result_image)
    buffered = io.BytesIO()
    result_pil.save(buffered, format="JPEG", quality=95)
    result_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return jsonify({
        'success': True,
        'result': result_base64
    })

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('process_frame')
def handle_process_frame(data):
    global source_image, face_swapper
    
    if source_image is None or face_swapper is None:
        return
    
    # Decode the frame
    frame_data = base64.b64decode(data['frame'])
    frame_np = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    
    # Process the frame
    result_frame = process_frame(source_image, frame)
    
    # Convert the processed frame back to base64
    _, buffer = cv2.imencode('.jpg', result_frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Send the processed frame back to the client
    emit('processed_frame', {'frame': frame_base64})

if __name__ == '__main__':
    args = parse_args()
    initialize_face_swapper()
    socketio.run(app, host="0.0.0.0", debug=False) 