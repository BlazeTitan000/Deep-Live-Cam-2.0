import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from PIL import Image
import modules.globals
from modules.processors.frame.core import get_frame_processors_modules
import argparse
import onnxruntime

# Initialize globals without UI dependencies
modules.globals.keep_fps = True
modules.globals.keep_audio = True
modules.globals.many_faces = False
modules.globals.mouth_mask = False
modules.globals.nsfw_filter = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deep-live-cam-secret'
socketio = SocketIO(app)

# Global variables
source_image = None
target_image = None
frame_processors = []

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

def encode_execution_providers(execution_providers: list) -> list:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]

def decode_execution_providers(execution_providers: list) -> list:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]

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
    global source_image, target_image, frame_processors
    
    if source_image is None or target_image is None:
        return jsonify({'error': 'Both source and target images are required'}), 400
    
    # Process the target image with face swapping
    result_image = target_image.copy()
    for frame_processor in frame_processors:
        result_image = frame_processor.process_image(source_image, result_image)
    
    # Convert the result to base64
    result_pil = Image.fromarray(result_image)
    buffered = io.BytesIO()
    result_pil.save(buffered, format="JPEG")
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
    global source_image, frame_processors
    
    if source_image is None:
        return
    
    # Decode the frame
    frame_data = base64.b64decode(data['frame'])
    frame_np = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    
    # Process the frame
    for frame_processor in frame_processors:
        frame = frame_processor.process_frame(source_image, frame)
    
    # Convert the processed frame back to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Send the processed frame back to the client
    emit('processed_frame', {'frame': frame_base64})

@socketio.on('update_settings')
def handle_update_settings(settings):
    # Update global settings
    modules.globals.keep_fps = settings.get('keep_fps', True)
    modules.globals.keep_audio = settings.get('keep_audio', True)
    modules.globals.many_faces = settings.get('many_faces', False)
    modules.globals.mouth_mask = settings.get('mouth_mask', False)
    modules.globals.nsfw_filter = settings.get('nsfw_filter', False)
    
    # Update frame processors
    global frame_processors
    frame_processors = get_frame_processors_modules(settings.get('frame_processors', ['face_swapper']))

if __name__ == '__main__':
    parse_args()
    socketio.run(app, debug=True) 