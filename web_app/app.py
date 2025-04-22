import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"Python path: {sys.path}")
print(f"Project root: {project_root}")

# Import required modules
import modules.globals
from face_swapper import get_face_swapper, swap_face, process_frame, process_image, process_video
import cv2
import numpy as np
import base64
import io
from PIL import Image
import argparse
import onnxruntime
from flask import Flask, render_template, request, jsonify
import tempfile
import shutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deep-live-cam-secret'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
source_image = None
target_image = None
face_swapper = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution-provider', 
                       help='execution provider',
                       dest='execution_provider',
                       default=['cuda'],
                       choices=['cpu', 'cuda', 'dml', 'rocm', 'coreml'],
                       nargs='+')
    args = parser.parse_args()
    
    # Set execution providers
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    print(f"Using execution providers: {modules.globals.execution_providers}")
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

@app.route('/process_video', methods=['POST'])
def process_video_route():
    global source_image, face_swapper
    
    if source_image is None:
        return jsonify({'error': 'Source image is required'}), 400
    
    if face_swapper is None:
        return jsonify({'error': 'Face swapper not initialized'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No video file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        file.save(temp_video.name)
        temp_video_path = temp_video.name
    
    # Create a temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    try:
        # Extract frames from video
        cap = cv2.VideoCapture(temp_video_path)
        frame_paths = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = os.path.join(temp_dir, f'frame_{frame_count:06d}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_count += 1
            
        cap.release()
        
        # Process frames with face swapping
        process_video(source_image, frame_paths)
        
        # Create output video
        output_path = os.path.join(temp_dir, 'output.mp4')
        first_frame = cv2.imread(frame_paths[0])
        height, width = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            out.write(frame)
            
        out.release()
        
        # Read the output video and convert to base64
        with open(output_path, 'rb') as f:
            video_data = f.read()
        video_base64 = base64.b64encode(video_data).decode('utf-8')
        
        return jsonify({
            'success': True,
            'video': video_base64
        })
        
    finally:
        # Clean up temporary files
        os.unlink(temp_video_path)
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    args = parse_args()
    initialize_face_swapper()
    app.run(host="0.0.0.0", debug=False) 