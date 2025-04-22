import sys
import os
import logging

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

print(f"Python path: {sys.path}")
print(f"Project root: {project_root}")

# Import required modules
import modules.globals
from face_swapper import get_face_swapper, swap_face, process_frame, process_image, process_video, get_one_face
import cv2
import numpy as np
import base64
import io
from PIL import Image
import argparse
import onnxruntime
from flask import Flask, render_template, request, jsonify, Response
import tempfile
import shutil
import time
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deep-live-cam-secret'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
source_image = None
target_image = None
target_video = None
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
    logging.info(f"Using execution providers: {modules.globals.execution_providers}")
    return args

def decode_execution_providers(execution_providers: list) -> list:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]

def encode_execution_providers(execution_providers: list) -> list:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]

# Initialize face swapper at startup
def initialize_face_swapper():
    global face_swapper
    logging.info("Initializing face swapper...")
    try:
        face_swapper = get_face_swapper()
        logging.info("Successfully initialized face swapper")
    except Exception as e:
        logging.error(f"Error initializing face swapper: {str(e)}")
        face_swapper = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_source', methods=['POST'])
def upload_source():
    global source_image
    if 'file' not in request.files:
        logging.error("No file part in upload_source request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file in upload_source request")
        return jsonify({'error': 'No selected file'}), 400
    
    logging.info(f"Uploading source image: {file.filename}")
    # Read and process the image
    img = Image.open(file.stream)
    source_image = np.array(img)
    logging.info("Source image uploaded successfully")
    return jsonify({'success': True})

@app.route('/upload_target', methods=['POST'])
def upload_target():
    global target_image
    if 'file' not in request.files:
        logging.error("No file part in upload_target request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file in upload_target request")
        return jsonify({'error': 'No selected file'}), 400
    
    logging.info(f"Uploading target image: {file.filename}")
    # Read and process the image
    img = Image.open(file.stream)
    target_image = np.array(img)
    logging.info("Target image uploaded successfully")
    return jsonify({'success': True})

@app.route('/upload_target_video', methods=['POST'])
def upload_target_video():
    global target_video
    if 'file' not in request.files:
        logging.error("No file part in upload_target_video request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file in upload_target_video request")
        return jsonify({'error': 'No selected file'}), 400
    
    logging.info(f"Uploading target video: {file.filename}")
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        file.save(temp_video.name)
        target_video = temp_video.name
        logging.info(f"Target video saved to temporary file: {target_video}")
    
    return jsonify({'success': True, 'filename': file.filename})

@app.route('/swap_faces', methods=['POST'])
def swap_faces():
    global source_image, target_image, face_swapper
    
    if source_image is None or target_image is None:
        logging.error("Missing source or target image for face swap")
        return jsonify({'error': 'Both source and target images are required'}), 400
    
    if face_swapper is None:
        logging.error("Face swapper not initialized for face swap")
        return jsonify({'error': 'Face swapper not initialized'}), 500
    
    logging.info("Starting face swap process")
    # Process the target image with face swapping
    result_image = process_image(source_image, target_image)
    logging.info("Face swap completed successfully")
    
    # Convert the result to base64 with high quality
    result_pil = Image.fromarray(result_image)
    buffered = io.BytesIO()
    result_pil.save(buffered, format="JPEG", quality=95)
    result_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return jsonify({
        'success': True,
        'result': result_base64
    })

@app.route('/process_video', methods=['GET', 'POST'])
def process_video_route():
    global source_image, target_video, face_swapper
    
    def generate():
        start_time = time.time()
        logging.info("Starting video processing request")
        
        if source_image is None:
            yield json.dumps({'error': 'Source image is required'}) + '\n'
            return
            
        if target_video is None:
            yield json.dumps({'error': 'Target video is required'}) + '\n'
            return
            
        if face_swapper is None:
            yield json.dumps({'error': 'Face swapper not initialized'}) + '\n'
            return
            
        # Get source face from source image
        source_face = get_one_face(source_image)
        if source_face is None:
            yield json.dumps({'error': 'No face detected in source image'}) + '\n'
            return
            
        # Create a temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        logging.info(f"Created temporary directory for frames: {temp_dir}")
        
        try:
            # Extract frames from video
            cap = cv2.VideoCapture(target_video)
            frame_paths = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            yield json.dumps({'progress': 0, 'stage': 'extracting', 'message': 'Extracting frames...'}) + '\n'
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_path = os.path.join(temp_dir, f'frame_{frame_count:06d}.jpg')
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                frame_count += 1
                
                if frame_count % 10 == 0:
                    progress = int((frame_count / total_frames) * 33)
                    yield json.dumps({'progress': progress, 'stage': 'extracting', 'message': f'Extracted {frame_count}/{total_frames} frames'}) + '\n'
                
            cap.release()
            
            # Process frames with face swapping
            yield json.dumps({'progress': 33, 'stage': 'processing', 'message': 'Processing frames...'}) + '\n'
            
            processed_frames = []
            for i, frame_path in enumerate(frame_paths):
                frame = cv2.imread(frame_path)
                processed_frame = process_frame(source_face, frame)
                processed_frames.append(processed_frame)
                
                if i % 10 == 0:
                    progress = 33 + int((i / len(frame_paths)) * 33)
                    yield json.dumps({'progress': progress, 'stage': 'processing', 'message': f'Processed {i+1}/{len(frame_paths)} frames'}) + '\n'
            
            # Create output video
            yield json.dumps({'progress': 66, 'stage': 'creating', 'message': 'Creating output video...'}) + '\n'
            output_path = os.path.join(temp_dir, 'output.mp4')
            
            # Use the first processed frame to get dimensions
            height, width = processed_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for i, frame in enumerate(processed_frames):
                out.write(frame)
                if i % 10 == 0:
                    progress = 66 + int((i / len(processed_frames)) * 33)
                    yield json.dumps({'progress': progress, 'stage': 'creating', 'message': f'Writing frame {i+1}/{len(processed_frames)}'}) + '\n'
                
            out.release()
            
            # Send video metadata
            yield json.dumps({
                'stage': 'video_start',
                'message': 'Starting video transfer',
                'progress': 90
            }) + '\n'
            
            # Stream the video file
            with open(output_path, 'rb') as f:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    # Send chunk with proper headers
                    yield f'--frame\r\nContent-Type: application/octet-stream\r\nContent-Length: {len(chunk)}\r\n\r\n'.encode()
                    yield chunk
                    yield b'\r\n'
            
            # Send final boundary
            yield b'--frame--\r\n'
            
            # Send completion message
            yield json.dumps({
                'success': True,
                'progress': 100,
                'stage': 'complete',
                'message': 'Processing completed',
                'processing_time': time.time() - start_time,
                'frame_count': frame_count,
                'fps': fps
            }) + '\n'
            
        except Exception as e:
            logging.error(f"Error during video processing: {str(e)}", exc_info=True)
            yield json.dumps({'error': f'Error processing video: {str(e)}'}) + '\n'
            
        finally:
            try:
                shutil.rmtree(temp_dir)
                logging.info("Temporary files cleaned up")
            except Exception as e:
                logging.error(f"Error cleaning up temporary files: {str(e)}")
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    args = parse_args()
    initialize_face_swapper()
    app.run(host="0.0.0.0", debug=False) 