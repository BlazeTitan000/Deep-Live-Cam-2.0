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
import time

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

@app.route('/process_video', methods=['POST'])
def process_video_route():
    global source_image, target_video, face_swapper
    
    start_time = time.time()
    logging.info("Starting video processing request")
    
    if source_image is None:
        logging.error("Source image not found for video processing")
        return jsonify({'error': 'Source image is required'}), 400
    
    if target_video is None:
        logging.error("Target video not found for video processing")
        return jsonify({'error': 'Target video is required'}), 400
    
    if face_swapper is None:
        logging.error("Face swapper not initialized for video processing")
        return jsonify({'error': 'Face swapper not initialized'}), 500
    
    logging.info(f"Processing video: {target_video}")
    
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
        
        logging.info(f"Video properties - Total frames: {total_frames}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = os.path.join(temp_dir, f'frame_{frame_count:06d}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_count += 1
            
            if frame_count % 10 == 0:  # Log every 10 frames
                logging.info(f"Extracted frame {frame_count}/{total_frames}")
            
        cap.release()
        logging.info(f"Finished extracting {frame_count} frames")
        
        # Process frames with face swapping
        logging.info("Starting face swap processing on frames")
        process_video(source_image, frame_paths)
        logging.info("Face swap processing completed")
        
        # Create output video
        output_path = os.path.join(temp_dir, 'output.mp4')
        first_frame = cv2.imread(frame_paths[0])
        height, width = first_frame.shape[:2]
        
        logging.info(f"Creating output video with dimensions: {width}x{height}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            out.write(frame)
            if i % 10 == 0:  # Log every 10 frames
                logging.info(f"Writing frame {i+1}/{len(frame_paths)} to output video")
            
        out.release()
        logging.info("Output video created successfully")
        
        # Read the output video and convert to base64
        with open(output_path, 'rb') as f:
            video_data = f.read()
        video_base64 = base64.b64encode(video_data).decode('utf-8')
        
        processing_time = time.time() - start_time
        logging.info(f"Video processing completed in {processing_time:.2f} seconds")
        
        return jsonify({
            'success': True,
            'video': video_base64,
            'processing_time': processing_time,
            'frame_count': frame_count,
            'fps': fps
        })
        
    except Exception as e:
        logging.error(f"Error during video processing: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500
        
    finally:
        # Clean up temporary files
        try:
            shutil.rmtree(temp_dir)
            logging.info("Temporary files cleaned up")
        except Exception as e:
            logging.error(f"Error cleaning up temporary files: {str(e)}")

if __name__ == '__main__':
    args = parse_args()
    initialize_face_swapper()
    app.run(host="0.0.0.0", debug=False) 