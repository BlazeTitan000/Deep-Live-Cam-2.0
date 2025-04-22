import sys
import os
import logging
import cv2
import gfpgan
import torch
import platform
from modules.utilities import conditional_download

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
from modules.processors.frame.face_enhancer import get_face_enhancer, enhance_face
import cv2
import numpy as np
import base64
import io
from PIL import Image
import argparse
import onnxruntime
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import tempfile
import shutil
import time
import json
from modules.face_analyser import get_one_face, get_many_faces
from modules.processors.frame.face_swapper import get_face_swapper, swap_face
from modules.processors.frame.face_enhancer import get_face_enhancer, enhance_face

# Set ONNX runtime execution providers to use CUDA only
onnxruntime.set_default_logger_severity(3)  # Reduce logging

# Set CUDA environment variables (matching desktop version)
os.environ['OMP_NUM_THREADS'] = '1'  # Single thread for better CUDA performance
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set execution providers in globals (matching desktop version)
modules.globals.execution_providers = ['CUDAExecutionProvider']
modules.globals.execution_threads = 1  # Single thread for CUDA

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deep-live-cam-secret'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size

# Create static and videos directories if they don't exist
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
videos_dir = os.path.join(static_dir, 'videos')
os.makedirs(videos_dir, exist_ok=True)
logging.info(f"Created/verified static directory: {static_dir}")
logging.info(f"Created/verified videos directory: {videos_dir}")

# Global variables
source_image = None
target_image = None
target_video = None
face_swapper = None
face_enhancer = None

# Define models directory
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def initialize_models():
    global face_swapper, face_enhancer
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Download GFPGAN model if needed
    conditional_download(
        models_dir,
        [
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
        ],
    )
    
    logging.info("Initializing face swapper and face enhancer...")
    try:
        face_swapper = get_face_swapper()
        
        # Initialize face enhancer with platform-specific settings
        model_path = os.path.join(models_dir, "GFPGANv1.4.pth")
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            face_enhancer = gfpgan.GFPGANer(model_path=model_path, upscale=1, device=mps_device)
        else:
            face_enhancer = gfpgan.GFPGANer(model_path=model_path, upscale=1)
        
        logging.info("Successfully initialized face swapper and face enhancer")
    except Exception as e:
        logging.error(f"Error initializing models: {str(e)}")
        face_swapper = None
        face_enhancer = None

@app.before_first_request
def before_first_request():
    initialize_models()

def enhance_face(frame):
    global face_enhancer
    if face_enhancer is not None:
        try:
            _, _, enhanced_frame = face_enhancer.enhance(frame, paste_back=True)
            return enhanced_frame
        except Exception as e:
            logging.warning(f"Face enhancement failed: {str(e)}")
            return frame
    return frame

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
    global source_image, target_image, face_swapper, face_enhancer
    
    if source_image is None or target_image is None:
        logging.error("Missing source or target image for face swap")
        return jsonify({'error': 'Both source and target images are required'}), 400
    
    if face_swapper is None:
        logging.error("Face swapper not initialized for face swap")
        return jsonify({'error': 'Face swapper not initialized'}), 500
    
    try:
        start_time = time.time()
        
        # Get source face with optimized settings
        source_face = get_one_face(source_image)
        if source_face is None:
            logging.error("No face detected in source image")
            return jsonify({'error': 'No face detected in source image'}), 400
        logging.info(f"Source face detected with confidence: {source_face.get('det_score', 'unknown')}")
        
        # Get target face
        target_face = get_one_face(target_image)
        if target_face is None:
            logging.error("No face detected in target image")
            return jsonify({'error': 'No face detected in target image'}), 400
        logging.info(f"Target face detected with confidence: {target_face.get('det_score', 'unknown')}")
        
        # Process the image with face swapping
        result_image = swap_face(source_face, target_face, target_image)
        
        # Apply face enhancement if available and enabled
        if face_enhancer is not None:
            try:
                result_image = enhance_face(result_image)
                logging.info("Face enhancement applied successfully")
            except Exception as e:
                logging.warning(f"Face enhancement failed: {str(e)}")
        
        # Convert the result to base64 with optimized settings
        result_pil = Image.fromarray(result_image)
        buffered = io.BytesIO()
        result_pil.save(buffered, format="JPEG", quality=95, optimize=True)
        result_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'result': result_base64,
            'processing_time': processing_time
        })
        
    except Exception as e:
        logging.error(f"Error during face swap: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/process_video', methods=['GET', 'POST'])
def process_video_route():
    global source_image, target_video, face_swapper, face_enhancer
    
    def generate():
        if not all([source_image, target_video, face_swapper]):
            error_msg = json.dumps({"error": "Missing required variables"})
            yield f"data: {error_msg}\n\n"
            return

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract frames
                yield f"data: {json.dumps({'progress': 10, 'message': 'Extracting frames...'})}\n\n"
                
                cap = cv2.VideoCapture(target_video)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                frame_paths = []
                frame_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_path = os.path.join(temp_dir, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    frame_count += 1
                    
                    if frame_count % 10 == 0:
                        progress = min(10 + (frame_count / total_frames * 40), 50)
                        yield f"data: {json.dumps({'progress': progress, 'message': f'Extracted {frame_count}/{total_frames} frames...'})}\n\n"
                
                cap.release()
                
                # Process frames
                yield f"data: {json.dumps({'progress': 50, 'message': 'Processing frames...'})}\n\n"
                
                source_face = get_one_face(source_image)
                if source_face is None:
                    error_msg = json.dumps({"error": "No face detected in source image"})
                    yield f"data: {error_msg}\n\n"
                    return
                
                processed_count = 0
                for frame_path in frame_paths:
                    frame = cv2.imread(frame_path)
                    target_face = get_one_face(frame)
                    
                    if target_face:
                        frame = swap_face(source_face, target_face, frame)
                        frame = enhance_face(frame)
                    
                    cv2.imwrite(frame_path, frame)
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        progress = min(50 + (processed_count / total_frames * 40), 90)
                        yield f"data: {json.dumps({'progress': progress, 'message': f'Processed {processed_count}/{total_frames} frames...'})}\n\n"
                
                # Create output video
                yield f"data: {json.dumps({'progress': 90, 'message': 'Creating output video...'})}\n\n"
                
                output_path = os.path.join(temp_dir, "output.mp4")
                frame = cv2.imread(frame_paths[0])
                height, width = frame.shape[:2]
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                for frame_path in frame_paths:
                    frame = cv2.imread(frame_path)
                    out.write(frame)
                
                out.release()
                
                # Read the output video and convert to base64
                with open(output_path, 'rb') as f:
                    video_data = f.read()
                    video_base64 = base64.b64encode(video_data).decode('utf-8')
                
                processing_time = time.time() - start_time
                completion_data = {
                    'success': True,
                    'video_data': video_base64,
                    'processing_time': processing_time,
                    'frame_count': total_frames,
                    'fps': fps,
                    'progress': 100,
                    'message': 'Processing complete!'
                }
                
                yield f"data: {json.dumps(completion_data)}\n\n"
                
        except Exception as e:
            error_msg = json.dumps({"error": str(e)})
            yield f"data: {error_msg}\n\n"
            logging.error(f"Error in video processing: {str(e)}")
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False) 