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
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import tempfile
import shutil
import time
import json

# Set ONNX runtime execution providers to use CUDA only
onnxruntime.set_default_logger_severity(3)  # Reduce logging

# Create session options with optimized settings
session_options = onnxruntime.SessionOptions()
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_options.enable_cpu_mem_arena = True
session_options.enable_mem_pattern = True
session_options.enable_mem_reuse = True

# Set CUDA environment variables for optimal performance and quality
os.environ['OMP_NUM_THREADS'] = '1'  # Single thread for better CUDA performance
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable asynchronous execution
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth
os.environ['CUDA_CACHE_DISABLE'] = '0'  # Enable CUDA cache
os.environ['CUDA_CACHE_PATH'] = '/tmp/cuda_cache'  # Set cache path

# Create CUDA cache directory
os.makedirs('/tmp/cuda_cache', exist_ok=True)

# Define optimized CUDA provider options for quality and performance
cuda_provider_options = {
    'device_id': '0',
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'gpu_mem_limit': str(40 * 1024 * 1024 * 1024),  # 40GB limit
    'cudnn_conv_algo_search': 'EXHAUSTIVE',  # Best quality algorithm
    'do_copy_in_default_stream': '1',
    'cudnn_conv_use_max_workspace': '1',
    'enable_cuda_graph': '1',
    'tunable_op_enable': '1',
    'tunable_op_tuning_enable': '1',
    'tunable_op_max_tuning_duration_ms': '2000',  # Increased tuning time for better quality
    'use_ep_level_unified_stream': '1',
    'enable_skip_layer_norm_strict_mode': '1',  # Better quality for normalization
    'prefer_nhwc': '1',
    'cudnn_conv1d_pad_to_nc1d': '1',
    'gpu_external_alloc': '1',
    'gpu_external_free': '1',
    'gpu_external_empty_cache': '1',
    'has_user_compute_stream': '1',
    'enable_cuda_graph_capture': '1',  # Enable graph capture for better quality
    'cudnn_conv_use_max_workspace': '1',  # Use maximum workspace for better quality
    'cudnn_conv_algo_search': 'EXHAUSTIVE',  # Use exhaustive search for best quality
    'enable_cpu_mem_arena': '1',  # Enable CPU memory arena for better quality
    'enable_mem_pattern': '1',  # Enable memory pattern for better quality
    'enable_mem_reuse': '1'  # Enable memory reuse for better quality
}

# Set execution providers in globals
modules.globals.execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
modules.globals.execution_threads = 1
modules.globals.execution_provider_options = {
    'CUDAExecutionProvider': cuda_provider_options,
    'CPUExecutionProvider': {
        'num_threads': '1',
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'enable_cpu_mem_arena': '1',
        'enable_mem_pattern': '1',
        'enable_mem_reuse': '1'
    }
}

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

# Initialize face swapper at startup
def initialize_face_swapper():
    global face_swapper
    logging.info("Initializing face swapper...")
    try:
        # Initialize with default parameters, settings will be applied through globals
        face_swapper = get_face_swapper()
        logging.info("Successfully initialized face swapper with optimized CUDA settings")
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
    
    try:
        start_time = time.time()
        
        # Get source face with optimized settings
        source_face = get_one_face(source_image)
        if source_face is None:
            return jsonify({'error': 'No face detected in source image'}), 400
        
        # Process the image with face swapping
        result_image = process_image(source_face, target_image)
        
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
            # Extract frames from video with optimized settings
            cap = cv2.VideoCapture(target_video)
            frame_paths = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Set video capture properties for better performance
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Reduce buffer size
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'mp4v'))
            
            yield json.dumps({'progress': 0, 'stage': 'extracting', 'message': 'Extracting frames...'}) + '\n'
            
            # Process frames in batches for better GPU utilization
            batch_size = 32  # Process 32 frames at a time
            processed_frames = []
            
            while True:
                frames = []
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                
                if not frames:
                    break
                
                # Process batch of frames
                for frame in frames:
                    processed_frame = process_frame(source_face, frame)
                    processed_frames.append(processed_frame)
                    frame_count += 1
                    
                    if frame_count % 10 == 0:
                        progress = int((frame_count / total_frames) * 100)
                        yield json.dumps({
                            'progress': progress,
                            'stage': 'processing',
                            'message': f'Processed {frame_count}/{total_frames} frames'
                        }) + '\n'
            
            cap.release()
            
            # Create output video with optimized settings
            yield json.dumps({'progress': 90, 'stage': 'creating', 'message': 'Creating output video...'}) + '\n'
            
            # Generate unique filename
            timestamp = int(time.time())
            output_filename = f'output_{timestamp}.mp4'
            output_path = os.path.join(videos_dir, output_filename)
            
            # Use the first processed frame to get dimensions
            height, width = processed_frames[0].shape[:2]
            
            # Use hardware acceleration for video encoding
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in processed_frames:
                out.write(frame)
            
            out.release()
            
            # Send completion message with video URL
            video_url = f'/static/videos/{output_filename}'
            processing_time = time.time() - start_time
            
            yield json.dumps({
                'success': True,
                'progress': 100,
                'stage': 'complete',
                'message': 'Processing completed',
                'video_url': video_url,
                'processing_time': processing_time,
                'frame_count': frame_count,
                'fps': fps,
                'average_fps': frame_count / processing_time
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
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    initialize_face_swapper()
    app.run(host="0.0.0.0", debug=False) 