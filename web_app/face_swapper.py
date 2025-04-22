import os
from typing import Any, List
import cv2
import insightface
import threading
import numpy as np

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            # Define paths for both FP32 and FP16 models
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
            model_path_fp32 = os.path.join(model_dir, 'inswapper_128.onnx')
            model_path_fp16 = os.path.join(model_dir, 'inswapper_128_fp16.onnx')
            chosen_model_path = None

            # Prioritize FP32 model
            if os.path.exists(model_path_fp32):
                chosen_model_path = model_path_fp32
                print(f"Loading FP32 model: {os.path.basename(chosen_model_path)}")
            # Fallback to FP16 model
            elif os.path.exists(model_path_fp16):
                chosen_model_path = model_path_fp16
                print(f"FP32 model not found. Loading FP16 model: {os.path.basename(chosen_model_path)}")
            # Error if neither model is found
            else:
                error_message = f"Face Swapper model not found. Please ensure 'inswapper_128.onnx' (recommended) or 'inswapper_128_fp16.onnx' exists in the '{model_dir}' directory."
                print(error_message)
                raise FileNotFoundError(error_message)

            # Load the chosen model
            try:
                FACE_SWAPPER = insightface.model_zoo.get_model(chosen_model_path)
            except Exception as e:
                print(f"Error loading Face Swapper model {os.path.basename(chosen_model_path)}: {e}")
                raise e
    return FACE_SWAPPER

def get_one_face(frame: np.ndarray) -> Any:
    """Get one face from the frame using insightface."""
    try:
        face_analyser = insightface.app.FaceAnalysis(name='buffalo_l')
        face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        faces = face_analyser.get(frame)
        if faces:
            return faces[0]
        return None
    except Exception as e:
        print(f"Error in get_one_face: {e}")
        return None

def get_many_faces(frame: np.ndarray) -> List[Any]:
    """Get multiple faces from the frame using insightface."""
    try:
        face_analyser = insightface.app.FaceAnalysis(name='buffalo_l')
        face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        faces = face_analyser.get(frame)
        return faces if faces else []
    except Exception as e:
        print(f"Error in get_many_faces: {e}")
        return []

def swap_face(source_face: Any, target_face: Any, temp_frame: np.ndarray) -> np.ndarray:
    swapper = get_face_swapper()
    if swapper is None:
        print("Face swapper model not loaded, skipping swap.")
        return temp_frame
    return swapper.get(temp_frame, target_face, source_face, paste_back=True)

def process_frame(source_face: Any, temp_frame: np.ndarray, many_faces: bool = False) -> np.ndarray:
    """Process a single frame with face swapping."""
    if many_faces:
        target_faces = get_many_faces(temp_frame)
        if target_faces:
            for target_face in target_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame

def process_image(source_image: np.ndarray, target_image: np.ndarray) -> np.ndarray:
    """Process an image with face swapping."""
    # Get source face
    source_face = get_one_face(source_image)
    if source_face is None:
        print("No face detected in source image")
        return target_image
    
    # Process the target image
    return process_frame(source_face, target_image)

def process_video(source_image: np.ndarray, video_path: str, output_path: str, many_faces: bool = False) -> None:
    """Process a video with face swapping."""
    # Get source face
    source_face = get_one_face(source_image)
    if source_face is None:
        print("No face detected in source image")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame = process_frame(source_face, frame, many_faces)
        
        # Write processed frame
        out.write(processed_frame)
        
        # Print progress
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    # Release resources
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to: {output_path}")

def process_video_frame(source_face: Any, frame: np.ndarray, many_faces: bool = False) -> np.ndarray:
    """Process a single video frame with face swapping."""
    return process_frame(source_face, frame, many_faces) 