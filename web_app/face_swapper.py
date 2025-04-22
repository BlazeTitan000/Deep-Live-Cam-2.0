import os
from typing import Any, List
import cv2
import insightface
import threading
import numpy as np

# Import modules for execution providers
import modules.globals

# Import custom types from modules
from modules.typing import Face, Frame
from modules.utilities import is_image, is_video
from modules.face_analyser import default_source_face
from modules.cluster_analysis import find_closest_centroid

# Global caches
FACE_SWAPPER = None
FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()

def get_face_analyser() -> Any:
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l')
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER

def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
            model_path_fp16 = os.path.join(model_dir, 'inswapper_128_fp16.onnx')
            
            if not os.path.exists(model_path_fp16):
                raise FileNotFoundError(f"Face Swapper model not found at {model_path_fp16}")
            
            print(f"Loading FP16 model: {os.path.basename(model_path_fp16)}")
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path_fp16, providers=modules.globals.execution_providers)
    return FACE_SWAPPER

def get_one_face(frame: Frame) -> Face:
    try:
        face_analyser = get_face_analyser()
        faces = face_analyser.get(frame)
        if faces:
            return faces[0]
        return None
    except Exception as e:
        print(f"Error in get_one_face: {e}")
        return None

def get_many_faces(frame: Frame) -> List[Face]:
    try:
        face_analyser = get_face_analyser()
        faces = face_analyser.get(frame)
        return faces if faces else []
    except Exception as e:
        print(f"Error in get_many_faces: {e}")
        return []

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    swapper = get_face_swapper()
    if swapper is None:
        print("Face swapper model not loaded, skipping swap.")
        return temp_frame
    return swapper.get(temp_frame, target_face, source_face, paste_back=True)

def process_frame(source_face: Face, temp_frame: Frame, many_faces: bool = False) -> Frame:
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

def process_image(source_image: Frame, target_image: Frame) -> Frame:
    source_face = get_one_face(source_image)
    if source_face is None:
        print("No face detected in source image")
        return target_image
    return process_frame(source_face, target_image)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Process a video with face swapping."""
    if modules.globals.map_faces and modules.globals.many_faces:
        print('Many faces enabled. Using first source image (if applicable in v2). Processing...')
    # Delegate to core video processing
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

def process_video_frame(source_face: Face, frame: Frame, many_faces: bool = False) -> Frame:
    """Process a single video frame with face swapping."""
    return process_frame(source_face, frame, many_faces)

def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    """Process multiple frames with face swapping."""
    # Note: Ensure get_one_face is called only once if possible for efficiency if !map_faces
    source_face = None
    if not modules.globals.map_faces:
        source_img = cv2.imread(source_path)
        if source_img is None:
            print(f"Could not read source image: {source_path}, skipping swap.")
            return
        source_face = get_one_face(source_img)
        if source_face is None:
            print(f"Could not find face in source image: {source_path}, skipping swap.")
            return

    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            print(f"Warning: Could not read frame {temp_frame_path}")
            if progress: progress.update(1)
            continue

        try:
            if not modules.globals.map_faces:
                if source_face: # Only process if source face was found
                    result = process_frame(source_face, temp_frame)
                else:
                    result = temp_frame # No source face, return original frame
            else:
                result = process_frame_v2(temp_frame, temp_frame_path)

            cv2.imwrite(temp_frame_path, result)
        except Exception as exception:
            print(f"Error processing frame {os.path.basename(temp_frame_path)}: {exception}")
            pass # Continue processing other frames
        finally:
            if progress:
                progress.update(1) 

def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    """Process a frame with face mapping support."""
    if is_image(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map_entry in modules.globals.souce_target_map:
                target_face = map_entry['target']['face']
                temp_frame = swap_face(source_face, target_face, temp_frame)
        elif not modules.globals.many_faces:
            for map_entry in modules.globals.souce_target_map:
                if "source" in map_entry:
                    source_face = map_entry['source']['face']
                    target_face = map_entry['target']['face']
                    temp_frame = swap_face(source_face, target_face, temp_frame)
    elif is_video(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map_entry in modules.globals.souce_target_map:
                target_frame = [f for f in map_entry['target_faces_in_frame'] if f['location'] == temp_frame_path]
                for frame in target_frame:
                    for target_face in frame['faces']:
                        temp_frame = swap_face(source_face, target_face, temp_frame)
        elif not modules.globals.many_faces:
            for map_entry in modules.globals.souce_target_map:
                if "source" in map_entry:
                    target_frame = [f for f in map_entry['target_faces_in_frame'] if f['location'] == temp_frame_path]
                    source_face = map_entry['source']['face']
                    for frame in target_frame:
                        for target_face in frame['faces']:
                            temp_frame = swap_face(source_face, target_face, temp_frame)
    else: # Fallback for neither image nor video (e.g., live feed?)
        detected_faces = get_many_faces(temp_frame)
        if modules.globals.many_faces:
            if detected_faces:
                source_face = default_source_face()
                for target_face in detected_faces:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
        elif not modules.globals.many_faces:
            if detected_faces and hasattr(modules.globals, 'simple_map') and modules.globals.simple_map:
                if len(detected_faces) <= len(modules.globals.simple_map['target_embeddings']):
                    for detected_face in detected_faces:
                        closest_centroid_index, _ = find_closest_centroid(modules.globals.simple_map['target_embeddings'], detected_face.normed_embedding)
                        temp_frame = swap_face(modules.globals.simple_map['source_faces'][closest_centroid_index], detected_face, temp_frame)
                else:
                    detected_faces_centroids = [face.normed_embedding for face in detected_faces]
                    i = 0
                    for target_embedding in modules.globals.simple_map['target_embeddings']:
                        closest_centroid_index, _ = find_closest_centroid(detected_faces_centroids, target_embedding)
                        if closest_centroid_index < len(detected_faces):
                            temp_frame = swap_face(modules.globals.simple_map['source_faces'][i], detected_faces[closest_centroid_index], temp_frame)
                        i += 1
    return temp_frame 