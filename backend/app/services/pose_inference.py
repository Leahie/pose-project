from fastapi import File, UploadFile
import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
import mediapipe as mp
from mediapipe.tasks import python
from PIL import Image
from pathlib import Path
import cv2
import io

BASE_DIR = Path(__file__).resolve().parent.parent

features = {
    "head" : [(8,7), (10,9), (8,10), (7,9)],
    "torso" : [(12,11), (24,23), (12,24), (11,23)],
    "left-leg": [(24,26), (26,28)],
    "right-leg": [(23,25), (25,27)],
    "legs":[(24,23), (26,25), (28,27), (30,32), (29,31)],
    "left-arm": [(12,14), (14,16)], 
    "right-arm": [(11,13), (13,15)],
    "full-body":[(0,24), (0,23), (0,12), (0,11)]
}

base_options = python.BaseOptions(
    model_asset_path=str(BASE_DIR / "data" / "pose_landmarker_full.task")
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# Process PNG to return pose landmarks
def process_png(contents: bytes):
    file_bytes = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img
    )    
    result = pose_landmarker.detect(mp_image)
    
    if not result.pose_landmarks:
        return None, None

    landmarks = result.pose_landmarks[0]  # first detected person

    points = []
    visibility = []

    for lm in landmarks:
        points.append([lm.x, lm.y, lm.z])
        visibility.append(lm.visibility)
        
    return points, visibility 
