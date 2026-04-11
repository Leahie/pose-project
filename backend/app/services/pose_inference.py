import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
import mediapipe as mp
from mediapipe.tasks import python
import matplotlib.pyplot as plt
import pandas as pd 
import pickle as pkl
from tqdm import tqdm
import os 
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

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

# Process PNG to return pose landmarks
def process_png(image):
    landmarks = _process_png_to_landmarks(image)
    embed = _convert_landmark_to_embed(landmarks)
    return embed 

def _normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8: 
        return np.zeros(3)
    return v/norm

def _calculate_body_frame(points):
    mid_shoulders = (points[11] + points[12]) / 2
    mid_hips = (points[23] + points[24]) / 2

    up = mid_shoulders - mid_hips
    up = _normalize_vector(up)

    right = points[12] - points[11]
    right = _normalize_vector(up)

    forward = np.cross(right, up)
    forward = _normalize_vector(forward)
    return right, up, forward

def _process_png_to_landmarks(image):
    base_options = python.BaseOptions(model_asset_path=f'{BASE_DIR}/data/pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    detection_result = detector.detect(image)

def _convert_landmark_to_embed(points, visibility): 
    Right, Up, Forward = _calculate_body_frame(points)
    embedding_result = {}
    for name, pairs in features.items():
        current = {}
        vectors = []
        vector_visibility = []
        for p1,p2 in pairs:
            v = points[p2]-points[p1]
            v = _normalize_vector(v)
            # Rotate frame so image rotations don't matter
            # v = np.array([v.dot(Right), v.dot(Up), v.dot(Forward)])
            # v = v / np.linalg.norm(v) 
            vectors.append(v)
            vector_visibility.append(min(visibility[p1], visibility[p2]))
        fv = sum(vector_visibility)/len(vector_visibility)
        current["vectors"]=vectors
        current["vector_visibility"]=vector_visibility
        current["feature_visibility"]=fv
        embedding_result[name] = current 
    return embedding_result
    
# Get [n] similar images to input given dataset
def find_n_nearest_neighbors(dataset, query_file, selected_features=features, k=5, feature_weights=None):
    query_embed = dataset[query_file]["embedding"]
    
    results = []
    for file, data in tqdm(dataset.items()):
        if file == query_file:
            continue
        
        dist = _pose_distance(
            query_embed, 
            data["embedding"], 
            selected_features, 
            feature_weights
        )
        
        results.append((file,dist))
    
    results.sort(key= lambda x: x[1])
    return results[:k]

def _pose_distance(embed1, embed2, selected_features, feature_weights=None):
    total_dist = 0.0
    total_weight = 0.0
    
    for feature in selected_features:
        if feature not in embed1 or feature not in embed2:
            continue 
        f1 = embed1[feature]
        f2 = embed2[feature]
        
        vecs1 = f1["vectors"]
        vecs2 = f2["vectors"]
        vis1 = f1["vector_visibility"]
        vis2 = f2["vector_visibility"]
        fv1 = f1["feature_visibility"]
        fv2 = f2["feature_visibility"]
        
        fw = feature_weights.get(feature, 1.0) if feature_weights else 1.0
        
        feature_dist = 0.0
        feature_weight = 0.0
        
        for i in range(len(vecs1)):
            v1 = np.array(vecs1[i])
            v2 = np.array(vecs2[i])
            
            vv = min(vis1[i], vis2[i])
            
            if vv < 0.2:
                continue
            
            d =  np.linalg.norm(v1-v2) #1-np.dot(v1,v2) #
            feature_dist += vv * d
            feature_weight += vv
        
        if feature_weight == 0:
            continue
        
        feature_dist /= feature_weight
        fvis = min(fv1, fv2)
        total_dist += fw * fvis * feature_dist
        total_weight += fw * fvis 
    
    if total_weight == 0: 
        return float("inf")
    
    return total_dist / total_weight

