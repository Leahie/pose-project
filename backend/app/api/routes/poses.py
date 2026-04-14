from fastapi import APIRouter, HTTPException, UploadFile, File

from app.schemas.pose import PoseResponse
from PIL import Image

import numpy as np
import uuid 
from datetime import datetime

from app.services.pose_inference import process_png
from app.services.s3 import upload_to_s3
router = APIRouter(prefix="/poses", tags=["poses"])


@router.post("/create-from-png", response_model=PoseResponse)
async def create_pose(file: UploadFile = File(...)) -> PoseResponse:
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=415, detail="Incorrect File Type")
    
    contents = await file.read()

    # Save the PNG into the database get the host url 
    
    # key = f"poses/{uuid.uuid4()}.jpg"
    # upload_to_s3(contents, key)
    
    # Get the embeddings 
    points, visibility = process_png(contents)
    
    if points is None:
        raise HTTPException(status_code=400, detail="No pose detected")
    
    # Convert visibility floats to ints for schema compliance
    visibility_ints = [int(v * 100) for v in visibility]
    
    return {
        "id": str(uuid.uuid4()), 
        "embedding": points,
        "visibility": visibility_ints, 
        "metadata": {
            # "png": key,
            "created_at": datetime.now().date(),
            "title": file.filename         
        }
    }


# @router.patch("/update/{pose_id}", response_model=PoseResponse)
# async def update_pose(pose_id: str, pose: PoseUpdate) -> PoseResponse:
#     raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/read/{pose_id}", response_model=PoseResponse)
async def read_pose(pose_id: str) -> PoseResponse:
    raise HTTPException(status_code=501, detail="Not implemented")


@router.delete("/delete/{pose_id}", response_model=PoseResponse)
async def delete_pose(pose_id: str) -> PoseResponse:
    raise HTTPException(status_code=501, detail="Not implemented")
