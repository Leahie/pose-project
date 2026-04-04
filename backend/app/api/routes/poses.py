from fastapi import APIRouter, HTTPException

from backend.app.schemas.pose import (
	PoseCreate,
	PoseResponse,
	PoseUpdate,
)

router = APIRouter(prefix="/poses", tags=["poses"])


@router.post("/create", response_model=PoseResponse)
async def create_pose(pose: PoseCreate) -> PoseResponse:
    
	raise HTTPException(status_code=501, detail="Not implemented")


@router.patch("/update/{pose_id}", response_model=PoseResponse)
async def update_pose(pose_id: str, pose: PoseUpdate) -> PoseResponse:
	raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/read/{pose_id}", response_model=PoseResponse)
async def read_pose(pose_id: str) -> PoseResponse:
	raise HTTPException(status_code=501, detail="Not implemented")


@router.delete("/delete/{pose_id}", response_model=PoseResponse)
async def delete_pose(pose_id: str) -> PoseResponse:
	raise HTTPException(status_code=501, detail="Not implemented")
