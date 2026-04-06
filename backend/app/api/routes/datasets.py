from fastapi import APIRouter, HTTPException
from app.schemas.dataset import DatasetCreate, DatasetUpdate, DatasetResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/create", response_model=DatasetResponse)
async def create_dataset(dataset: DatasetCreate):
    

@router.patch("/update/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(dataset_id: str, dataset: DatasetUpdate):
    raise HTTPException(status_code=501, detail="Not implemented")

@router.get("/read/{dataset_id}", response_model=DatasetResponse)
async def read_dataset(dataset_id: str):
    raise HTTPException(status_code=501, detail="Not implemented")

@router.delete("/delete/{dataset_id}", response_model=DatasetResponse)
async def delete_dataset(dataset_id: str):
    raise HTTPException(status_code=501, detail="Not implemented")