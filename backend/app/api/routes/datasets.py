from typing import Optional
 
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
 
from app.core.infrastructure import get_db, get_redis, get_s3
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.redis_repository import RedisRepository
from app.repositories.s3_repository import S3Repository
from app.schemas.dataset import DatasetResponse, DatasetUpdate
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["datasets"])

def get_service(
    session: AsyncSession = Depends(get_db)
) -> DatasetService:
    return DatasetService(
        db_repo=DatasetRepository(session),
        redis_repo=RedisRepository(get_redis()),
        s3_repo=S3Repository(get_s3())
    )


@router.post("/create", response_model=DatasetResponse)
async def create_dataset(
    files: list[UploadFile] = File(..., description="One image per pose"),
    name: Optional[str] = Form(None),
    creator_id: Optional[str] = Form(None),
    service: DatasetService = Depends(get_service),
):
    """Upload images → run MediaPipe → cache landmarks in Redis. Does not touch SQL."""
    if not files:
        raise HTTPException(status_code=422, detail="At least one image is required")
    return await service.create(files, name, creator_id)
    
@router.get("/list", response_model=list[DatasetResponse])
async def list_datasets(service: DatasetService = Depends(get_service)):
    """List all temporary datasets currently in Redis."""
    return await service.list_temp()
 
 
@router.get("/read/{dataset_id}", response_model=DatasetResponse)
async def read_dataset(dataset_id: str, service: DatasetService = Depends(get_service)):
    """Read a committed dataset from the permanent SQL database."""
    return await service.read(dataset_id)
@router.post("/commit/{dataset_id}", response_model=DatasetResponse)
async def commit_dataset(dataset_id: str, service: DatasetService = Depends(get_service)):
    """Flush Redis → SQL and promote images from S3 temp/ → permanent/."""
    return await service.commit(dataset_id)
 
 
@router.post("/cache/{dataset_id}", response_model=DatasetResponse)
async def cache_dataset(dataset_id: str, service: DatasetService = Depends(get_service)):
    """Pull a committed SQL dataset back into Redis as a numpy array."""
    return await service.cache_dataset(dataset_id)
 
 
@router.patch("/update/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: str,
    payload: DatasetUpdate,
    service: DatasetService = Depends(get_service),
):
    """Patch metadata on a temporary (Redis) or permanent (SQL) dataset."""
    return await service.update(dataset_id, payload)
 
 
@router.delete("/delete/{dataset_id}", response_model=DatasetResponse)
async def delete_dataset(dataset_id: str, service: DatasetService = Depends(get_service)):
    """Delete a dataset and its S3 images from whichever store(s) hold it."""
    return await service.delete(dataset_id)