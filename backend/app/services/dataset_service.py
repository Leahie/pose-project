import uuid
from datetime import datetime, timezone
from typing import Optional
 
import numpy as np
from fastapi import HTTPException, UploadFile
 
from app.models.dataset import DatasetModel
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.redis_repository import RedisRepository
from app.repositories.s3_repository import S3Repository
from app.schemas.dataset import DatasetResponse, DatasetUpdate
from app.utils.enums import StorageType

def _mediapipe(image_bytes: bytes) -> list[list[float]]:
    """
    Run MediaPipe Holistic on raw image bytes.
    Returns a list of [x, y, z] landmarks, or [] if no pose is detected.
    """
    import cv2
    import mediapipe as mp
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=2) as holistic:
        results = holistic.process(image_rgb)
 
    if not results.pose_landmarks:
        return []
    return [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]

def _to_response(meta: dict, storage: StorageType) -> DatasetResponse:
    return DatasetResponse(
        id=meta["id"],
        name=meta.get("name", ""),
        date=datetime.fromisoformat(meta["date"]),
        pose_ids=meta.get("pose_ids", []),
        image_keys=meta.get("image_keys", []),
        creator_id=meta.get("creator_id"),
        storage=storage,
    )

def _model_to_response(model: DatasetModel, storage: StorageType) -> DatasetResponse:
    return DatasetResponse(
        id=model.id,
        name=model.name or "",
        date=model.date,
        pose_ids=model.pose_ids,
        image_keys=model.image_keys,
        creator_id=model.creator_id,
        storage=storage,
    )
    
class DatasetService:
    def __init__(
        self, 
        db_repo: DatasetRepository, 
        redis_repo: RedisRepository, 
        s3_repo: S3Repository
    ): 
        self.db = db_repo
        self.cache = redis_repo
        self.s3 = s3_repo
    
    async def create(
        self, 
        files: list[UploadFile], 
        name: Optional[str], 
        creator_id: Optional[str]
    ) -> DatasetResponse: 
        """
        For each uploaded image:
          1. Upload to S3 under datasets/temp/<id>/
          2. Run MediaPipe → (N, 3) landmarks
        Stack all landmarks into one numpy array and store in Redis.
        Does NOT touch SQL.
        """
        dataset_id = str(uuid.uuid4())
        pose_ids, image_keys, all_landmarks = [], [], []
        
        for file in files: 
            image_bytes = await file.read()
            filename = file.filename or f"{uuid.uuid4()}.jpg"
            
            key = await self.s3.upload_temp(dataset_id, filename, image_bytes, file.content_type or "image/jpeg")
            image_keys.append(key)
            
            landmarks = _mediapipe(image_bytes)
            all_landmarks.append(landmarks)
            pose_ids.append(str(uuid.uuid4()))
        
        arr = np.array(all_landmarks, dtype=np.float64)
        now = datetime.now(timezone.utc)
        meta = {
            "id": dataset_id,
            "name": name or "",
            "date": now.isoformat(),
            "pose_ids": pose_ids,
            "image_keys": image_keys,
            "creator_id": creator_id,
        }
        await self.cache.set(dataset_id, arr, meta)
        
        return _to_response(meta, StorageType.TEMP)
    
    async def read(self, dataset_id: str) -> DatasetResponse:
        """Read a committed dataset from SQL."""
        model = await self.db.get(dataset_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Dataset not found in database")
        return _model_to_response(model, StorageType.PERMANENT)
    
    async def commit(self, dataset_id: str) -> DatasetResponse:
        """
        Flush Redis → SQL and promote S3 images from temp/ → permanent/.
        Cleans up Redis only after a successful SQL write.
        """
        meta = await self.cache.get_meta(dataset_id)
        arr = await self.cache.get_array(dataset_id)
        if meta is None or arr is None:
            raise HTTPException(status_code=404, detail="Dataset not found in Redis cache")
 
        perm_keys = await self.s3.promote_to_permanent(meta["image_keys"], dataset_id)
 
        model = DatasetModel(
            id=dataset_id,
            name=meta.get("name", ""),
            date=datetime.fromisoformat(meta["date"]),
            data=arr.tolist(),
            pose_ids=meta.get("pose_ids", []),
            image_keys=perm_keys,
            creator_id=meta.get("creator_id"),
        )
        model = await self.db.create(model)
        await self.cache.delete(dataset_id)
 
        return _model_to_response(model, StorageType.PERMANENT)

    async def cache_dataset(self, dataset_id: str) -> DatasetResponse:
        """
        Pull a committed SQL dataset back into Redis as a numpy array.
        Image keys remain under datasets/permanent/ – not moved back to temp.
        """
        model = await self.db.get(dataset_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Dataset not found in database")
 
        arr = np.array(model.data, dtype=np.float64)
        meta = {
            "id": model.id,
            "name": model.name or "",
            "date": model.date.isoformat(),
            "pose_ids": model.pose_ids,
            "image_keys": model.image_keys,
            "creator_id": model.creator_id,
        }
        await self.cache.set(model.id, arr, meta)
 
        return _model_to_response(model, StorageType.TEMPORARY)
 
    async def update(self, dataset_id: str, payload: DatasetUpdate) -> DatasetResponse:
        """
        Patch metadata. Tries Redis first; falls back to SQL.
        Only metadata fields are updated – the numpy array is unchanged.
        """
        meta = await self.cache.get_meta(dataset_id)
        if meta:
            updates = {}
            if payload.name is not None:
                updates["name"] = payload.name
            if payload.date is not None:
                updates["date"] = payload.date.isoformat()
            if payload.pose_ids is not None:
                updates["pose_ids"] = payload.pose_ids
            if payload.creator_id is not None:
                updates["creator_id"] = payload.creator_id
            meta = await self.cache.update_meta(dataset_id, updates)
            return _to_response(meta, StorageType.TEMPORARY)
 
        model = await self.db.get(dataset_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
 
        model = await self.db.update(model, {
            "name": payload.name,
            "date": payload.date,
            "pose_ids": payload.pose_ids,
            "creator_id": payload.creator_id,
        })
        return _model_to_response(model, StorageType.PERMANENT)
 
    async def delete(self, dataset_id: str) -> DatasetResponse:
        """
        Delete from Redis + temp S3 images if uncommitted.
        Delete from SQL + permanent S3 images if committed.
        Checks both stores so a re-cached dataset is fully cleaned up.
        """
        snapshot: Optional[DatasetResponse] = None
 
        meta = await self.cache.get_meta(dataset_id)
        if meta:
            await self.s3.delete_many(meta.get("image_keys", []))
            await self.cache.delete(dataset_id)
            snapshot = _to_response(meta, StorageType.TEMPORARY)
 
        model = await self.db.get(dataset_id)
        if model:
            await self.s3.delete_many(model.image_keys or [])
            snapshot = _model_to_response(model, StorageType.PERMANENT)
            await self.db.delete(model)
 
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
 
        return snapshot
 
        