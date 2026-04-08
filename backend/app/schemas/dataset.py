from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel
from app.utils.enums import StorageType

class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    date: Optional[datetime] = None
    pose_ids: Optional[List[str]] = None
    creator_id: Optional[str] = None

class DatasetResponse(BaseModel):
    id: str
    name: str
    date: datetime
    pose_ids: List[str]
    image_keys: List[str]   # S3 object keys; generate presigned URLs client-side if needed
    creator_id: Optional[str]
    storage: StorageType
 
    class Config:
        from_attributes = True