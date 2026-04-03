from datetime import datetime

from pydantic import BaseModel
from typing import List, Optional

from backend.app.utils.enums import StorageType



class DatasetCreate(BaseModel):
    name: str
    date: datetime
    pose_ids: List[str]
    creator_id: Optional[str]
    storage: StorageType

class DatasetUpdate(BaseModel):
    name: Optional[str]
    date: Optional[datetime]
    pose_ids: Optional[List[str]]
    creator_id: Optional[str]
    storage: Optional[StorageType]
    
class DatasetRead(DatasetCreate):
    id: str

class DatasetDelete(BaseModel):
    id: str

class DatasetResponse(BaseModel):
    id: str
    name: str
    date: datetime
    pose_ids: List[str]
    creator_id: Optional[str]
    storage: StorageType