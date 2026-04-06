from datetime import datetime

from pydantic import BaseModel
from typing import List, Optional

from app.utils.enums import StorageType

"""Schema for creating a new dataset. Working model, all created datasets will be temporary, only modifies a database if it is committed. """


class DatasetCreate(BaseModel):
    name: Optional[str] = None
    date: Optional[datetime] = None
    data: List[List[float]] # List of 3D vectors (x, y, z)
    pose_ids: List[str]
    creator_id: Optional[str]

class DatasetUpdate(BaseModel):
    name: Optional[str]
    date: Optional[datetime]
    pose_ids: Optional[List[str]]
    creator_id: Optional[str]
    

class DatasetResponse(BaseModel):
    id: str
    name: str
    date: datetime
    pose_ids: List[str]
    creator_id: Optional[str]
    storage: StorageType