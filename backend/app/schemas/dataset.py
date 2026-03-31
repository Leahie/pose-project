from pydantic import BaseModel
from typing import List


class DatasetCreate(BaseModel):
    name: str
    pose_ids: List[str]


class DatasetResponse(BaseModel):
    id: str
    name: str
    pose_ids: List[str]