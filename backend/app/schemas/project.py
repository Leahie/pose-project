from pydantic import BaseModel
from typing import List


class ProjectCreate(BaseModel):
    name: str
    pose_ids: List[str]   # enforce permanent in service


class ProjectResponse(BaseModel):
    id: str
    name: str
    pose_ids: List[str]