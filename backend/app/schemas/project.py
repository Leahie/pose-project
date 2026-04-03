from pydantic import BaseModel
from typing import List, Optional


class ProjectCreate(BaseModel):
    name: str
    pose_ids: List[str]   # enforce permanent in service
    
class ProjectUpdate(BaseModel):
    name: Optional[str]
    pose_ids: Optional[List[str]]

class ProjectRead(ProjectCreate):
    id: str

class ProjectDelete(BaseModel):
    id: str

class ProjectResponse(BaseModel):
    id: str
    name: str
    pose_ids: List[str]