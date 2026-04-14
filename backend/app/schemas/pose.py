from pydantic import BaseModel
from typing import List, Dict, Optional
from app.utils.enums import StorageType
from datetime import date 

class PoseMetadata(BaseModel):
    png: Optional[str] = None
    created_at: Optional[date] = None
    title: Optional[str] = None

    
class PoseResponse(BaseModel):
    id: str
    embedding: List[List[float]]    
    visibility: List[int] = []
    metadata: Optional[PoseMetadata] = None 