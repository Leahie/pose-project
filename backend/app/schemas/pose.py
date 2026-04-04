from pydantic import BaseModel
from typing import List, Dict, Optional
from app.utils.enums import StorageType


class Feature(BaseModel):
    vectors: List[List[float]]  # List of 3D vectors (x, y, z)
    vector_visibility: List[float]  # List of visibility scores for each vector
    feature_visibility: float  # Overall visibility score for the feature
    
class PoseEmbedding(BaseModel):
    features: Dict[str, Feature]  # Dictionary of features (e.g., 'pose', 'face', 'left_hand', 'right_hand')        

class PoseMetadata(BaseModel):
    name: str | None = None
    tags: List[str] = []
    description: str | None = None
    

class PoseCreate(BaseModel):
    metadata: PoseMetadata
    embedding: PoseEmbedding
    image_url: str | None = None
    storage: StorageType

class PoseUpdate(BaseModel):
    metadata: Optional[PoseMetadata] = None
    embedding: Optional[PoseEmbedding] = None
    image_url: Optional[str] = None
    storage: Optional[StorageType] = None

    
class PoseResponse(PoseCreate):
    id: str