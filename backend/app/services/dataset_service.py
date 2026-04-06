from app.core.config import settings
from app.schemas.dataset import DatasetResponse
from app.utils.enums import StorageType
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, DateTime, JSON
import io
import numpy as np
import redis.asyncio as aioredis

# infra
REDIS_URL = settings.REDIS_URL
DATABASE_URL = settings.DATABASE_URL

S3_BUCKET = settings.S3_BUCKET

TEMP_PREFIX = "datasets/temp"
PERM_PREFIX = "datasets/perm"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
 
Base = declarative_base()
 
# Namespace key used in Redis so dataset keys never collide with other data
REDIS_NS = "dataset:"

# sql model 
class DatasetModel(Base):
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    date = Column(DateTime)
    data = Column(JSON, nullable=False)          # stored as list[list[float]]
    pose_ids = Column(JSON, nullable=False)      # stored as list[str]
    creator_id = Column(String, nullable=True)
    

# helpers 
def _ndarray_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

def _bytes_to_ndarray(b: bytes) -> np.ndarray:
    buf = io.BytesIO(b)
    return np.load(buf, allow_pickle=True)

def _model_to_response(model: DatasetModel, storage: StorageType) -> DatasetResponse:
    return {
        "id": model.id,
        "name": model.name or "",
        "date": model.date,
        "pose_ids": model.pose_ids,
        "creator_id": model.creator_id or None,
        "storage": storage.value
    }
    
# service 
class DatasetService:
    """
    Thin service layer that owns all Redis ↔ SQL orchestration.
 
    Redis key schema
    ----------------
    dataset:<id>:array   – numpy .npy bytes of the 3-D point cloud
    dataset:<id>:meta    – JSON blob with name / date / pose_ids / creator_id
    """
    # helpers
    async def _redis(self) -> aioredis.Redis:
        return await aioredis.from_url(REDIS_URL)
    
    async def _get_session(self) -> AsyncSession:
        return AsyncSessionLocal()
    
    # array is the numpy array. 
    async def _redis_get_array(self, r: aioredis.Redis, dataset_id: str) -> Optional[np.ndarray]:
        raw = await r.get(f"{REDIS_NS}{dataset_id}:array")
        if raw is None:
            return None
        return _bytes_to_ndarray(raw)
    
    # meta is json blob with name / date / pose_ids / creator_id
    async def _redis_get_meta(self, r: aioredis.Redis, dataset_id: str) -> Optional[dict]:
        raw = await r.get(f"{REDIS_NS}{dataset_id}:meta")
        if raw is None:
            return None
        return json.loads(raw)
    
    @staticmethod
    async def create_dataset(dataset_create: DatasetCreate) -> DatasetResponse:
        # Create a new dataset model instance
        new_dataset = DatasetModel(
            id=str(uuid.uuid4()),
            name=dataset_create.name,
            date=dataset_create.date,
            data=dataset_create.data,  # Store as list of lists
            pose_ids=dataset_create.pose_ids,
            creator_id=dataset_create.creator_id
        )
        
        # Save to database
        async with AsyncSessionLocal() as session:
            session.add(new_dataset)
            await session.commit()
            await session.refresh(new_dataset)
        
        return _model_to_response(new_dataset, StorageType.DATABASE)