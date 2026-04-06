import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from app.core.config import Settings
import boto3
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker     

_exectutor = ThreadPoolExecutor()

async def run_sync(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_exectutor, partial(fn, *args, **kwargs))

# PostgreSQL setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/dbname")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
    
# Redis setup
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_redis_pool: aioredis.Redis | None = None
def get_redis() -> aioredis.Redis:
    if _redis_pool is None:
        raise RuntimeError("Redis not initialised – call init_redis() at startup")
    return _redis_pool

async def init_redis():
    global _redis_pool 
    _redis_pool = await aioredis.from_url(REDIS_URL, decode_responses=False)

async def close_redis():
    global _redis_pool
    if _redis_pool:
        await _redis_pool.close()
        _redis_pool = None
        
# S3 setup
S3_BUCKET = Settings.S3_BUCKET
_s3_client: boto3.client | None = None

def get_s3() -> boto3.client:
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        )
    return _s3_client

def setup_minio() -> None: 
    s3 = get_s3()
    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if S3_BUCKET not in existing:
        s3.create_bucket(Bucket=S3_BUCKET)
        print(f"[minio] Created bucket: {S3_BUCKET}")
    else:
        print(f"[minio] Bucket already exists: {S3_BUCKET}")
 
    s3.put_bucket_lifecycle_configuration(
        Bucket=S3_BUCKET,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "expire-temp-datasets",
                    "Filter": {"Prefix": "datasets/temp/"},
                    "Status": "Enabled",
                    "Expiration": {"Days": 7},
                }
            ]
        },
    )
    print("[minio] Lifecycle rule set: datasets/temp/ expires after 7 days")