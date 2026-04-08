from contextlib import asynccontextmanager
 
from fastapi import FastAPI
 
from app.api.router import router as dataset_router
from app.core.infrastructure import (
    close_redis,
    engine,
    init_redis,
    setup_minio,
)
from app.models.dataset import Base

@asynccontextmanager
async def lifespan(app:FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    await init_redis()
    
    setup_minio()
    
    yield
    
    await close_redis()
    
app = FastAPI(lifespan=lifespan)
app.include_router(dataset_router)