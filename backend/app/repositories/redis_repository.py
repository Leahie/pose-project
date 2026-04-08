import io 
import json 
from typing import Optional 

import numpy as np
import redis.asyncio as aioredis

NS = "dataset:"

def _ndarray_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

def _bytes_to_ndarray(raw: bytes) -> np.ndarray:
    return np.load(io.BytesIO(raw))

class RedisRepository: 
    """
    Encapsulates all Redis operations for datasets. 
    
    Key schema:
      dataset:<id>:array  – numpy .npy bytes  (shape: num_images × N × 3)
      dataset:<id>:meta   – JSON blob         (name, date, pose_ids, image_keys, creator_id)
    """
    def __init__(self, redis: aioredis.Redis):
        self.r = redis
        
    # Core get/set/delete
    async def get_array(self, dataset_id: str) -> Optional[np.ndarray]:
        raw = await self.r.get(f"{NS}{dataset_id}:array")
        return _bytes_to_ndarray(raw) if raw else None
    
    async def get_meta(self, dataset_id: str) -> Optional[dict]:
        raw = await self.r.get(f"{NS}{dataset_id}:meta")
        return json.loads(raw) if raw else None
    
    async def set(self, dataset_id: str, arr: np.ndarray, meta: dict) -> None: 
        await self.r.set(f"{NS}{dataset_id}:array", _ndarray_to_bytes(arr))
        await self.r.set(f"{NS}{dataset_id}:meta", json.dumps(meta))
        
    async def update_meta(self, dataset_id: str, fields: dict) -> dict:
        meta = await self.get_meta(dataset_id)
        if meta is None: 
            raise KeyError(f"No Redis meta for dataset {dataset_id}")
        meta.update({k: v for k, v in fields.items() if v is not None})
        await self.r.set(f"{NS}{dataset_id}:meta", json.dumps(meta))
        return meta
    
    async def delete(self, dataset_id: str) -> None:
        await self.r.delete(f"{NS}{dataset_id}:array", f"{NS}{dataset_id}:meta")
    
    # List
    async def list_all_meta(self) -> list[dict]:
        """Scan for all dataset meta blobs and return them as a list of dicts."""
        metas: list[dict] = []
        async for key in self.r.scan_iter(match=f"{NS}*:meta"):
            raw = await self.r.get(key)
            if raw:
                metas.append(json.loads(raw))
        return metas