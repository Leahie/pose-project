import boto3
from backend.app.core.infrastructure import S3_BUCKET, run_sync

TEMP_PREFIX = "datasets/temp"
PERM_PREFIX = "datasets/permanent"
 
def _temp_key(dataset_id: str, filename: str) -> str:
    return f"{TEMP_PREFIX}/{dataset_id}/{filename}"


def _perm_key(dataset_id: str, filename: str) -> str:
    return f"{PERM_PREFIX}/{dataset_id}/{filename}"

def _filename(key: str) -> str:
    return key.rsplit("/", 1)[-1]
 
class S3Repository: 
    """
    Encapsulates all S3/MinIO operations for dataset images.
    All boto3 calls are dispatched through run_sync() to avoid
    blocking the async event loop.
    """
    
    def __init__(self, s3: boto3.client):
        self.s3 = s3
    
    async def upload_temp(self, dataset_id: str, filename:str, data:bytes, content_type: str="image/jpeg") -> str:
        """Upload an image to the temp prefix. Return the S3 key."""
        key = _temp_key(dataset_id, filename)
        await run_sync(
            self.s3.put_object, 
            Bucket=S3_BUCKET, 
            key=key, 
            Body=data, 
            ContentType=content_type
        )
        return key
    
    async def promote_to_permanant(self, temp_keys: list[str], dataset_id: str) -> list[str]:
        """
        Copy each temp key to the permanent prefix then delete the originals.
        Returns the list of new permanent keys.
        """
        perm_keys: list[str] = []
        for key in temp_keys:
            new_key = _perm_key(dataset_id, _filename(key))
            await run_sync(
                self.s3.copy_object, 
                Bucket=S3_BUCKET, 
                CopySource = {"Bucket": S3_BUCKET, "Key": key}, 
                Key= new_key,
            )
            perm_keys.append(new_key)
        
        await self.delete_many(temp_keys)
        return perm_keys
    
    async def delete_many(self, keys: list[str]) -> None:
        """Batch-delete S3 objects. No-op on empty list."""
        if not keys: 
            return 
        await run_sync(
            self.s3.delete_objects,
            Bucket=S3_BUCKET, 
            Delete={"Objects": [{"Key": k} for k in keys]}  
        )