import boto3 
import os 

s3 = boto3.client("s3")
BUCKET = os.getenv("S3_BUCKET")

def upload_to_s3(file_bytes, key): 
    s3.put_object(
        bucket=BUCKET, 
        Key=key, 
        Body=file_bytes, 
        ContentType="image/jpeg"
    )