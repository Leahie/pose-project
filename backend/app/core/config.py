from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Pose API"

    # SQL
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost/pose_db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # S3
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET: str = "datasets"
    
    # optional
    DEBUG: bool = True

    class Config:
        env_file = ".env"


settings = Settings()