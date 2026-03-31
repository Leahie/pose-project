from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Pose API"

    # SQL
    DATABASE_URL: str = "postgresql://user:password@localhost/db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # optional
    DEBUG: bool = True

    class Config:
        env_file = ".env"


settings = Settings()