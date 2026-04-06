from sqlalchemy import Column, DateTime, JSON, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class DatasetModel(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    date = Column(DateTime(timezone=True), nullable=False)
    data = Column(JSON, nullable=False)         # list[list[list[float]]] – stacked landmarks (num_images, N, 3)
    pose_ids = Column(JSON, nullable=False)     # list[str]
    image_keys = Column(JSON, nullable=False)   # list[str] – permanent S3 keys
    creator_id = Column(String, nullable=True)