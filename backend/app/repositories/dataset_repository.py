from sqlalchemy import select 
from sqlalchemy.ext.asyncio import AsyncSession 

from backend.app.models.dataset import DatasetModel

class DatasetRepository: 
    """Encapsulates all SQL operations for datasets. Receives a session via DI."""
    def __init__(self, session: AsyncSession): 
        self.session = session 
    
    async def get(self, dataset_id: str) -> DatasetModel | None: 
        result = await self.session.execute(
            select(DatasetModel).where(DatasetModel.id == dataset_id)
        )
        return result.scaler_one_or_none()
    
    async def create(self, model: DatasetModel) -> DatasetModel: 
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return model
    
    async def update(self, model: DatasetModel, fields: dict) -> DatasetModel:
        for key, value in fields.items():
            if value is not None: 
                setattr(model, key, value)
        await self.session.commit()
        await self.session.refresh(model)
        return model
    
    async def delete(self, model: DatasetModel) -> None:
        await self.session.delete(model)
        await self.session.commit()