from fastapi import APIRouter

from app.api.routes import datasets, poses, projects

api_router = APIRouter()

api_router.include_router(poses.router)
api_router.include_router(datasets.router)
api_router.include_router(projects.router)
