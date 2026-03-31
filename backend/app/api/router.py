from fastapi import APIRouter

from app.api.routes import datasets, poses, projects, search

api_router = APIRouter()

api_router.include_router(poses.router)
api_router.include_router(datasets.router)
api_router.include_router(projects.router)
api_router.include_router(search.router)