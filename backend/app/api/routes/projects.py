from fastapi import APIRouter, HTTPException

from app.schemas.project import (
	ProjectCreate,
	ProjectResponse,
	ProjectUpdate,
)

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("/create", response_model=ProjectResponse)
async def create_project(project: ProjectCreate) -> ProjectResponse:
	raise HTTPException(status_code=501, detail="Not implemented")


@router.patch("/update/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, project: ProjectUpdate) -> ProjectResponse:
	raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/read/{project_id}", response_model=ProjectResponse)
async def read_project(project_id: str) -> ProjectResponse:
	raise HTTPException(status_code=501, detail="Not implemented")


@router.delete("/delete/{project_id}", response_model=ProjectResponse)
async def delete_project(project_id: str) -> ProjectResponse:
	raise HTTPException(status_code=501, detail="Not implemented")
