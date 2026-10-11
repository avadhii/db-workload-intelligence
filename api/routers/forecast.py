"""Stub routers — to be fully implemented in later phases."""
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def list_forecasts():
    return {"message": "Forecast endpoint — Phase 2 (Weeks 4-5)"}
