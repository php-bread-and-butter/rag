"""
API v1 router aggregation
"""
from fastapi import APIRouter

from app.api.v1.endpoints import unified_ingestion, unified_splitting

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    unified_ingestion.router,
    prefix="/documents",
    tags=["Document Ingestion"]
)

api_router.include_router(
    unified_splitting.router,
    prefix="/splitting",
    tags=["Text Splitting"]
)
