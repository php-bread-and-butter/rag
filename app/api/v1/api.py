"""
API v1 router aggregation
"""
from fastapi import APIRouter

from app.api.v1.endpoints import document_ingestion, text_splitting

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    document_ingestion.router,
    prefix="/documents",
    tags=["Document Ingestion"]
)

api_router.include_router(
    text_splitting.router,
    prefix="/splitting",
    tags=["Text Splitting"]
)
