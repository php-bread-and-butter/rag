"""
API v1 router aggregation
"""
from fastapi import APIRouter

from app.api.v1.endpoints import document_ingestion

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    document_ingestion.router,
    prefix="/documents",
    tags=["Document Ingestion"]
)
