"""
API v2 router aggregation
"""
from fastapi import APIRouter

from app.api.v2.endpoints import rag_training

api_router = APIRouter()

# V2 RAG Training endpoints
api_router.include_router(
    rag_training.router,
    prefix="/rag",
    tags=["RAG Training V2"]
)
