"""
API v1 router aggregation
"""
from fastapi import APIRouter

from app.api.v1.endpoints import unified_ingestion, unified_splitting, embeddings, rag_training, rag_training_v2

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

api_router.include_router(
    embeddings.router,
    prefix="/embeddings",
    tags=["Embeddings"]
)

# Legacy RAG Training endpoints (kept for reference)
api_router.include_router(
    rag_training.router,
    prefix="/rag",
    tags=["RAG Training (Legacy)"]
)

# New Unified RAG Training endpoints
api_router.include_router(
    rag_training_v2.router,
    prefix="/rag/v2",
    tags=["RAG Training V2"]
)
