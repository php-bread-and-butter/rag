"""
Application configuration settings
"""
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Project Information
    PROJECT_NAME: str = "FastAPI Tutorial"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "A simple FastAPI tutorial project"
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FILE: Optional[str] = None  # Path to log file (e.g., "logs/app.log"), None for console only

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None  # OpenAI API key for embeddings and LLM

    # GROQ Configuration
    GROQ_API_KEY: Optional[str] = None  # GROQ API key for LLM

    # ChromaDB Configuration
    CHROMA_DB_PATH: Optional[str] = None  # Path to ChromaDB persistence directory (default: ./chroma_db)

    # AWS S3 Configuration (optional - if not set, uses local storage)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_S3_BUCKET: Optional[str] = None
    AWS_REGION: Optional[str] = "us-east-1"
    AWS_S3_PREFIX: Optional[str] = "rag-documents"  # S3 key prefix

    # Local Storage Configuration
    LOCAL_STORAGE_PATH: Optional[str] = None  # Default: ./storage

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
