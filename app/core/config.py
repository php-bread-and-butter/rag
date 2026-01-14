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

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
