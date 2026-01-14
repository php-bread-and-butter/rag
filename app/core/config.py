"""
Application configuration settings
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Project Information
    PROJECT_NAME: str = "FastAPI Tutorial"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "A simple FastAPI tutorial project"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
