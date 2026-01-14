"""
Main FastAPI application entry point
"""
from fastapi import FastAPI

app = FastAPI(
    title="FastAPI Tutorial",
    version="0.1.0",
    description="A simple FastAPI tutorial project",
)


@app.get("/")
async def hello_world():
    """Hello World endpoint"""
    return {
        "message": "Hello, World!",
        "status": "success"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "FastAPI Tutorial"
    }
