"""
File Storage Abstraction Layer

Handles storage of raw files to S3 (if configured) or local filesystem.
"""
import os
import uuid
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from app.core.logging_config import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class FileStorageManager:
    """
    Manages file storage with S3 or local filesystem fallback.
    """
    
    def __init__(self):
        """Initialize storage manager"""
        self.storage_type = self._detect_storage_type()
        self.storage_backend = self._initialize_backend()
        logger.info(f"FileStorageManager initialized | Storage type: {self.storage_type}")
        
    def _detect_storage_type(self) -> str:
        """Detect if S3 is configured"""
        if (settings.AWS_ACCESS_KEY_ID and 
            settings.AWS_SECRET_ACCESS_KEY and 
            settings.AWS_S3_BUCKET):
            return "s3"
        return "local"
    
    def _initialize_backend(self):
        """Initialize storage backend"""
        if self.storage_type == "s3":
            try:
                return S3StorageBackend()
            except Exception as e:
                logger.warning(f"Failed to initialize S3 backend, falling back to local: {str(e)}")
                return LocalStorageBackend()
        return LocalStorageBackend()
    
    async def save_file(
        self,
        content: bytes,
        filename: str,
        collection_name: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Save file to storage (S3 or local)
        
        Args:
            content: File content as bytes
            filename: Original filename
            collection_name: Collection/namespace for organization
            metadata: Optional metadata dictionary
        
        Returns:
            dict with storage info: {
                "storage_type": "s3" or "local",
                "file_path": "s3://bucket/path" or "/local/path",
                "file_id": "unique-id",
                "original_filename": "original.pdf"
            }
        """
        return await self.storage_backend.save_file(
            content=content,
            filename=filename,
            collection_name=collection_name,
            metadata=metadata
        )
    
    async def get_file(self, file_path: str) -> bytes:
        """Retrieve file from storage"""
        return await self.storage_backend.get_file(file_path)
    
    async def delete_file(self, file_path: str):
        """Delete file from storage"""
        return await self.storage_backend.delete_file(file_path)


class S3StorageBackend:
    """S3 storage backend"""
    
    def __init__(self):
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
        
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION or 'us-east-1'
        )
        self.bucket = settings.AWS_S3_BUCKET
        self.prefix = settings.AWS_S3_PREFIX or "rag-documents"
        logger.info(f"S3StorageBackend initialized | Bucket: {self.bucket} | Prefix: {self.prefix}")
    
    async def save_file(
        self,
        content: bytes,
        filename: str,
        collection_name: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Save file to S3"""
        file_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d")
        s3_key = f"{self.prefix}/{collection_name}/{timestamp}/{file_id}/{filename}"
        
        # Prepare metadata
        s3_metadata = {
            "original_filename": filename,
            "file_id": file_id,
            "collection_name": collection_name,
            "uploaded_at": datetime.now().isoformat()
        }
        if metadata:
            s3_metadata.update({k: str(v) for k, v in metadata.items()})
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=content,
            Metadata=s3_metadata
        )
        
        logger.info(f"File saved to S3 | Bucket: {self.bucket} | Key: {s3_key} | Size: {len(content)} bytes")
        
        return {
            "storage_type": "s3",
            "file_path": f"s3://{self.bucket}/{s3_key}",
            "s3_bucket": self.bucket,
            "s3_key": s3_key,
            "file_id": file_id,
            "original_filename": filename,
            "size_bytes": len(content)
        }
    
    async def get_file(self, s3_path: str) -> bytes:
        """Retrieve file from S3"""
        # Parse s3://bucket/key or bucket/key
        if s3_path.startswith("s3://"):
            parts = s3_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            # Assume format: bucket/key
            parts = s3_path.split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read()
        logger.info(f"File retrieved from S3 | Bucket: {bucket} | Key: {key} | Size: {len(content)} bytes")
        return content
    
    async def delete_file(self, s3_path: str):
        """Delete file from S3"""
        if s3_path.startswith("s3://"):
            parts = s3_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1]
        else:
            parts = s3_path.split("/", 1)
            bucket = parts[0]
            key = parts[1]
        
        self.s3_client.delete_object(Bucket=bucket, Key=key)
        logger.info(f"File deleted from S3 | Bucket: {bucket} | Key: {key}")


class LocalStorageBackend:
    """Local filesystem storage backend"""
    
    def __init__(self):
        self.base_path = settings.LOCAL_STORAGE_PATH or os.path.join(os.getcwd(), "storage")
        Path(self.base_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorageBackend initialized | Base path: {self.base_path}")
    
    async def save_file(
        self,
        content: bytes,
        filename: str,
        collection_name: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Save file to local filesystem"""
        file_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d")
        collection_dir = Path(self.base_path) / collection_name / timestamp
        collection_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = collection_dir / f"{file_id}_{filename}"
        file_path.write_bytes(content)
        
        logger.info(f"File saved locally | Path: {file_path} | Size: {len(content)} bytes")
        
        return {
            "storage_type": "local",
            "file_path": str(file_path),
            "file_id": file_id,
            "original_filename": filename,
            "size_bytes": len(content)
        }
    
    async def get_file(self, file_path: str) -> bytes:
        """Retrieve file from local filesystem"""
        content = Path(file_path).read_bytes()
        logger.info(f"File retrieved locally | Path: {file_path} | Size: {len(content)} bytes")
        return content
    
    async def delete_file(self, file_path: str):
        """Delete file from local filesystem"""
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.info(f"File deleted locally | Path: {file_path}")
        else:
            logger.warning(f"File not found for deletion | Path: {file_path}")
