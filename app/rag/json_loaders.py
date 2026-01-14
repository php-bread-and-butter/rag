"""
Step 7: JSON File Parsing and Processing

This module provides JSON document loaders for .json and .jsonl files.
Supports multiple loading strategies:
- JSONLoader with jq_schema (langchain, field extraction)
- Custom intelligent processing (nested structure flattening)
"""
import json
import os
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class JSONDocumentLoader:
    """
    Loader for JSON and JSONL files

    Supports:
    - Regular JSON files (.json)
    - JSON Lines files (.jsonl)
    - Field extraction using jq queries
    - Intelligent nested structure processing
    """

    def __init__(
        self,
        loader_type: str = "intelligent",  # "intelligent", "jsonloader", "jsonloader_jq"
        jq_schema: Optional[str] = None,  # jq query for JSONLoader (e.g., ".employees[]")
        text_content: bool = False  # For JSONLoader, whether to extract text content
    ):
        """
        Initialize the JSON document loader

        Args:
            loader_type: Type of loader to use ('intelligent', 'jsonloader', 'jsonloader_jq')
            jq_schema: jq query schema for JSONLoader (e.g., ".employees[]" to extract each employee)
            text_content: For JSONLoader, whether to extract text content or full JSON objects
        """
        self.loader_type = loader_type.lower()
        self.jq_schema = jq_schema
        self.text_content = text_content

        if self.loader_type not in ["intelligent", "jsonloader", "jsonloader_jq"]:
            raise ValueError(
                f"Unsupported JSON loader type: {loader_type}. "
                "Choose from 'intelligent', 'jsonloader', 'jsonloader_jq'."
            )

    def _is_jsonl(self, file_path: str) -> bool:
        """Check if file is JSONL format"""
        return file_path.lower().endswith('.jsonl')

    def _load_jsonl(self, file_path: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Load JSONL file (one JSON object per line)

        Args:
            file_path: Path to the JSONL file
            metadata: Optional metadata to add

        Returns:
            List of Document objects (one per line)
        """
        documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    json_obj = json.loads(line)
                    content = json.dumps(json_obj, indent=2)

                    doc_metadata = {
                        "source_file": file_path,
                        "file_type": "application/jsonl",
                        "line_number": line_num,
                        "data_type": "jsonl_record"
                    }
                    if metadata:
                        doc_metadata.update(metadata)

                    document = Document(page_content=content, metadata=doc_metadata)
                    documents.append(document)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSONL line {line_num} in {file_path}: {str(e)}")
                    continue

        logger.info(f"Loaded {len(documents)} records from JSONL file: {file_path}")
        return documents

    def _process_json_intelligently(
        self,
        data: Dict[str, Any],
        file_path: str,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Process JSON with intelligent flattening and context preservation

        Args:
            data: Parsed JSON data
            file_path: Path to the JSON file
            metadata: Optional metadata to add

        Returns:
            List of Document objects
        """
        documents = []

        # Strategy 1: Create documents for each employee with full context
        if 'employees' in data and isinstance(data['employees'], list):
            for emp in data['employees']:
                content = f"""Employee Profile:
Name: {emp.get('name', 'N/A')}
Role: {emp.get('role', 'N/A')}
Skills: {', '.join(emp.get('skills', []))}

Projects:"""
                for proj in emp.get('projects', []):
                    content += f"\n- {proj.get('name', 'N/A')} (Status: {proj.get('status', 'N/A')})"

                doc_metadata = {
                    'source_file': file_path,
                    'data_type': 'employee_profile',
                    'employee_id': emp.get('id'),
                    'employee_name': emp.get('name'),
                    'role': emp.get('role'),
                    'file_type': 'application/json'
                }
                if metadata:
                    doc_metadata.update(metadata)

                doc = Document(page_content=content, metadata=doc_metadata)
                documents.append(doc)

        # Strategy 2: Create documents for departments
        if 'departments' in data and isinstance(data['departments'], dict):
            for dept_name, dept_info in data['departments'].items():
                content = f"""Department: {dept_name}
Head: {dept_info.get('head', 'N/A')}
Budget: ${dept_info.get('budget', 0):,}
Team Size: {dept_info.get('team_size', 0)}"""

                doc_metadata = {
                    'source_file': file_path,
                    'data_type': 'department_info',
                    'department_name': dept_name,
                    'file_type': 'application/json'
                }
                if metadata:
                    doc_metadata.update(metadata)

                doc = Document(page_content=content, metadata=doc_metadata)
                documents.append(doc)

        # Strategy 3: If no specific structure found, create a general document
        if not documents:
            # Flatten the entire JSON structure
            content = json.dumps(data, indent=2)
            doc_metadata = {
                'source_file': file_path,
                'data_type': 'json_document',
                'file_type': 'application/json'
            }
            if metadata:
                doc_metadata.update(metadata)

            doc = Document(page_content=content, metadata=doc_metadata)
            documents.append(doc)

        return documents

    def _load_with_jsonloader(
        self,
        file_path: str,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """Load using JSONLoader from langchain_community"""
        try:
            if self.jq_schema:
                # Use jq schema for field extraction
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema=self.jq_schema,
                    text_content=self.text_content
                )
            else:
                # Load entire JSON
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema='.',
                    text_content=self.text_content
                )

            documents = loader.load()

            # Add custom metadata if provided
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)

            # Add file-specific metadata
            for doc in documents:
                doc.metadata["source_file"] = file_path
                doc.metadata["file_type"] = "application/json"
                doc.metadata["loader_type"] = "jsonloader"
                if self.jq_schema:
                    doc.metadata["jq_schema"] = self.jq_schema

            logger.info(
                f"JSON file loaded (JSONLoader) | "
                f"File: {file_path} | "
                f"Documents: {len(documents)} | "
                f"JQ Schema: {self.jq_schema or '.'}"
            )

            return documents

        except Exception as e:
            logger.error(f"Error loading JSON with JSONLoader | File: {file_path} | Error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load JSON file '{file_path}': {str(e)}")

    def load_file(self, file_path: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Load a JSON or JSONL file and return Document objects

        Args:
            file_path: Path to the JSON/JSONL file
            metadata: Optional metadata to add to documents

        Returns:
            List of Document objects
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        logger.debug(f"Loading JSON file | Path: {file_path} | Loader: {self.loader_type}")

        # Handle JSONL files
        if self._is_jsonl(file_path):
            return self._load_jsonl(file_path, metadata)

        # Handle regular JSON files
        if self.loader_type == "jsonloader" or self.loader_type == "jsonloader_jq":
            return self._load_with_jsonloader(file_path, metadata)
        else:  # intelligent (default)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                documents = self._process_json_intelligently(data, file_path, metadata)
                logger.info(
                    f"JSON file loaded successfully (intelligent) | "
                    f"File: {file_path} | "
                    f"Documents: {len(documents)}"
                )
                return documents
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON format | File: {file_path} | Error: {str(e)}", exc_info=True)
                raise ValueError(f"Invalid JSON format in file '{file_path}': {str(e)}")
            except Exception as e:
                logger.error(f"Error loading JSON file | File: {file_path} | Error: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to load JSON file '{file_path}': {str(e)}")

    def load_bytes(self, json_bytes: bytes, metadata: Optional[dict] = None) -> List[Document]:
        """
        Load JSON document from bytes

        Args:
            json_bytes: JSON file content as bytes
            metadata: Optional metadata to add to documents

        Returns:
            List of Document objects
        """
        logger.debug(f"Loading JSON document from bytes | Size: {len(json_bytes)} bytes")

        # Try to detect JSONL (check if multiple lines with JSON objects)
        try:
            text_content = json_bytes.decode('utf-8')
            lines = text_content.strip().split('\n')
            # If multiple lines and each line is valid JSON, treat as JSONL
            if len(lines) > 1:
                try:
                    for line in lines[:3]:  # Check first 3 lines
                        json.loads(line.strip())
                    # Looks like JSONL
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl', encoding='utf-8') as temp_file:
                        temp_file.write(text_content)
                        temp_path = temp_file.name
                    try:
                        documents = self._load_jsonl(temp_path, metadata)
                        logger.info(f"JSONL loaded from bytes | Documents: {len(documents)}")
                        return documents
                    finally:
                        os.unlink(temp_path)
                except (json.JSONDecodeError, ValueError):
                    pass  # Not JSONL, continue as regular JSON
        except UnicodeDecodeError:
            raise ValueError("Invalid JSON file: File does not appear to be valid UTF-8 encoded JSON")

        # Regular JSON
        try:
            data = json.loads(json_bytes.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")

        # Create a temporary file for JSONLoader if needed
        if self.loader_type == "jsonloader" or self.loader_type == "jsonloader_jq":
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as temp_file:
                json.dump(data, temp_file, indent=2)
                temp_path = temp_file.name
            try:
                documents = self._load_with_jsonloader(temp_path, metadata)
                logger.info(f"JSON loaded from bytes (JSONLoader) | Documents: {len(documents)}")
                return documents
            finally:
                os.unlink(temp_path)
        else:  # intelligent
            documents = self._process_json_intelligently(data, "json_bytes_input", metadata)
            logger.info(f"JSON loaded from bytes (intelligent) | Documents: {len(documents)}")
            return documents

    def load_files(self, file_paths: List[str], metadata: Optional[dict] = None) -> List[Document]:
        """
        Load multiple JSON files

        Args:
            file_paths: List of JSON file paths
            metadata: Optional metadata to add to all documents

        Returns:
            List of Document objects from all files
        """
        all_documents = []
        for file_path in file_paths:
            documents = self.load_file(file_path, metadata)
            all_documents.extend(documents)
        return all_documents


# Convenience functions
def load_json_file(
    file_path: str,
    metadata: Optional[dict] = None,
    loader_type: str = "intelligent",
    jq_schema: Optional[str] = None,
    text_content: bool = False
) -> List[Document]:
    """
    Convenience function to load a JSON file

    Args:
        file_path: Path to the JSON file
        metadata: Optional metadata dictionary
        loader_type: Type of loader to use ('intelligent', 'jsonloader', 'jsonloader_jq')
        jq_schema: jq query schema for JSONLoader
        text_content: For JSONLoader, whether to extract text content

    Returns:
        List of Document objects
    """
    loader = JSONDocumentLoader(
        loader_type=loader_type,
        jq_schema=jq_schema,
        text_content=text_content
    )
    return loader.load_file(file_path, metadata)


def load_json_bytes(
    json_bytes: bytes,
    metadata: Optional[dict] = None,
    loader_type: str = "intelligent",
    jq_schema: Optional[str] = None,
    text_content: bool = False
) -> List[Document]:
    """
    Convenience function to load JSON document from bytes

    Args:
        json_bytes: JSON file content as bytes
        metadata: Optional metadata dictionary
        loader_type: Type of loader to use ('intelligent', 'jsonloader', 'jsonloader_jq')
        jq_schema: jq query schema for JSONLoader
        text_content: For JSONLoader, whether to extract text content

    Returns:
        List of Document objects
    """
    loader = JSONDocumentLoader(
        loader_type=loader_type,
        jq_schema=jq_schema,
        text_content=text_content
    )
    return loader.load_bytes(json_bytes, metadata)
