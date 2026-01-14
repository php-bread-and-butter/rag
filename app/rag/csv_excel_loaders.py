"""
Step 6: CSV and Excel File Loaders for Ingestion and Parsing

This module provides loaders for CSV and Excel files (.csv, .xlsx, .xls).
"""
import os
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_core.documents import Document
import pandas as pd

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class CSVExcelLoader:
    """
    Loader for CSV and Excel files
    
    Supports:
    - CSV files (.csv)
    - Excel files (.xlsx, .xls)
    - Multiple sheets in Excel files
    - Custom delimiters for CSV
    - Header row detection
    """
    
    def __init__(
        self,
        include_headers: bool = True,
        sheet_names: Optional[List[str]] = None,
        delimiter: Optional[str] = None
    ):
        """
        Initialize the CSV/Excel loader
        
        Args:
            include_headers: Whether to include column headers in text
            sheet_names: Specific sheet names to load (None for all sheets)
            delimiter: CSV delimiter (None for auto-detection)
        """
        self.include_headers = include_headers
        self.sheet_names = sheet_names
        self.delimiter = delimiter
    
    def _dataframe_to_text(self, df: pd.DataFrame, sheet_name: Optional[str] = None) -> str:
        """
        Convert DataFrame to text representation
        
        Args:
            df: pandas DataFrame
            sheet_name: Optional sheet name for Excel files
            
        Returns:
            Text representation of the DataFrame
        """
        text_parts = []
        
        if sheet_name:
            text_parts.append(f"Sheet: {sheet_name}")
            text_parts.append("=" * 50)
        
        if self.include_headers and not df.empty:
            # Add column headers
            headers = " | ".join(str(col) for col in df.columns)
            text_parts.append(f"Columns: {headers}")
            text_parts.append("-" * 50)
        
        # Convert rows to text
        for idx, row in df.iterrows():
            row_text = " | ".join(str(val) if pd.notna(val) else "" for val in row.values)
            text_parts.append(f"Row {idx + 1}: {row_text}")
        
        return "\n".join(text_parts)
    
    def _create_metadata(
        self,
        file_path: str,
        df: pd.DataFrame,
        sheet_name: Optional[str] = None,
        file_type: str = "csv"
    ) -> Dict[str, Any]:
        """
        Create metadata for the document
        
        Args:
            file_path: Path to the file
            df: pandas DataFrame
            sheet_name: Optional sheet name
            file_type: File type (csv, xlsx, xls)
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "source_file": file_path,
            "file_type": file_type,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
        }
        
        if sheet_name:
            metadata["sheet_name"] = sheet_name
        
        # Add data types information
        metadata["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Add sample values (first non-null value for each column)
        sample_values = {}
        for col in df.columns:
            first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if first_valid is not None:
                sample_values[col] = str(first_valid)[:50]  # Limit to 50 chars
        if sample_values:
            metadata["sample_values"] = sample_values
        
        return metadata
    
    def load_csv_file(
        self,
        file_path: str,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Load a CSV file
        
        Args:
            file_path: Path to the CSV file
            metadata: Optional metadata to add
            
        Returns:
            List of Document objects
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        logger.debug(f"Loading CSV file | Path: {file_path}")
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path, delimiter=self.delimiter)
            
            if df.empty:
                logger.warning(f"CSV file is empty | File: {file_path}")
                raise ValueError("CSV file contains no data")
            
            # Convert to text
            text_content = self._dataframe_to_text(df)
            
            # Create metadata
            doc_metadata = self._create_metadata(file_path, df, file_type="csv")
            if metadata:
                doc_metadata.update(metadata)
            
            # Create Document
            document = Document(page_content=text_content, metadata=doc_metadata)
            
            logger.info(
                f"CSV file loaded successfully | "
                f"File: {file_path} | "
                f"Rows: {len(df)} | "
                f"Columns: {len(df.columns)}"
            )
            
            return [document]
            
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty | File: {file_path}")
            raise ValueError("CSV file contains no data")
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error | File: {file_path} | Error: {str(e)}")
            raise ValueError(f"Failed to parse CSV file: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading CSV file | File: {file_path} | Error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load CSV file '{file_path}': {str(e)}")
    
    def load_excel_file(
        self,
        file_path: str,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Load an Excel file (.xlsx or .xls)
        
        Args:
            file_path: Path to the Excel file
            metadata: Optional metadata to add
            
        Returns:
            List of Document objects (one per sheet)
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        
        logger.debug(f"Loading Excel file | Path: {file_path}")
        
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            
            # Determine which sheets to load
            sheets_to_load = self.sheet_names if self.sheet_names else excel_file.sheet_names
            
            if not sheets_to_load:
                logger.warning(f"Excel file has no sheets | File: {file_path}")
                raise ValueError("Excel file contains no sheets")
            
            documents = []
            
            for sheet_name in sheets_to_load:
                if sheet_name not in excel_file.sheet_names:
                    logger.warning(f"Sheet '{sheet_name}' not found in Excel file | File: {file_path}")
                    continue
                
                # Read sheet
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                if df.empty:
                    logger.debug(f"Sheet '{sheet_name}' is empty | File: {file_path}")
                    continue
                
                # Convert to text
                text_content = self._dataframe_to_text(df, sheet_name=sheet_name)
                
                # Create metadata
                doc_metadata = self._create_metadata(
                    file_path,
                    df,
                    sheet_name=sheet_name,
                    file_type="xlsx" if file_path.endswith('.xlsx') else "xls"
                )
                doc_metadata["total_sheets"] = len(excel_file.sheet_names)
                doc_metadata["sheet_index"] = excel_file.sheet_names.index(sheet_name) + 1
                
                if metadata:
                    doc_metadata.update(metadata)
                
                # Create Document
                document = Document(page_content=text_content, metadata=doc_metadata)
                documents.append(document)
                
                logger.debug(
                    f"Sheet loaded | "
                    f"File: {file_path} | "
                    f"Sheet: {sheet_name} | "
                    f"Rows: {len(df)} | "
                    f"Columns: {len(df.columns)}"
                )
            
            if not documents:
                raise ValueError("No valid data found in Excel file")
            
            excel_file.close()
            
            logger.info(
                f"Excel file loaded successfully | "
                f"File: {file_path} | "
                f"Sheets: {len(documents)} | "
                f"Total sheets in file: {len(excel_file.sheet_names)}"
            )
            
            return documents
            
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error loading Excel file | File: {file_path} | Error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load Excel file '{file_path}': {str(e)}")
    
    def load_file(self, file_path: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Load a CSV or Excel file (auto-detects type)
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata to add
            
        Returns:
            List of Document objects
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            return self.load_csv_file(file_path, metadata)
        elif file_ext in ['.xlsx', '.xls']:
            return self.load_excel_file(file_path, metadata)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported: .csv, .xlsx, .xls")
    
    def load_bytes(
        self,
        filename: str,
        content: bytes,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Load CSV or Excel file from bytes
        
        Args:
            filename: Name of the file (for type detection)
            content: File content as bytes
            metadata: Optional metadata to add
            
        Returns:
            List of Document objects
        """
        file_ext = Path(filename).suffix.lower()
        
        # Create a temporary file
        suffix = file_ext if file_ext else ('.csv' if filename.endswith('.csv') else '.xlsx')
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            documents = self.load_file(temp_path, metadata)
            logger.info(f"File loaded from bytes | Filename: {filename} | Documents: {len(documents)}")
            return documents
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    
    def load_files(
        self,
        file_paths: List[str],
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Load multiple CSV/Excel files
        
        Args:
            file_paths: List of file paths
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
def load_csv_file(
    file_path: str,
    metadata: Optional[dict] = None,
    include_headers: bool = True,
    delimiter: Optional[str] = None
) -> List[Document]:
    """
    Convenience function to load a CSV file
    
    Args:
        file_path: Path to the CSV file
        metadata: Optional metadata dictionary
        include_headers: Whether to include headers
        delimiter: CSV delimiter (None for auto-detection)
        
    Returns:
        List of Document objects
    """
    loader = CSVExcelLoader(include_headers=include_headers, delimiter=delimiter)
    return loader.load_csv_file(file_path, metadata)


def load_excel_file(
    file_path: str,
    metadata: Optional[dict] = None,
    include_headers: bool = True,
    sheet_names: Optional[List[str]] = None
) -> List[Document]:
    """
    Convenience function to load an Excel file
    
    Args:
        file_path: Path to the Excel file
        metadata: Optional metadata dictionary
        include_headers: Whether to include headers
        sheet_names: Specific sheets to load (None for all)
        
    Returns:
        List of Document objects (one per sheet)
    """
    loader = CSVExcelLoader(include_headers=include_headers, sheet_names=sheet_names)
    return loader.load_excel_file(file_path, metadata)
