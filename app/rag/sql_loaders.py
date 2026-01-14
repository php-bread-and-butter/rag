"""
Step 8: SQL Database Parsing and Processing

This module provides SQL database loaders for extracting data from SQL databases.
Supports multiple loading strategies:
- SQLDatabase utility (langchain, table info and queries)
- SQLDatabaseLoader (langchain, query-based loading)
- Custom SQL to Document conversion (intelligent table and relationship extraction)
"""
import sqlite3
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders import SQLDatabaseLoader

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class SQLDocumentLoader:
    """
    Loader for SQL databases

    Supports:
    - SQLite databases
    - Table schema extraction
    - Table data extraction
    - Relationship documents (JOINs)
    - Custom query execution
    """

    def __init__(
        self,
        loader_type: str = "intelligent",  # "intelligent", "sqldatabase", "sqldatabaseloader"
        include_sample_rows: int = 5,  # Number of sample rows to include per table
        include_relationships: bool = True  # Whether to create relationship documents
    ):
        """
        Initialize the SQL document loader

        Args:
            loader_type: Type of loader to use ('intelligent', 'sqldatabase', 'sqldatabaseloader')
            include_sample_rows: Number of sample rows to include per table
            include_relationships: Whether to create relationship documents from JOINs
        """
        self.loader_type = loader_type.lower()
        self.include_sample_rows = include_sample_rows
        self.include_relationships = include_relationships

        if self.loader_type not in ["intelligent", "sqldatabase", "sqldatabaseloader"]:
            raise ValueError(
                f"Unsupported SQL loader type: {loader_type}. "
                "Choose from 'intelligent', 'sqldatabase', 'sqldatabaseloader'."
            )

    def _get_table_names(self, conn) -> List[str]:
        """Get list of table names from database"""
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        return [table[0] for table in tables]

    def _get_table_info(self, conn, table_name: str) -> Dict[str, Any]:
        """Get table schema information"""
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        # SQLite PRAGMA table_info returns: (cid, name, type, notnull, default_value, pk)
        column_info = []
        for col in columns:
            column_info.append({
                "name": col[1],
                "type": col[2],
                "not_null": bool(col[3]),
                "default": col[4],
                "primary_key": bool(col[5])
            })
        
        return {
            "table_name": table_name,
            "columns": column_info,
            "column_names": [col["name"] for col in column_info]
        }

    def _process_sql_intelligently(
        self,
        db_path: str,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Convert SQL Database to documents with intelligent context preservation

        Args:
            db_path: Path to the SQLite database file
            metadata: Optional metadata to add

        Returns:
            List of Document objects
        """
        if not Path(db_path).exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        documents = []

        try:
            # Strategy 1: Create documents for each table
            table_names = self._get_table_names(conn)

            for table_name in table_names:
                table_info = self._get_table_info(conn, table_name)
                column_names = table_info["column_names"]

                # Get table data
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()

                # Create table overview document
                table_content = f"Table: {table_name}\n"
                table_content += f"Columns: {', '.join(column_names)}\n"
                table_content += f"Total Records: {len(rows)}\n\n"

                # Add column details
                table_content += "Column Details:\n"
                for col_info in table_info["columns"]:
                    col_type = f" ({col_info['type']})" if col_info['type'] else ""
                    pk_marker = " [PRIMARY KEY]" if col_info['primary_key'] else ""
                    not_null_marker = " [NOT NULL]" if col_info['not_null'] else ""
                    table_content += f"- {col_info['name']}{col_type}{pk_marker}{not_null_marker}\n"

                # Add sample records
                if rows:
                    table_content += f"\nSample Records (showing first {min(self.include_sample_rows, len(rows))}):\n"
                    for idx, row in enumerate(rows[:self.include_sample_rows], start=1):
                        record = dict(zip(column_names, row))
                        table_content += f"\nRecord {idx}:\n"
                        for key, value in record.items():
                            table_content += f"  {key}: {value}\n"

                doc_metadata = {
                    'source': db_path,
                    'table_name': table_name,
                    'num_records': len(rows),
                    'num_columns': len(column_names),
                    'data_type': 'sql_table',
                    'loader_type': 'intelligent'
                }
                if metadata:
                    doc_metadata.update(metadata)

                doc = Document(page_content=table_content, metadata=doc_metadata)
                documents.append(doc)

            # Strategy 2: Create relationship documents (if enabled)
            if self.include_relationships and len(table_names) > 1:
                # Try to find foreign key relationships
                # For SQLite, we'll look for common patterns (e.g., lead_id, employee_id, etc.)
                relationship_docs = self._extract_relationships(conn, table_names, db_path, metadata)
                documents.extend(relationship_docs)

        finally:
            conn.close()

        logger.info(
            f"SQL database processed successfully (intelligent) | "
            f"Database: {db_path} | "
            f"Tables: {len(table_names)} | "
            f"Documents: {len(documents)}"
        )

        return documents

    def _extract_relationships(
        self,
        conn,
        table_names: List[str],
        db_path: str,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Extract relationship documents by trying common JOIN patterns

        Args:
            conn: Database connection
            table_names: List of table names
            db_path: Path to database
            metadata: Optional metadata

        Returns:
            List of relationship Document objects
        """
        documents = []
        cursor = conn.cursor()

        # Common relationship patterns to try
        # Example: employees.id = projects.lead_id
        relationship_queries = []

        # Try to find foreign key-like columns
        for table1 in table_names:
            for table2 in table_names:
                if table1 == table2:
                    continue

                # Try common foreign key patterns
                for fk_pattern in ['id', '_id', 'lead_id', 'employee_id', 'user_id', 'owner_id']:
                    try:
                        # Check if table2 has a column matching the pattern
                        cursor.execute(f"PRAGMA table_info({table2})")
                        table2_columns = [col[1] for col in cursor.fetchall()]

                        # Check if table1 has an 'id' column
                        cursor.execute(f"PRAGMA table_info({table1})")
                        table1_columns = [col[1] for col in cursor.fetchall()]

                        if 'id' in table1_columns and fk_pattern in table2_columns:
                            # Try to execute a JOIN query
                            query = f"""
                                SELECT {table1}.*, {table2}.*
                                FROM {table1}
                                JOIN {table2} ON {table1}.id = {table2}.{fk_pattern}
                                LIMIT 10
                            """
                            try:
                                cursor.execute(query)
                                relationships = cursor.fetchall()

                                if relationships:
                                    # Get column names for both tables
                                    all_columns = [f"{table1}.{col}" for col in table1_columns] + \
                                                 [f"{table2}.{col}" for col in table2_columns]

                                    rel_content = f"Relationship: {table1} <-> {table2}\n"
                                    rel_content += f"Join Condition: {table1}.id = {table2}.{fk_pattern}\n\n"
                                    rel_content += "Sample Relationships:\n"

                                    for rel in relationships[:5]:
                                        rel_dict = dict(zip(all_columns, rel))
                                        rel_content += f"\n{rel_dict}\n"

                                    rel_doc = Document(
                                        page_content=rel_content,
                                        metadata={
                                            'source': db_path,
                                            'data_type': 'sql_relationship',
                                            'table1': table1,
                                            'table2': table2,
                                            'join_condition': f"{table1}.id = {table2}.{fk_pattern}",
                                            'loader_type': 'intelligent'
                                        }
                                    )
                                    documents.append(rel_doc)
                                    break  # Found a relationship, move to next table pair
                            except sqlite3.OperationalError:
                                continue  # Query failed, try next pattern
                    except Exception:
                        continue  # Error checking columns, continue

        return documents

    def _load_with_sqldatabase(
        self,
        db_path: str,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """Load using SQLDatabase utility from langchain_community"""
        try:
            # Convert file path to URI
            db_uri = f"sqlite:///{db_path}"
            db = SQLDatabase.from_uri(db_uri)

            # Get table information
            table_names = db.get_usable_table_names()
            table_info = db.get_table_info()

            # Create a document with database schema information
            content = f"Database Schema:\n\n{table_info}"

            doc_metadata = {
                'source': db_path,
                'data_type': 'sql_schema',
                'tables': table_names,
                'loader_type': 'sqldatabase'
            }
            if metadata:
                doc_metadata.update(metadata)

            document = Document(page_content=content, metadata=doc_metadata)
            logger.info(
                f"SQL database loaded (SQLDatabase) | "
                f"Database: {db_path} | "
                f"Tables: {len(table_names)}"
            )

            return [document]

        except Exception as e:
            logger.error(f"Error loading SQL database with SQLDatabase | Database: {db_path} | Error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load SQL database '{db_path}': {str(e)}")

    def _load_with_sqldatabaseloader(
        self,
        db_path: str,
        query: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """Load using SQLDatabaseLoader from langchain_community"""
        try:
            db_uri = f"sqlite:///{db_path}"

            if query:
                # Load specific query results
                loader = SQLDatabaseLoader(db_uri, query=query)
            else:
                # Load all tables (default behavior)
                loader = SQLDatabaseLoader(db_uri)

            documents = loader.load()

            # Add custom metadata if provided
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)

            # Add file-specific metadata
            for doc in documents:
                doc.metadata["source"] = db_path
                doc.metadata["loader_type"] = "sqldatabaseloader"
                if query:
                    doc.metadata["query"] = query

            logger.info(
                f"SQL database loaded (SQLDatabaseLoader) | "
                f"Database: {db_path} | "
                f"Documents: {len(documents)} | "
                f"Query: {query or 'all tables'}"
            )

            return documents

        except Exception as e:
            logger.error(f"Error loading SQL database with SQLDatabaseLoader | Database: {db_path} | Error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load SQL database '{db_path}': {str(e)}")

    def load_database(
        self,
        db_path: str,
        query: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Load a SQLite database and return Document objects

        Args:
            db_path: Path to the SQLite database file
            query: Optional SQL query to execute (for SQLDatabaseLoader)
            metadata: Optional metadata to add to documents

        Returns:
            List of Document objects
        """
        if not Path(db_path).exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")

        logger.debug(f"Loading SQL database | Path: {db_path} | Loader: {self.loader_type}")

        if self.loader_type == "sqldatabase":
            return self._load_with_sqldatabase(db_path, metadata)
        elif self.loader_type == "sqldatabaseloader":
            return self._load_with_sqldatabaseloader(db_path, query, metadata)
        else:  # intelligent (default)
            return self._process_sql_intelligently(db_path, metadata)


# Convenience functions
def load_sql_database(
    db_path: str,
    metadata: Optional[dict] = None,
    loader_type: str = "intelligent",
    query: Optional[str] = None,
    include_sample_rows: int = 5,
    include_relationships: bool = True
) -> List[Document]:
    """
    Convenience function to load a SQLite database

    Args:
        db_path: Path to the SQLite database file
        metadata: Optional metadata dictionary
        loader_type: Type of loader to use ('intelligent', 'sqldatabase', 'sqldatabaseloader')
        query: Optional SQL query to execute (for SQLDatabaseLoader)
        include_sample_rows: Number of sample rows to include per table (for intelligent loader)
        include_relationships: Whether to create relationship documents (for intelligent loader)

    Returns:
        List of Document objects
    """
    loader = SQLDocumentLoader(
        loader_type=loader_type,
        include_sample_rows=include_sample_rows,
        include_relationships=include_relationships
    )
    return loader.load_database(db_path, query, metadata)
