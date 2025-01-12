import hashlib
from datetime import datetime, UTC
from typing import Optional, List

from langchain_postgres import PGVector
from langchain_redis import RedisVectorStore
from pydantic import BaseModel
from sqlalchemy import text

from config.common_settings import CommonConfig
from preprocess.index_log import Status, IndexLog, SourceType
from preprocess.vector_store_helper import VectorStoreHelper
from utils.logging_util import logger


class DocumentChunk(BaseModel):
    id: str
    content: str
    metadata: dict
    source: str
    source_type: str

class ChunkListResponse(BaseModel):
    chunks: List[DocumentChunk]
    total: int
    page: int
    page_size: int

class DocEmbeddingsProcessor:
    def __init__(self, embedding_model, vector_store, index_log_helper, config: CommonConfig):
        self.logger = logger
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.index_log_helper = index_log_helper
        self.config = config
        self.vector_store_helper = VectorStoreHelper(vector_store)

    def _get_source_type(self, extension: str) -> Optional[str]:
        """Map file extension to source type"""
        extension_mapping = {
            'pdf': 'pdf',
            'txt': 'text',
            'csv': 'csv',
            'json': 'json',
            'docx': 'docx'
        }
        return extension_mapping.get(extension.lower())

    def add_index_log(self, source: str, source_type: str, user_id: str) -> dict:
        """Add a new document to the index log or update existing one"""
        self.logger.info(f"Adding document: {source}")

        if source_type == SourceType.WEB_PAGE.value or source_type == SourceType.CONFLUENCE.value:
            # Create new log
            new_log = self.index_log_helper.create(
                source=source,
                source_type=source_type,
                checksum="To be generated",
                status=Status.PENDING,
                user_id=user_id
            )
            return {
                "message": "Document is queued for processing",
                "id": new_log.id,
                "source": source,
                "source_type": source_type
            }
        else:

            # Calculate checksum from source file
            checksum = self._calculate_checksum(source, source_type)

            # First check by checksum
            existing_log = self.index_log_helper.find_by_checksum(checksum)
            if existing_log:
                return {
                    "message": "Document with same content already exists",
                    "source": existing_log.source,
                    "source_type": existing_log.source_type,
                    "id": existing_log.id
                }

            # Then check by source path
            existing_log = self.index_log_helper.find_by_source(source, source_type)
            if existing_log:
                # Content changed, update existing log
                self._remove_existing_embeddings(source, source_type,existing_log.checksum)
                existing_log.checksum = checksum
                existing_log.status = Status.PENDING
                existing_log.modified_at = datetime.now(UTC)
                existing_log.modified_by = user_id
                self.index_log_helper.save(existing_log)
                return {
                    "message": "Document updated and queued for processing",
                    "id": existing_log.id,
                    "source": source,
                    "source_type": source_type
                }

            # Create new log
            new_log = self.index_log_helper.create(
                source=source,
                source_type=source_type,
                checksum=checksum,
                status=Status.PENDING,
                user_id=user_id
            )
            return {
                "message": "Document is queued for processing",
                "id": new_log.id,
                "source": source,
                "source_type": source_type
            }

    def _calculate_checksum(self, source: str, source_type: str) -> str:
        """Calculate checksum for a document"""
        try:
            if source_type == SourceType.WEB_PAGE.value or source_type == SourceType.CONFLUENCE.value:
                return "To be generated"

            self.logger.info(f"Calculating checksum for file: {source}")
            with open(source, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating checksum for {source}: {str(e)}")
            raise

    def _remove_existing_embeddings(self, source: str, source_type: str, checksum: str):
        """Remove existing document embeddings from vector store

        Args:
            source (str): The identifier of the document source
            source_type (str): The type of the document source
            checksum (str): The checksum of the document, used to uniquely identify the document
        """
        self.vector_store_helper.remove_existing_embeddings(source, source_type, checksum)


    def get_document_by_id(self, log_id) -> IndexLog:
        """Get document by ID"""
        log = self.index_log_helper.find_by_id(log_id)
        self.logger.info(f"Found document with id {log_id}: {log}")
        return log

    def get_document_chunks(self, log_id: str, page: int = 1, page_size: int = 10) -> ChunkListResponse:
        """
        Retrieve embedded chunks for a document with pagination, ordered by page number
        """
        # First get the document details
        doc = self.get_document_by_id(log_id)
        if not doc:
            raise ValueError(f"Document with id {log_id} not found")

        # Verify vector store type
        if not isinstance(self.vector_store, PGVector):
            raise ValueError("Operation only supported for PGVector store")

        # Calculate offset
        offset = (page - 1) * page_size

        # Use PGVector's similarity_search with filter and order_by
        filter_dict = {
            "source": doc.source,
            "source_type": doc.source_type,
            "checksum": doc.checksum
        }
        
        # Add order_by parameter for metadata.page
        order_by = [("metadata.page", "asc")]
        
        self.logger.info(f"Searching for chunks with filter: {filter_dict}")
        
        # Get all matching documents to count total
        all_docs = self.vector_store.similarity_search(
            query="",  # Empty query to get all documents
            k=10000,  # Large number to get all documents
            filter=filter_dict,
            order_by=order_by  # Add ordering
        )
        total = len(all_docs)
        
        # Get paginated results
        docs = self.vector_store.similarity_search(
            query="",  # Empty query to get all documents
            k=page_size,
            filter=filter_dict,
            offset=offset,
            order_by=order_by  # Add ordering
        )
        self.logger.info(f"Found {len(docs)} chunks for document with id {log_id}")

        # Convert to response model
        chunk_responses = [
            DocumentChunk(
                id=str(doc.metadata.get('id', '')),
                content=doc.page_content,
                metadata=doc.metadata,
                source=doc.metadata.get('source', ''),
                source_type=doc.metadata.get('source_type', '')
            )
            for doc in docs
        ]

        return ChunkListResponse(
            chunks=chunk_responses,
            total=total,
            page=page,
            page_size=page_size
        )
