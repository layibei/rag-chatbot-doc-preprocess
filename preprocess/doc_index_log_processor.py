import hashlib
from datetime import datetime, UTC
from typing import Optional, List

from langchain_postgres import PGVector
from pydantic import BaseModel

from config.common_settings import CommonConfig
from preprocess.index_log import Status, IndexLog, SourceType, ProcessingType
from preprocess.store.graph_store_helper import GraphStoreHelper
from preprocess.store.vector_store_helper import VectorStoreHelper
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
        self.graph_store_enabled = self.config.get_embedding_config("graph_store.enabled", False)
        if self.graph_store_enabled:
            self.graph_store_helper = GraphStoreHelper(self.config.get_graph_store(), self.config)

    def add_index_log(self, source: str, source_type: str, user_id: str, metadata: Optional[dict] = None) -> dict:
        """Add a new document to the index log or update existing one"""
        try:
            checksum = self._calculate_checksum(source, source_type)
            
            existing_log = self.index_log_helper.find_by_checksum(checksum)
            if existing_log:
                processing_type = existing_log.processing_type or ProcessingType.STANDARD.value
                return {
                    "message": "Document with same content already exists",
                    "id": existing_log.id,
                    "source": existing_log.source,
                    "source_type": existing_log.source_type,
                    "processing_type": processing_type
                }
            
            existing_log = self.index_log_helper.find_by_source(source, source_type)
            if existing_log:
                self.vector_store_helper.remove_existing_embeddings(source, source_type, existing_log.checksum)
                existing_log.checksum = checksum
                existing_log.status = Status.PENDING
                existing_log.modified_at = datetime.now(UTC)
                existing_log.modified_by = user_id
                
                if metadata and 'processing_type' in metadata:
                    existing_log.processing_type = metadata['processing_type']
                    
                self.index_log_helper.save(existing_log)
                
                processing_type = existing_log.processing_type or ProcessingType.STANDARD.value
                return {
                    "message": "Document updated and queued for processing",
                    "id": existing_log.id,
                    "source": source,
                    "source_type": source_type,
                    "processing_type": processing_type
                }
            
            log_data = {
                "source": source,
                "source_type": source_type,
                "checksum": checksum,
                "status": Status.PENDING,
                "user_id": user_id
            }
            
            if metadata and 'processing_type' in metadata:
                log_data["processing_type"] = metadata['processing_type']
                
            new_log = self.index_log_helper.create(**log_data)
            
            processing_type = new_log.processing_type or ProcessingType.STANDARD.value
            return {
                "message": "Document queued for processing",
                "id": new_log.id,
                "source": source,
                "source_type": source_type,
                "processing_type": processing_type
            }
        except Exception as e:
            self.logger.error(f"Error adding index log: {str(e)}")
            raise

    def _calculate_checksum(self, source: str, source_type: str) -> str:
        """Calculate checksum for a document"""
        try:
            if source_type == SourceType.WEB_PAGE.value or source_type == SourceType.CONFLUENCE.value:
                return "To be generated"
            
            if source_type == SourceType.KNOWLEDGE_SNIPPET.value:
                self.logger.info(f"Calculating checksum for knowledge snippet")
                return hashlib.sha256(source.encode()).hexdigest()

            self.logger.info(f"Calculating checksum for file: {source}")
            with open(source, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            raise

    def get_document_by_id(self, log_id) -> IndexLog:
        """Get document by ID"""
        log = self.index_log_helper.find_by_id(log_id)
        self.logger.info(f"Found document with id {log_id}: {log}")
        return log

    @DeprecationWarning
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

    def remove_existing_embeddings(self, index_log: IndexLog) -> None:
        """Remove existing embeddings for a document"""
        self.logger.info(f"Removing existing embeddings for document with source: {index_log.source}, source_type: {index_log.source_type}, checksum: {index_log.checksum}")
        self.vector_store_helper.remove_existing_embeddings(index_log.source, index_log.source_type, index_log.checksum)
        self.logger.info(f"Removed existing embeddings for document with source from vector database: {index_log.source}, source_type: {index_log.source_type}, checksum: {index_log.checksum}")

        if self.graph_store_enabled:
            self.graph_store_helper.remove_document(index_log.id)
            self.logger.info(f"Removed existing embeddings for document from graph database: {index_log.id}")

    def add_hierarchical_index_log(self, source: str, source_type: str, user_id: str) -> dict:
        """Add a new document to the index log using hierarchical processing"""
        self.logger.info(f"Adding document for hierarchical processing: {source}")
        
        
        source_type_key = source_type.lower()
        hierarchical_enabled = self.config.get_embedding_config(f"hierarchical.enabled_for.{source_type_key}", False)
        
        if not hierarchical_enabled:
            self.logger.warning(f"Hierarchical processing not enabled for {source_type}, using standard processing instead")
            return self.add_index_log(source, source_type, user_id)
        
        
        is_file_based = source_type not in [SourceType.CONFLUENCE.value, SourceType.WEB_PAGE.value, 
                                         SourceType.KNOWLEDGE_SNIPPET.value]
        
        if is_file_based:
            try:
                checksum = self._calculate_checksum(source, source_type)
                
                existing_log = self.index_log_helper.find_by_checksum(checksum)
                if existing_log:
                    processing_type = existing_log.processing_type or ProcessingType.STANDARD.value
                    return {
                        "message": "Document with same content already exists",
                        "id": existing_log.id,
                        "source": existing_log.source,
                        "source_type": existing_log.source_type,
                        "processing_type": processing_type
                    }
                
                
                existing_log = self.index_log_helper.find_by_source(source, source_type)
                if existing_log:
                    
                    self.vector_store_helper.remove_existing_embeddings(source, source_type, existing_log.checksum)
                    existing_log.checksum = checksum
                    existing_log.status = Status.PENDING
                    existing_log.modified_at = datetime.now(UTC)
                    existing_log.modified_by = user_id
                    existing_log.processing_type = ProcessingType.HIERARCHICAL.value
                    self.index_log_helper.save(existing_log)
                    return {
                        "message": "Document updated and queued for hierarchical processing",
                        "id": existing_log.id,
                        "source": source,
                        "source_type": source_type,
                        "processing_type": ProcessingType.HIERARCHICAL.value
                    }
                
                
                new_log = self.index_log_helper.create(
                    source=source,
                    source_type=source_type,
                    checksum=checksum,
                    status=Status.PENDING,
                    user_id=user_id,
                    processing_type=ProcessingType.HIERARCHICAL.value
                )
                return {
                    "message": "Document is queued for hierarchical processing",
                    "id": new_log.id,
                    "source": source,
                    "source_type": source_type,
                    "processing_type": ProcessingType.HIERARCHICAL.value
                }
            except Exception as e:
                self.logger.error(f"Error adding file for hierarchical processing: {str(e)}")
                raise
        else:
            # Web pages, Confluence, Knowledge snippets
            checksum = "To be generated" if source_type != SourceType.KNOWLEDGE_SNIPPET.value else hashlib.sha256(source.encode()).hexdigest()
            new_log = self.index_log_helper.create(
                source=source,
                source_type=source_type,
                checksum=checksum,
                status=Status.PENDING,
                user_id=user_id,
                processing_type=ProcessingType.HIERARCHICAL.value
            )
            return {
                "message": "Document is queued for hierarchical processing",
                "id": new_log.id,
                "source": source,
                "source_type": source_type,
                "processing_type": ProcessingType.HIERARCHICAL.value
            }
