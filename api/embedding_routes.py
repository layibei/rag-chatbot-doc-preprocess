import traceback
from enum import Enum
import json

import dotenv
from fastapi import APIRouter, HTTPException, Query, Header, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

from config.common_settings import CommonConfig
from preprocess.index_log import SourceType, ProcessingType
from preprocess.doc_index_log_processor import DocEmbeddingsProcessor, DocumentChunk, ChunkListResponse
import os
from pathlib import Path

from preprocess.index_log.index_log_helper import IndexLogHelper
import re
from urllib.parse import unquote

from preprocess.index_log.repositories import IndexLogRepository
from utils.logging_util import logger

from preprocess.loader.loader_factories import DocumentLoaderFactory

router = APIRouter(tags=['pre-process'])

base_config = CommonConfig()
STAGING_PATH = base_config.get_embedding_config("staging_path")

URL_PATTERN = re.compile(
    r'^https?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)


class EmbeddingRequest(BaseModel):
    source: str
    source_type: SourceType


class EmbeddingResponse(BaseModel):
    message: Optional[str]
    source: str
    source_type: str
    id: Optional[str]

class DocumentCategory(Enum):
    FILE = "file"
    WEB_PAGE = "web_page"
    CONFLUENCE = "confluence"
    KNOWLEDGE_SNIPPET = "knowledge_snippet"


class IndexLogResponse(BaseModel):
    id: str
    source: str
    source_type: str
    status: str
    checksum: str
    created_at: datetime
    created_by: str
    modified_at: datetime
    modified_by: str
    error_message: Optional[str]


class TextContent(BaseModel):
    content: str
    title: Optional[str] = None


class PaginatedIndexLogResponse(BaseModel):
    items: List[IndexLogResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


@router.post("/docs", response_model=EmbeddingResponse)
def add_document(
        request: EmbeddingRequest,
        x_user_id: str = Header(...)
):
    try:
        doc_processor = DocEmbeddingsProcessor(base_config.get_model("embedding"), base_config.get_vector_store(),
                                               IndexLogHelper(IndexLogRepository(base_config.get_db_manager())), base_config)
        result = doc_processor.add_index_log(
            source=request.source,
            source_type=request.source_type,
            user_id=x_user_id
        )
        return result
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/docs/{log_id}")
def get_document_by_id(log_id: str):
    try:
        doc_processor = DocEmbeddingsProcessor(base_config.get_model("embedding"), base_config.get_vector_store(),
                                               IndexLogHelper(IndexLogRepository(base_config.get_db_manager())), base_config)
        return doc_processor.get_document_by_id(log_id)
    except ValueError as e:
        logger.error(e)
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/docs", response_model=PaginatedIndexLogResponse)
def list_documents(
    page: int = Query(1, gt=0),
    page_size: int = Query(10, gt=0),
    source: Optional[str] = None,
    source_type: Optional[SourceType] = None,
    status: Optional[str] = None,
    created_by: Optional[str] = Query(None),
    from_date: Optional[datetime] = Query(None),
    to_date: Optional[datetime] = Query(None)
):
    try:
        doc_processor = DocEmbeddingsProcessor(
            base_config.get_model("embedding"), 
            base_config.get_vector_store(),
            IndexLogHelper(IndexLogRepository(base_config.get_db_manager())), 
            base_config
        )
        
        # Build filter conditions
        filters = {}
        if source:
            filters['source'] = source
        if source_type:
            filters['source_type'] = source_type
        if status:
            filters['status'] = status.upper()
        if created_by:
            filters['created_by'] = created_by
        if from_date:
            filters['created_at_from'] = from_date
        if to_date:
            filters['created_at_to'] = to_date
        
        logger.info(f"Search by filters: {filters}")

        # Get paginated results with total count
        logs, total = doc_processor.index_log_helper.list_logs_with_count(
            page=page,
            page_size=page_size,
            filters=filters
        )

        # Convert IndexLog objects to dictionaries matching IndexLogResponse
        log_responses = [
            IndexLogResponse(
                id=log.id,
                source=log.source,
                source_type=log.source_type,
                status=log.status,
                checksum=log.checksum,
                created_at=log.created_at,
                created_by=log.created_by,
                modified_at=log.modified_at,
                modified_by=log.modified_by,
                error_message=log.error_message
            ) for log in logs
        ]

        # Calculate total pages
        total_pages = (total + page_size - 1) // page_size

        return PaginatedIndexLogResponse(
            items=log_responses,  # Use the converted responses
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f'Error:{traceback.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by:
    1. Decoding URL-encoded characters
    2. Removing or replacing special characters
    3. Replacing spaces with underscores
    """
    # Decode URL-encoded characters
    filename = unquote(filename)
    
    # Remove or replace special characters, keeping only alphanumeric, dots, dashes and underscores
    filename = re.sub(r'[^\w\-\.]', '_', filename)
    
    # Replace multiple underscores with single underscore
    filename = re.sub(r'_+', '_', filename)
    
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    
    return filename


@router.post("/docs/upload", response_model=EmbeddingResponse)
async def upload_document(
    category: DocumentCategory = Form(...),
    file: Optional[UploadFile] = None,
    url: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    x_user_id: str = Header(...)
):
    try:
        doc_processor = DocEmbeddingsProcessor(
            base_config.get_model("embedding"),
            base_config.get_vector_store(),
            IndexLogHelper(IndexLogRepository(base_config.get_db_manager())), 
            base_config
        )

        if category == DocumentCategory.KNOWLEDGE_SNIPPET:
            if not content:
                raise HTTPException(
                    status_code=400,
                    detail="Content is required for knowledge snippet"
                )
            
            # Validate content length
            if len(content) > 2000:
                raise HTTPException(
                    status_code=400,
                    detail="Content must be less than 2000 characters"
                )

            # Create a JSON string as source
            snippet_data = {
                "content": content,
                "title": title or "Untitled Snippet",
                "created_at": datetime.now().isoformat()
            }
            source = json.dumps(snippet_data)
            source_type = SourceType.KNOWLEDGE_SNIPPET.value

            # Check if similar content exists to avoid duplicates
            # We use content as a key part of checksum for knowledge snippets
            import hashlib
            content_checksum = hashlib.sha256(source.encode()).hexdigest()
            
            existing_snippets = doc_processor.index_log_helper.find_by_checksum(content_checksum)
            if existing_snippets:
                raise HTTPException(
                    status_code=400,
                    detail="Similar content already exists in the knowledge base"
                )

        elif category == DocumentCategory.FILE:
            if not file:
                raise HTTPException(
                    status_code=400,
                    detail="File is required when category is 'file'"
                )

            # Infer source type from file extension
            file_extension = Path(file.filename).suffix.lower()[1:]  # Remove the dot
            source_type_enum = DocumentLoaderFactory.infer_source_type(file_extension)
            
            if not source_type_enum:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_extension}"
                )

            # Create staging directory if it doesn't exist
            staging_path = base_config.get_embedding_config()["staging_path"]
            os.makedirs(staging_path, exist_ok=True)
            
            # Generate staging file path with sanitized filename
            safe_filename = sanitize_filename(file.filename)
            staging_file_path = os.path.join(staging_path, safe_filename)

            # check if file already exists in staging
            if os.path.exists(staging_file_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"File with the same name already exists in staging: {staging_file_path}"
                )

            # check if file already exists in archive
            if os.path.exists(os.path.join(base_config.get_embedding_config()["archive_path"], safe_filename)):
                raise HTTPException(
                    status_code=500,
                    detail=f"File with the same name already exists in archive: {os.path.join(base_config.get_embedding_config()['archive_path'], safe_filename)}"
                )
            
            # Save file to staging using async read
            content = await file.read()
            with open(staging_file_path, "wb") as buffer:
                buffer.write(content)

            source = staging_file_path
            source_type = source_type_enum.value  # Convert enum to string

        else:  # WEB_PAGE or CONFLUENCE
            if not url:
                raise HTTPException(
                    status_code=400,
                    detail="URL is required for web page or confluence documents"
                )
            # Add URL validation
            if not URL_PATTERN.match(url):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid URL format. URL must start with http:// or https://"
                )
            source = url
            # Map category to source type
            category_to_source_type = {
                DocumentCategory.WEB_PAGE: SourceType.WEB_PAGE,
                DocumentCategory.CONFLUENCE: SourceType.CONFLUENCE
            }
            source_type_enum = category_to_source_type[category]
            source_type = source_type_enum.value

            # check if the source + source type has already existed in index log
            if doc_processor.index_log_helper.find_by_source(source, source_type):
                raise HTTPException(
                    status_code=500,
                    detail=f"Document with source {source} and source type {source_type} already exists"
                )

        # Check if hierarchical processing is enabled for this source type
        hierarchical_enabled = base_config.get_embedding_config().get("hierarchical", {}).get("enabled_for", {}).get(source_type.lower(), False)
        
        # Set processing type based on configuration
        metadata = None
        if hierarchical_enabled:
            metadata = {
                "processing_type": ProcessingType.HIERARCHICAL.value
            }
            logger.info(f"Using hierarchical processing for {source_type}")
        else:
            logger.info(f"Using standard processing for {source_type}")
        
        # Process the document with appropriate metadata
        result = doc_processor.add_index_log(
            source=source,
            source_type=source_type,
            user_id=x_user_id,
            metadata=metadata
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in upload_document: {str(e)}, stack_trace:{traceback.format_exc()}")
        if category == DocumentCategory.FILE and 'staging_file_path' in locals() and os.path.exists(staging_file_path):
            os.remove(staging_file_path)
        raise HTTPException(status_code=500, detail=str(e))



@router.delete("/docs/{log_id}")
def delete_document(
    log_id: str,
    x_user_id: str = Header(...)
):
    try:
        doc_processor = DocEmbeddingsProcessor(
            base_config.get_model("embedding"), 
            base_config.get_vector_store(),
            IndexLogHelper(IndexLogRepository(base_config.get_db_manager())), base_config
        )
        
        # 1. Check if document exists
        index_log = doc_processor.get_document_by_id(log_id)
        if not index_log:
            raise HTTPException(
                status_code=404,
                detail=f"Document with id {log_id} not found"
            )
            
        # 2. Remove embedded chunks from vector store
        doc_processor.remove_existing_embeddings(index_log)
        
        # 3. Delete index log
        doc_processor.index_log_helper.delete_by_id(log_id)
        
        return {"message": f"Document {log_id} deleted successfully"}
        
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@DeprecationWarning
@router.get("/docs/{log_id}/chunks", response_model=ChunkListResponse)
def get_document_chunks(
    log_id: str,
    page: int = Query(1, gt=0),
    page_size: int = Query(10, gt=0),
    x_user_id: str = Header(...)
):
    try:
        doc_processor = DocEmbeddingsProcessor(
            base_config.get_model("embedding"),
            base_config.get_vector_store(),
            IndexLogHelper(IndexLogRepository(base_config.get_db_manager())), base_config
        )
        
        return doc_processor.get_document_chunks(
            log_id=log_id,
            page=page,
            page_size=page_size
        )

    except ValueError as e:
        logger.error(f"Error retrieving document chunks: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error retrieving document chunks: {str(e)}\nStacktrace:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_message": str(e),
                "error_code": "INTERNAL_SERVER_ERROR"
            }
        )

@router.post("/v2/docs/upload", response_model=EmbeddingResponse, deprecated=True)
async def upload_hierarchical_document(
    category: DocumentCategory = Form(...),
    file: Optional[UploadFile] = None,
    url: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    x_user_id: str = Header(...)
):
    """
    Upload and process documents with hierarchical structure (parent/child documents).
    Currently supports:
    - Confluence pages (via URL)
    - DOCX files (uploaded)
    
    The documents will be processed with a hierarchical structure, which preserves
    the parent-child relationship between document sections.
    
    DEPRECATED: Please use the regular /docs/upload endpoint, which now automatically 
    determines whether to use hierarchical processing based on configuration settings.
    """
    try:
        doc_processor = DocEmbeddingsProcessor(
            base_config.get_model("embedding"),
            base_config.get_vector_store(),
            IndexLogHelper(IndexLogRepository(base_config.get_db_manager())), 
            base_config
        )

        # Handle Confluence documents
        if category == DocumentCategory.CONFLUENCE:
            if not url:
                raise HTTPException(
                    status_code=400,
                    detail="URL is required for Confluence documents"
                )
            
            # Validate URL
            if not URL_PATTERN.match(url):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid URL format. URL must start with http:// or https://"
                )
            
            source = url
            source_type = SourceType.CONFLUENCE.value
            
            # Check if document already exists in index log
            existing_docs = doc_processor.index_log_helper.find_by_source(source, source_type)
            if existing_docs:
                # Document exists, return info about existing document
                existing_doc = existing_docs[0]
                return EmbeddingResponse(
                    message=f"Document already exists. Use ID: {existing_doc.id}",
                    source=existing_doc.source,
                    source_type=existing_doc.source_type,
                    id=existing_doc.id
                )
            
            # Process document hierarchically
            metadata = {
                "processing_type": ProcessingType.HIERARCHICAL.value
            }
            
        # Handle DOCX file uploads
        elif category == DocumentCategory.FILE:
            if not file:
                raise HTTPException(
                    status_code=400,
                    detail="File is required when category is 'file'"
                )
                
            # Infer source type from file extension
            file_extension = Path(file.filename).suffix.lower()[1:]  # Remove the dot
            source_type_enum = DocumentLoaderFactory.infer_source_type(file_extension)
            
            if not source_type_enum:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_extension}"
                )
                
            # Verify it's a DOCX file - only DOCX is supported for hierarchical processing
            if source_type_enum != SourceType.DOCX:
                raise HTTPException(
                    status_code=400,
                    detail=f"Only DOCX files are supported for hierarchical processing. Use the /docs/upload endpoint for other file types."
                )
                
            # Create staging directory if it doesn't exist
            staging_path = base_config.get_embedding_config()["staging_path"]
            os.makedirs(staging_path, exist_ok=True)
            
            # Generate staging file path with sanitized filename
            safe_filename = sanitize_filename(file.filename)
            staging_file_path = os.path.join(staging_path, safe_filename)
            
            # Check if file already exists in staging
            if os.path.exists(staging_file_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"File with the same name already exists in staging: {staging_file_path}"
                )
                
            # Check if file already exists in archive
            if os.path.exists(os.path.join(base_config.get_embedding_config()["archive_path"], safe_filename)):
                raise HTTPException(
                    status_code=500,
                    detail=f"File with the same name already exists in archive: {os.path.join(base_config.get_embedding_config()['archive_path'], safe_filename)}"
                )
                
            # Save file to staging using async read
            content = await file.read()
            with open(staging_file_path, "wb") as buffer:
                buffer.write(content)
                
            source = staging_file_path
            source_type = source_type_enum.value  # Convert enum to string
            
            # Set up metadata for hierarchical processing
            metadata = {
                "processing_type": ProcessingType.HIERARCHICAL.value
            }
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Only Confluence and DOCX files are supported for hierarchical processing. Category '{category.value}' is not supported."
            )
            
        # Process the document with hierarchical metadata
        result = doc_processor.add_index_log(
            source=source,
            source_type=source_type,
            user_id=x_user_id,
            metadata=metadata
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in upload_hierarchical_document: {str(e)}, stack_trace:{traceback.format_exc()}")
        if category == DocumentCategory.FILE and 'staging_file_path' in locals() and os.path.exists(staging_file_path):
            os.remove(staging_file_path)
        raise HTTPException(status_code=500, detail=str(e))
