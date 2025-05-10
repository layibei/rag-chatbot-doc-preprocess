import hashlib
import json
import os
import shutil
import traceback
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Optional, List

from apscheduler.schedulers.background import BackgroundScheduler
from langchain_core.documents import Document

from config.common_settings import CommonConfig
from preprocess.index_log import Status, SourceType, ProcessingType
from preprocess.index_log.index_log_helper import IndexLogHelper
from preprocess.index_log.repositories import IndexLogRepository
from preprocess.loader.loader_factories import DocumentLoaderFactory
from preprocess.store.graph_store_helper import GraphStoreHelper
from preprocess.store.vector_store_helper import VectorStoreHelper
from utils.id_util import get_id
from utils.lock.distributed_lock_helper import DistributedLockHelper
from utils.lock.repositories import DistributedLockRepository
from utils.logging_util import logger


class DocEmbeddingJob:
    def __init__(self):
        self.graph_store_helper = None
        self.vector_store_helper = None
        self.logger = logger
        self.config = CommonConfig()
        self.embeddings = None
        self.vector_store = None
        self.graph_store = None
        self.index_log_helper = None
        self.distributed_lock_helper = None
        self.scheduler = None

        self.embeddings = self.config.get_model("embedding")
        self.vector_store = self.config.get_vector_store()
        self.vector_store_helper = VectorStoreHelper(self.vector_store)

        self.graph_store_enabled = self.config.get_embedding_config("graph_store.enabled", False)

        # Initialize graph store
        if self.graph_store_enabled:
            self.logger.info("Graph store is enabled")
            self.graph_store = self.config.get_graph_store()
            self.graph_store_helper = GraphStoreHelper(self.graph_store, self.config)

        index_log_repo = IndexLogRepository(self.config.get_db_manager())
        self.index_log_helper = IndexLogHelper(index_log_repo)
        self.distributed_lock_helper = DistributedLockHelper(
            DistributedLockRepository(self.config.get_db_manager())
        )

    async def initialize(self):
        """Async initialization of components"""
        try:
            self.scheduler = BackgroundScheduler()
            self.setup_scheduler()
            self.logger.info("DocEmbeddingJob initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize DocEmbeddingJob: {str(e)}, stack:{traceback.format_exc()}")
            return False

    def setup_scheduler(self):
        self.scheduler.add_job(
            self.process_pending_documents,
            'interval',
            minutes=1,
            id='process_pending_documents',
            max_instances=1,  # Explicitly set max instances
            coalesce=True  # Combine missed runs into a single run
        )

        self.scheduler.add_job(
            self.scan_input_directory,
            'interval',
            minutes=1,
            id='scan_input_directory',
            max_instances=1,  # Explicitly set max instances
            coalesce=True  # Combine missed runs into a single run
        )

        self.scheduler.add_job(
            self.reset_stalled_documents,
            'interval',
            minutes=3,
            id='reset_stalled_documents',
            max_instances=1,
            coalesce=True
        )

        self.scheduler.start()

    def process_pending_documents(self):
        """Process one pending document at a time with distributed lock"""
        self.logger.info("Processing pending documents")
        if not self.distributed_lock_helper.acquire_lock("process_pending_documents"):
            self.logger.info("Failed to acquire lock, some other pod is processing the job.")
            return

        try:
            logs = self.index_log_helper.get_pending_index_logs()
            if not logs:
                self.logger.info("No pending documents found")
                return
            self.logger.info(f"Found {len(logs)} pending documents")

            for log in logs:
                self.logger.info(f"Processing a document: {log.source_type}:{log.source}")
                try:
                    # Update status to IN_PROGRESS
                    log.status = Status.IN_PROGRESS
                    log.modified_at = datetime.now(UTC)
                    log.error_message = None
                    self.index_log_helper.save(log)

                    # Process document
                    self._process_document(log)

                    source_type = SourceType(log.source_type)
                    if source_type.is_file_based():
                        # Move file to archive folder
                        archive_path = self.config.get_embedding_config()["archive_path"]
                        os.makedirs(archive_path, exist_ok=True)

                        source_file = Path(log.source)
                        archive_file = Path(archive_path) / source_file.name

                        # Move the file
                        shutil.move(str(source_file), str(archive_file))
                        # Update source path in log
                        log.source = str(archive_file).replace('\\', '/')

                    # Update status to COMPLETED
                    log.status = Status.COMPLETED
                    log.modified_at = datetime.now(UTC)
                    log.error_message = None
                    self.index_log_helper.save(log)
                    self.logger.info(f"Document processed: {log.source_type}:{log.source}")
                except Exception as e:
                    log.status = Status.FAILED
                    log.retry_count = (log.retry_count or 0) + 1
                    log.error_message = str(e)
                    log.modified_at = datetime.now(UTC)
                    self.index_log_helper.save(log)
                    self.logger.error(f"Error processing document: {log.source} - {str(e)}")
        finally:
            self.distributed_lock_helper.release_lock("process_pending_documents")
        self.logger.info("Finished processing pending documents")

    def scan_input_directory(self):
        """Scan archive directory for new documents to process"""
        self.logger.info("Scanning input directory for new documents")
        if not self.distributed_lock_helper.acquire_lock("scan_input_directory"):
            self.logger.info("Failed to acquire lock, some other pod is processing the job.")
            return

        try:
            input_path = self.config.get_embedding_config()["input_path"]
            if not os.path.exists(input_path):
                self.logger.info(f"Archive directory does not exist: {input_path}")
                return

            for file_name in os.listdir(input_path):
                file_path = os.path.join(input_path, file_name)
                if not os.path.isfile(file_path):
                    continue

                self.logger.info(f"Processing file: {file_name}")

                try:
                    # Determine source type from file extension
                    file_extension = Path(file_name).suffix.lower()[1:]  # Remove the dot
                    source_type = self._get_source_type(file_extension)
                    if not source_type:
                        self.logger.warning(f"Unsupported file type: {file_name}")
                        continue

                    # Calculate checksum
                    checksum = self._calculate_checksum(file_path)

                    # Check if already processed
                    existing_log = self.index_log_helper.find_by_checksum(checksum)
                    if existing_log:
                        # Get archive path
                        archive_path = self.config.get_embedding_config()["archive_path"]
                        staging_path = self.config.get_embedding_config()["staging_path"]

                        # if the file has been indexed
                        if (existing_log.source == os.path.join(staging_path,
                                                                file_name) or existing_log.source == os.path.join(
                            archive_path, file_name)) and existing_log.source_type == source_type:
                            self.logger.info(
                                f"Document has already been indexed: {file_name}, index_log_id: {existing_log.id}")
                            continue
                        else:
                            # move file to staging folder
                            staging_path = self.config.get_embedding_config()["staging_path"]
                            os.makedirs(staging_path, exist_ok=True)
                            stating_file_path = os.path.join(staging_path, file_name)
                            shutil.move(file_path, stating_file_path)
                            self.logger.info(f"Moved file to staging folder: {file_name} for later processing.")

                            # Same content (checksum) but different source or type - update the record
                            existing_log.source = stating_file_path
                            existing_log.source_type = source_type
                            existing_log.modified_at = datetime.now(UTC)
                            existing_log.modified_by = "system"
                            self.index_log_helper.save(existing_log)
                            self.logger.info(
                                f"Updated source information for existing document: {file_name}, index_log_id: {existing_log.id}")
                            continue

                    # Create new index log
                    self.add_index_log(
                        source=file_path,
                        source_type=source_type,
                        user_id="system",  # System user for automated processing
                        metadata={}
                    )
                    self.logger.info(f"Added new document for processing: {file_name}")

                except Exception as e:
                    self.logger.error(f"Error processing input file {file_name}: {str(e)}")
                    continue

        finally:
            self.distributed_lock_helper.release_lock("scan_input_directory")

        self.logger.info("Finished scanning input directory for new documents")

    def reset_stalled_documents(self):
        """Reset IN_PROGRESS documents that have been stuck for more than 5 minutes"""
        self.logger.info("Resetting stalled documents")
        if not self.distributed_lock_helper.acquire_lock("reset_stalled_documents"):
            self.logger.debug("Skipping reset_stalled_documents: distributed lock not acquired")
            return

        try:
            stalled_time = datetime.now(UTC) - timedelta(minutes=5)
            logs = self.index_log_helper.get_stalled_index_logs(stalled_time)

            if not logs:
                return

            for log in logs:
                self.logger.warning(f"Resetting stalled document: {log.source} (stuck since {log.modified_at})")
                log.status = Status.PENDING
                log.modified_at = datetime.now(UTC)
                log.modified_by = "system"
                log.error_message = "Reset due to stalled processing"
                self.index_log_helper.save(log)

        finally:
            self.distributed_lock_helper.release_lock("reset_stalled_documents")
        self.logger.info("Finished resetting stalled documents")

    def _calculate_checksum(self, source: str) -> str:
        """Calculate checksum for a document"""
        try:
            with open(source, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating checksum for {source}: {str(e)}")
            raise

    def add_index_log(self, source: str, source_type: str, user_id: str, metadata: Optional[dict] = None) -> dict:
        """Add a new document to the index log or update existing one"""
        # Calculate checksum from source file
        checksum = self._calculate_checksum(source)

        # Then check by source path
        existing_log = self.index_log_helper.find_by_source(source, source_type)
        if existing_log:
            # Content changed, update existing log
            self.vector_store_helper.remove_existing_embeddings(source, source_type, existing_log.checksum)
            existing_log.checksum = checksum
            existing_log.status = Status.PENDING
            existing_log.modified_at = datetime.now(UTC)
            existing_log.modified_by = user_id
            
            # Update processing_type if provided in metadata
            if metadata and 'processing_type' in metadata:
                existing_log.processing_type = metadata['processing_type']
                
            self.index_log_helper.save(existing_log)
            return {
                "message": "Document updated and queued for processing",
                "id": existing_log.id,
                "source": source,
                "source_type": source_type
            }

        # Create new log
        log_data = {
            "source": source,
            "source_type": source_type,
            "checksum": checksum,
            "status": Status.PENDING,
            "user_id": user_id
        }
        
        # Add processing_type if provided in metadata
        if metadata and 'processing_type' in metadata:
            log_data["processing_type"] = metadata['processing_type']
            
        new_log = self.index_log_helper.create(**log_data)
        
        return {
            "message": "Document queued for processing",
            "id": new_log.id,
            "source": source,
            "source_type": source_type
        }

    def add_documents_in_batches(self, documents: List[Document], batch_size: int = 10):
        """Add documents to vector store in batches to prevent memory issues"""
        try:
            total_batches = (len(documents) - 1) // batch_size + 1
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                current_batch = i // batch_size + 1
                self.logger.info(f"Adding batch {current_batch} of {total_batches} to vector store")
                self.vector_store.add_documents(batch)
            self.logger.info(f"Successfully added all {len(documents)} documents in {total_batches} batches")
        except Exception as e:
            self.logger.error(f"Error adding documents in batches: {str(e)}")
            raise

    def _process_document(self, log):
        """Process a single document"""
        try:
            self.logger.info(f"Processing document: {log.source_type}:{log.source}")
            
            # Check if this is a hierarchical processing request
            is_hierarchical = log.processing_type == ProcessingType.HIERARCHICAL.value
            
            # Get appropriate loader
            loader = DocumentLoaderFactory.get_loader(log.source_type)

            # Load document - using hierarchical approach if supported and requested
            if is_hierarchical:
                if log.source_type == SourceType.CONFLUENCE.value:
                    self.logger.info(f"Using hierarchical loading for Confluence: {log.source}")
                    documents = loader.hierarchical_load(log.source)
                elif log.source_type == SourceType.DOCX.value:
                    self.logger.info(f"Using hierarchical loading for DOCX: {log.source}")
                    documents = loader.hierarchical_load(log.source)
                else:
                    # Fallback to regular loading for unsupported types
                    self.logger.info(f"Hierarchical loading not supported for {log.source_type}, using regular loading")
                    documents = loader.load(log.source)
            else:
                # Use regular loading for non-hierarchical requests
                documents = loader.load(log.source)

            # generate checksum if its source type is web_page or confluence
            if log.source_type == SourceType.WEB_PAGE.value or log.source_type == SourceType.CONFLUENCE.value:
                log.checksum = self._calculate_checksum_for_url(log, documents)
                self.index_log_helper.save(log)

            if log.source_type == SourceType.KNOWLEDGE_SNIPPET.value:
                log.checksum = hashlib.sha256(log.source.encode()).hexdigest()
                self.index_log_helper.save(log)

            # Initialize archive_file as None
            archive_file = None
            # calc archive path for file-based documents
            if not (log.source_type == SourceType.WEB_PAGE.value or log.source_type == SourceType.CONFLUENCE.value):
                archive_path = self.config.get_embedding_config()["archive_path"]
                os.makedirs(archive_path, exist_ok=True)
                source_file = Path(log.source)
                archive_file = str(Path(archive_path) / source_file.name).replace('\\', '/')

            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": archive_file if archive_file is not None else log.source,
                    "source_type": log.source_type,
                    "checksum": log.checksum,
                    "trunk_id": get_id(),
                    "is_hierarchical": is_hierarchical
                })

            # Count parent/child documents if hierarchical
            if is_hierarchical:
                parent_count = sum(1 for doc in documents if doc.metadata.get("is_parent", False))
                child_count = sum(1 for doc in documents if not doc.metadata.get("is_parent", True))
                self.logger.info(f"Hierarchical processing: {parent_count} parent docs, {child_count} child docs")

            # Save to vector store in batches
            self.add_documents_in_batches(documents)

            # Save to graph store if enabled and available
            if self.graph_store_enabled:
                self.logger.info(f"Saving to graph store: {log.source_type}:{log.source}")
                try:
                    if is_hierarchical:
                        # Separate parent and child documents
                        parent_docs = [doc for doc in documents if doc.metadata.get("is_parent", False)]
                        child_docs = [doc for doc in documents if not doc.metadata.get("is_parent", True)]
                        
                        # Add with hierarchical structure if supported by graph store
                        if hasattr(self.graph_store_helper, "add_hierarchical_document"):
                            self.graph_store_helper.add_hierarchical_document(
                                doc_id=log.id,
                                metadata={
                                    "source": archive_file if archive_file is not None else log.source,
                                    "source_type": log.source_type,
                                    "checksum": log.checksum,
                                    "is_hierarchical": True,
                                    "processing_type": log.processing_type
                                },
                                parent_chunks=parent_docs,
                                child_chunks=child_docs
                            )
                        else:
                            # Fallback to regular document adding
                            self.graph_store_helper.add_document(
                                doc_id=log.id,
                                metadata={
                                    "source": archive_file if archive_file is not None else log.source,
                                    "source_type": log.source_type,
                                    "checksum": log.checksum,
                                    "is_hierarchical": True,
                                    "processing_type": log.processing_type
                                },
                                chunks=documents
                            )
                    else:
                        # Regular document processing
                        self.graph_store_helper.add_document(
                            doc_id=log.id,
                            metadata={
                                "source": archive_file if archive_file is not None else log.source,
                                "source_type": log.source_type,
                                "checksum": log.checksum,
                                "processing_type": log.processing_type
                            },
                            chunks=documents
                        )
                    self.logger.info(f"Successfully saved to graph store: {log.source_type}:{log.source}")
                except Exception as e:
                    self.logger.error(f"Error saving to graph store: {str(e)}, stack:{traceback.format_exc()}")
                    # Don't fail the whole process if graph store fails
                    # Just log the error and continue
                    raise e
            else:
                self.logger.info("Graph store is disabled or not available")

            # Clear error message on success
            log.error_message = None
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}, stack:{traceback.format_exc()}")
            raise e

    def _calculate_checksum_for_url(self, log, documents) -> str:
        self.logger.info(f"Calculating checksum for URL,{log.source_type}:{log.source}")
        if log.source_type == SourceType.WEB_PAGE.value:
            return hashlib.sha256(documents[0].page_content.encode('utf-8')).hexdigest()
        elif log.source_type == SourceType.CONFLUENCE.value:
            all_docs = []
            for doc in documents:
                doc_info = {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                all_docs.append(doc_info)

            all_docs.sort(key=lambda x: (
                x['metadata'].get('title', ''),
                x['metadata'].get('page_number', 0)
            ))
            combined_content = json.dumps(all_docs, sort_keys=True)
            return hashlib.sha256(combined_content.encode()).hexdigest()
        else:
            return None

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
