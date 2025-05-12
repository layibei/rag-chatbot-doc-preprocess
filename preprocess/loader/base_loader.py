import logging
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter

from config.common_settings import CommonConfig
from utils.logging_util import logger


class DocumentLoader(ABC):
    """
       Abstract base class for loading and splitting documents.
   """

    def __init__(self):
        self.logger = logger
        self.base_config = CommonConfig()

    def load(self, file_path: str) -> List[Document]:
        # Input validation
        if not file_path or not os.path.exists(file_path):
            self.logger.error(f"Invalid file path: {file_path}")
            raise ValueError(f"Invalid file path: {file_path}")
        try:
            loader = self.get_loader(file_path)
            if not loader:
                self.logger.error(f"Failed to create loader for file: {file_path}")
                raise ValueError(f"Failed to create loader for file: {file_path}")
            document = loader.load()

            if not document:
                self.logger.error(f"Loaded document content is empty: {file_path}")
                raise ValueError(f"Loaded document content is empty: {file_path}")

            splitter = self.get_splitter()
            if not splitter:
                self.logger.error(f"Failed to create splitter for file: {file_path}")
                raise ValueError(f"Failed to create splitter for file: {file_path}")

            return splitter.split_documents(document)
        except Exception as e:
            self.logger.error(f"Failed to load document: {file_path}, Error: {str(e)}")
            raise
    
    def hierarchical_load(self, file_path: str) -> List[Document]:
        """Load and process document with hierarchical structure (parent/child docs)
        
        This is a base implementation that can be overridden by subclasses that need
        specific handling for their document types.
        """
        try:
            self.logger.info(f"Loading with hierarchical approach: {file_path}")
            # Get the loader
            loader = self.get_loader(file_path)
            if not loader:
                raise ValueError(f"Failed to create loader for file: {file_path}")
                
            # Load the document
            documents = loader.load()
            if not documents:
                self.logger.error(f"Loaded document content is empty: {file_path}")
                raise ValueError(f"Loaded document content is empty: {file_path}")
            
            # Get parent and child splitters
            parent_splitter, child_splitter = self.get_hierarchical_splitters()
            
            # Create all documents with hierarchical structure
            all_docs = []
            
            # First create parent documents
            parent_docs = parent_splitter.split_documents(documents)
            self.logger.info(f"Created {len(parent_docs)} parent documents")
            
            # For each parent document, create child documents
            for i, parent_doc in enumerate(parent_docs):
                # Add parent document with metadata
                parent_id = f"parent_{i}"
                parent_doc.metadata.update({
                    "doc_type": "parent",
                    "doc_level": "parent",
                    "parent_id": parent_id,
                    "is_parent": True,
                    "page_number": i,
                    "total_pages": len(parent_docs)
                })
                all_docs.append(parent_doc)
                
                # Create child documents from parent
                child_texts = child_splitter.split_text(parent_doc.page_content)
                self.logger.info(f"Created {len(child_texts)} child documents for parent {i}")
                
                for j, child_text in enumerate(child_texts):
                    child_doc = Document(
                        page_content=child_text,
                        metadata={
                            **parent_doc.metadata.copy(),
                            "doc_type": "child",
                            "doc_level": "child",
                            "parent_id": parent_id,
                            "child_id": f"child_{i}_{j}",
                            "is_parent": False,
                            "child_index": j
                        }
                    )
                    all_docs.append(child_doc)
            
            self.logger.info(f"Total documents created: {len(all_docs)}")
            return all_docs
        except Exception as e:
            self.logger.error(f"Failed to load file hierarchically: {file_path}, Error: {str(e)}")
            raise
            
    def get_hierarchical_splitters(self) -> Tuple[TextSplitter, TextSplitter]:
        """Get parent and child text splitters for hierarchical document structure"""
        parent_chunk_size = 2000  # Default parent chunk size
        parent_overlap = 200      # Default parent overlap
        child_chunk_size = 400    # Default child chunk size  
        child_overlap = 50        # Default child overlap
        
        # Try to get values from config
        embedding_config = self.base_config.get_embedding_config()
        if embedding_config:
            parent_chunk_size = embedding_config.get('hierarchical.parent_chunk_size', 
                               embedding_config.get('parent_chunk_size', parent_chunk_size))
            parent_overlap = embedding_config.get('hierarchical.parent_overlap',
                           embedding_config.get('parent_overlap', parent_overlap))
            child_chunk_size = embedding_config.get('hierarchical.child_chunk_size',
                             embedding_config.get('child_chunk_size', child_chunk_size))
            child_overlap = embedding_config.get('hierarchical.child_overlap',
                         embedding_config.get('child_overlap', child_overlap))
        
        self.logger.info(f"Using hierarchical splitters: parent={parent_chunk_size}/{parent_overlap}, "
                        f"child={child_chunk_size}/{child_overlap}")
        
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap
        )
        
        return parent_splitter, child_splitter

    @abstractmethod
    def get_loader(self, file_path: str) -> BaseLoader:
        """
        Load the document from the given file path.
        """
        pass

    @abstractmethod
    def get_splitter(self) -> TextSplitter:
        """
        Split the loaded document into chunks.
        """
        pass

    @abstractmethod
    def is_supported_file_extension(self, file_path: str) -> bool:
        """
        Check if the given file path is supported by the loader.
        """
        return False

    def get_trunk_size(self):
        embedding_config = self.base_config.get_embedding_config()
        return embedding_config.get('trunk_size', 1024)

    def get_overlap(self):
        embedding_config = self.base_config.get_embedding_config()
        return embedding_config.get('overlap', 0)
