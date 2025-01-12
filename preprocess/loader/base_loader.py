import logging
import os
from abc import ABC, abstractmethod
from typing import List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
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

            splitter = self.get_splitter(document)
            if not splitter:
                self.logger.error(f"Failed to create splitter for file: {file_path}")
                raise ValueError(f"Failed to create splitter for file: {file_path}")

            return splitter.split_documents(document)
        except Exception as e:
            self.logger.error(f"Failed to load document: {file_path}, Error: {str(e)}")
            raise

    @abstractmethod
    def get_loader(self, file_path: str) -> BaseLoader:
        """
        Load the document from the given file path.
        """
        pass

    @abstractmethod
    def get_splitter(self, file_path: str) -> TextSplitter:
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
