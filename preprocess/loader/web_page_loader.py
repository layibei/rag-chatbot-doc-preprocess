from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from preprocess.loader.base_loader import DocumentLoader

class WebPageLoader(DocumentLoader):
    def load(self, url: str) -> List[Document]:
        """Override default load method since web pages don't use file paths"""
        try:
            loader = self.get_loader(url)
            if not loader:
                self.logger.error(f"Failed to create loader for URL: {url}")
                raise ValueError(f"Failed to create loader for URL: {url}")
            
            documents = loader.load()
            if not documents:
                self.logger.error(f"Loaded content is empty for URL: {url}")
                raise ValueError(f"Loaded content is empty for URL: {url}")

            splitter = self.get_splitter(documents)
            if not splitter:
                self.logger.error(f"Failed to create splitter for URL: {url}")
                raise ValueError(f"Failed to create splitter for URL: {url}")

            return splitter.split_documents(documents)
        except Exception as e:
            self.logger.error(f"Failed to load URL: {url}, Error: {str(e)}")
            raise

    def get_loader(self, url: str) -> BaseLoader:
        return WebBaseLoader(url)

    def get_splitter(self, documents: List[Document]) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.get_trunk_size(),
            chunk_overlap=self.get_overlap()
        )

    def is_supported_file_extension(self, file_path: str) -> bool:
        # Web pages don't have file extensions to check
        return True 