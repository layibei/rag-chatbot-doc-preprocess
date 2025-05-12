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

            splitter = self.get_splitter()
            if not splitter:
                self.logger.error(f"Failed to create splitter for URL: {url}")
                raise ValueError(f"Failed to create splitter for URL: {url}")

            return splitter.split_documents(documents)
        except Exception as e:
            self.logger.error(f"Failed to load URL: {url}, Error: {str(e)}")
            raise
            
    def hierarchical_load(self, url: str) -> List[Document]:
        """Load and process web page with hierarchical structure (parent/child docs)"""
        try:
            self.logger.info(f"Loading web page with hierarchical approach: {url}")
            # Get the loader
            loader = self.get_loader(url)
            if not loader:
                raise ValueError(f"Failed to create loader for URL: {url}")
                
            # Load the document
            documents = loader.load()
            if not documents:
                self.logger.error(f"Loaded content is empty for URL: {url}")
                raise ValueError(f"Loaded content is empty for URL: {url}")
            
            # Add source URL to metadata
            for doc in documents:
                doc.metadata["source_url"] = url
            
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
                    "total_pages": len(parent_docs),
                    "source_url": url
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
            self.logger.error(f"Failed to load web page hierarchically: {url}, Error: {str(e)}")
            raise

    def get_loader(self, url: str) -> BaseLoader:
        return WebBaseLoader(url)

    def get_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.get_trunk_size(),
            chunk_overlap=self.get_overlap()
        )

    def is_supported_file_extension(self, file_path: str) -> bool:
        # Web pages don't have file extensions to check
        return True 