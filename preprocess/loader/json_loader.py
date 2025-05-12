import json
import os
from typing import List
from datetime import datetime, timezone

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters.json import RecursiveJsonSplitter

from preprocess.loader.base_loader import DocumentLoader


class JsonDocLoader(DocumentLoader):

    def load(self, file_path: str) -> List[Document]:
        # Input validation
        if not file_path or not os.path.exists(file_path):
            self.logger.error(f"Invalid file path: {file_path}")
            raise ValueError(f"Invalid file path: {file_path}")

        try:
            with open(file_path, 'r') as file:
                document = json.load(file)

            if not document:
                self.logger.error(f"Loaded document content is empty: {file_path}")
                raise ValueError(f"Loaded document content is empty: {file_path}")

            splitter = self.get_splitter()
            if not splitter:
                self.logger.error(f"Failed to create splitter for file: {file_path}")
                raise ValueError(f"Failed to create splitter for file: {file_path}")

            json_chunks = splitter.split_json(document)

            metadata = self.create_metadata(file_path)
            metadatas = [metadata for _ in range(len(json_chunks))]
            return splitter.create_documents(json_chunks, metadatas=metadatas)
        except Exception as e:
            self.logger.error(f"Failed to load document: {file_path}, Error: {str(e)}")
            raise

    def hierarchical_load(self, file_path: str) -> List[Document]:
        """Load and process JSON file with hierarchical structure (parent/child docs)"""
        try:
            self.logger.info(f"Loading JSON with hierarchical approach: {file_path}")
            
            # Load JSON data
            with open(file_path, 'r') as file:
                document = json.load(file)

            if not document:
                self.logger.error(f"Loaded document content is empty: {file_path}")
                raise ValueError(f"Loaded document content is empty: {file_path}")
                
            # Get parent and child splitters
            parent_splitter, child_splitter = self.get_hierarchical_splitters()
            
            # Create parent documents
            parent_json_chunks = parent_splitter.split_json(document)
            metadata = self.create_metadata(file_path)
            parent_metadatas = [metadata for _ in range(len(parent_json_chunks))]
            parent_docs = parent_splitter.create_documents(parent_json_chunks, metadatas=parent_metadatas)
            
            # Create all documents with hierarchical structure
            all_docs = []
            
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
                
                # Create child documents from parent content
                # For JSON, we'll use the text splitter for the stringified content
                # This is a simplification - ideally we'd have a better way to split JSON hierarchically
                parent_content = parent_doc.page_content
                child_texts = child_splitter.split_text(parent_content)
                
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
            self.logger.error(f"Failed to load JSON file hierarchically: {file_path}, Error: {str(e)}")
            raise

    def create_metadata(self, file_path):
        creation_time = os.path.getctime(file_path)
        modification_time = os.path.getmtime(file_path)
        creation_date = datetime.fromtimestamp(creation_time, tz=timezone.utc).strftime('D:%Y%m%d%H%M%S') + 'Z'
        modification_date = datetime.fromtimestamp(modification_time, tz=timezone.utc).strftime(
            'D:%Y%m%d%H%M%S') + 'Z'
        metadata = {"source": file_path,
                    "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file_path": file_path,
                    "creationDate": creation_date,
                    "modDate": modification_date, }
        return metadata

    def get_loader(self, file_path: str) -> BaseLoader:
        pass

    def get_splitter(self):
        return RecursiveJsonSplitter(max_chunk_size=self.get_trunk_size(),min_chunk_size=self.get_overlap())

    def is_supported_file_extension(self, file_path: str) -> bool:
        if None != file_path and file_path.lower().endswith(".json"):
            return True

        return False
