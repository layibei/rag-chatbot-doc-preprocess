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

            splitter = self.get_splitter(document)
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

    def get_splitter(self, file_path: str):
        return RecursiveJsonSplitter(max_chunk_size=self.get_trunk_size(),min_chunk_size=self.get_overlap())

    def is_supported_file_extension(self, file_path: str) -> bool:
        if None != file_path and file_path.lower().endswith(".json"):
            return True

        return False
