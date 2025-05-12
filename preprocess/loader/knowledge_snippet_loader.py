from typing import List
import json
from langchain_core.document_loaders import BaseLoader as LangchainBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from preprocess.loader.base_loader import DocumentLoader

class KnowledgeSnippetLoader(DocumentLoader):
    def load(self, source: str) -> List[Document]:
        """Load knowledge snippet from JSON source"""
        try:
            # Parse the JSON source
            snippet_data = json.loads(source)
            
            # Create document with metadata
            doc = Document(
                page_content=snippet_data['content'],
                metadata={
                    'title': snippet_data.get('title', 'Untitled Snippet'),
                    'created_at': snippet_data.get('created_at'),
                    'type': 'knowledge_snippet'
                }
            )
            
            return [doc]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid knowledge snippet JSON: {str(e)}")
            
    def hierarchical_load(self, source: str) -> List[Document]:
        """Load knowledge snippet with hierarchical structure (parent/child docs)"""
        try:
            self.logger.info(f"Loading knowledge snippet with hierarchical approach")
            
            # Parse the JSON source
            snippet_data = json.loads(source)
            
            # Create parent document
            parent_doc = Document(
                page_content=snippet_data['content'],
                metadata={
                    'title': snippet_data.get('title', 'Untitled Snippet'),
                    'created_at': snippet_data.get('created_at'),
                    'type': 'knowledge_snippet',
                    'doc_type': 'parent',
                    'doc_level': 'parent',
                    'parent_id': 'parent_0',
                    'is_parent': True,
                    'page_number': 0,
                    'total_pages': 1
                }
            )
            
            # Get parent and child splitters
            _, child_splitter = self.get_hierarchical_splitters()
            
            # Create all documents with hierarchical structure
            all_docs = [parent_doc]  # Add parent document first
            
            # Create child documents from parent content
            child_texts = child_splitter.split_text(parent_doc.page_content)
            self.logger.info(f"Created {len(child_texts)} child documents for knowledge snippet")
            
            for j, child_text in enumerate(child_texts):
                child_doc = Document(
                    page_content=child_text,
                    metadata={
                        **parent_doc.metadata.copy(),
                        'doc_type': 'child',
                        'doc_level': 'child',
                        'parent_id': 'parent_0',
                        'child_id': f'child_0_{j}',
                        'is_parent': False,
                        'child_index': j
                    }
                )
                all_docs.append(child_doc)
            
            self.logger.info(f"Total documents created: {len(all_docs)}")
            return all_docs
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid knowledge snippet JSON: {str(e)}")
            
    def get_loader(self, source: str) -> LangchainBaseLoader:
        # Knowledge snippets don't use traditional file loaders
        return None
        
    def get_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.get_trunk_size(),
            chunk_overlap=self.get_overlap()
        )
        
    def is_supported_file_extension(self, file_path: str) -> bool:
        # Knowledge snippets don't have file extensions
        return True 