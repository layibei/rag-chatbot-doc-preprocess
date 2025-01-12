import logging
import os
from typing import Dict, Type, Optional

from preprocess.loader.base_loader import DocumentLoader
from preprocess.loader.csv_loader import CSVDocLoader
from preprocess.loader.docx_loader import DocxDocLoader
from preprocess.loader.json_loader import JsonDocLoader
from preprocess.loader.pdf_loader import PDFDocLoader
from preprocess.loader.text_loader import TextDocLoader
from preprocess.loader.web_page_loader import WebPageLoader
from preprocess.loader.confluence_loader import ConfluenceLoader
from preprocess.index_log import SourceType

logger = logging.getLogger(__name__)


class DocumentLoaderFactory:
    # Source type to loader mapping
    loader_mapping = {
        SourceType.PDF: PDFDocLoader,
        SourceType.TEXT: TextDocLoader,
        SourceType.CSV: CSVDocLoader,
        SourceType.JSON: JsonDocLoader,
        SourceType.DOCX: DocxDocLoader,
        SourceType.WEB_PAGE: WebPageLoader,
        SourceType.CONFLUENCE: ConfluenceLoader
    }

    @staticmethod
    def get_loader(source_type: str) -> DocumentLoader:
        """Get loader by source type"""
        try:
            source_type_enum = SourceType(source_type)
            loader_class = DocumentLoaderFactory.loader_mapping.get(source_type_enum)
            if not loader_class:
                raise ValueError(f"No loader found for source type: {source_type}")
            return loader_class()
        except ValueError as e:
            raise ValueError(f"Invalid source type: {source_type}")

    @staticmethod
    def infer_source_type(file_extension: str) -> Optional[SourceType]:
        """Infer source type from file extension"""
        extension_to_source_type = {
            'pdf': SourceType.PDF,
            'txt': SourceType.TEXT,
            'csv': SourceType.CSV,
            'json': SourceType.JSON,
            'docx': SourceType.DOCX
        }
        return extension_to_source_type.get(file_extension.lower())
