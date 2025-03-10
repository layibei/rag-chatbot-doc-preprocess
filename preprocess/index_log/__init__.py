from enum import Enum as PyEnum

from sqlalchemy import Column, String, DateTime, BigInteger, Text, UniqueConstraint, Integer
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class SourceType(PyEnum):
    PDF = "pdf"
    TEXT = 'text'
    KNOWLEDGE_SNIPPET = "knowledge_snippet"
    CSV = "csv"
    JSON = "json"
    DOCX = "docx"
    WEB_PAGE = "web_page"
    CONFLUENCE = "confluence"

    def is_file_based(self) -> bool:
        is_file_upload = self.value not in [SourceType.WEB_PAGE.value, SourceType.CONFLUENCE.value,
                              SourceType.KNOWLEDGE_SNIPPET.value]
        return is_file_upload


class Status(str, PyEnum):
    PENDING = 'PENDING'
    IN_PROGRESS = 'IN_PROGRESS'
    FAILED = 'FAILED'
    COMPLETED = 'COMPLETED'


class IndexLog(Base):
    __tablename__ = 'index_logs'

    id = Column(String(255), primary_key=True)
    source = Column(String(1024), nullable=False)
    source_type = Column(String(128), nullable=False)
    checksum = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False)
    created_by = Column(String(128), nullable=False)
    modified_at = Column(DateTime, nullable=False)
    modified_by = Column(String(128), nullable=False)
    status = Column(String(128), nullable=False)
    error_message = Column(Text)
    retry_count = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint('source', 'source_type', name='uix_source_source_type'),
        UniqueConstraint('checksum', name='uix_checksum'),
    )
