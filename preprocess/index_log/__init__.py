from enum import Enum as PyEnum

from sqlalchemy import Column, String, DateTime, BigInteger, Text, UniqueConstraint, Integer
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class SourceType(str, PyEnum):
    CSV = 'csv'
    PDF = 'pdf'
    TEXT = 'text'
    JSON = 'json'
    DOCX = 'docx'
    WEB_PAGE = "web_page"
    CONFLUENCE = "confluence"

    def is_file_based(self) -> bool:
        return self in [SourceType.PDF, SourceType.CSV, SourceType.TEXT, SourceType.JSON, SourceType.DOCX]


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
