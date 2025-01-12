from datetime import datetime, UTC
from sqlalchemy import String, DateTime, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DistributedLock(Base):
    __tablename__ = "distributed_locks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    lock_key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    instance_name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint('lock_key', name='uix_lock_key'),
    )
