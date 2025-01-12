from datetime import datetime, UTC, timedelta
from typing import Optional, List

from sqlalchemy import and_, or_
from sqlalchemy.sql import select
from sqlalchemy.exc import SQLAlchemyError, DatabaseError

from config.database.repository import BaseRepository
from preprocess.index_log import IndexLog, Status
from utils.id_util import get_id


class IndexLogRepository(BaseRepository[IndexLog]):
    def __init__(self, db_manager):
        super().__init__(db_manager, IndexLog)

    def _get_model_class(self) -> type:
        return IndexLog

    def save(self, index_log: IndexLog) -> IndexLog:
        with self.db_manager.session() as session:
            try:
                if not index_log.id:
                    index_log.id = get_id()
                    session.add(index_log)
                else:
                    index_log = session.merge(index_log)

                session.flush()
                session.refresh(index_log)
                return self._create_detached_copy(index_log)

            except Exception as e:
                session.rollback()
                raise

    def find_by_checksum(self, checksum: str) -> Optional[IndexLog]:
        results = self.find_by_filter(checksum=checksum)
        return results[0] if results else None

    def find_by_source(self, source: str, source_type: str = None) -> Optional[IndexLog]:
        filters = {"source": source}
        if source_type:
            filters["source_type"] = source_type
        result = self.find_by_filter(**filters)
        if not result or len(result) < 1:
            return None
        return self._create_detached_copy(result[0]) if result else None

    def delete_by_source(self, file_path: str) -> None:
        with self.db_manager.session() as session:
            session.query(self.model_class).filter(
                self.model_class.source == file_path
            ).delete()

    def get_pending_index_logs(self) -> List[IndexLog]:
        with self.db_manager.session() as session:
            stmt = session.query(self.model_class).filter(
                or_(
                    self.model_class.status == Status.PENDING,
                    and_(
                        self.model_class.status == Status.FAILED,
                        self.model_class.retry_count <= 3
                    )
                )
            ).with_for_update(skip_locked=True)

            results = stmt.all()
            return [self._create_detached_copy(result) for result in results]

    def find_by_id(self, log_id: str) -> Optional[IndexLog]:
        result = self.find_by_filter(id=log_id)
        if not result or len(result) < 1:
            return None

        return result[0]

    def create(self, source: str, source_type: str, checksum: str, status: Status, user_id: str) -> IndexLog:
        with self.db_manager.session() as session:
            try:
                now = datetime.now(UTC)
                index_log = IndexLog(
                    id=get_id(),
                    source=source,
                    source_type=source_type,
                    checksum=checksum,
                    status=status,
                    created_at=now,
                    created_by=user_id,
                    modified_at=now,
                    modified_by=user_id
                )
                session.add(index_log)
                session.flush()
                session.refresh(index_log)
                return self._create_detached_copy(index_log)
            except Exception as e:
                session.rollback()
                if "UNIQUE constraint failed" in str(e):
                    existing = session.query(self.model_class) \
                        .filter_by(checksum=checksum) \
                        .first()
                    if existing:
                        return self._create_detached_copy(existing)
                raise


    def _create_detached_copy(self, db_obj: Optional[IndexLog]) -> Optional[IndexLog]:
        if not db_obj:
            return None

        return IndexLog(
            id=db_obj.id,
            source=db_obj.source,
            source_type=db_obj.source_type,
            checksum=db_obj.checksum,
            status=db_obj.status,
            created_at=db_obj.created_at,
            created_by=db_obj.created_by,
            modified_at=db_obj.modified_at,
            modified_by=db_obj.modified_by,
            error_message=db_obj.error_message
        )

    def find_all(self, page: int = 1, page_size: int = 10, filters: dict = None) -> List[IndexLog]:
        """
        Find all index logs with pagination and filtering support
        
        Args:
            page: Page number (1-based)
            page_size: Number of items per page
            filters: Dictionary of filter conditions
                Supported filters:
                - source: str (case-insensitive partial match)
                - source_type: SourceType
                - status: str
                - created_by: str (case-insensitive partial match)
                - created_at_from: datetime
                - created_at_to: datetime
        """
        with self.db_manager.session() as session:
            try:
                query = session.query(IndexLog)

                if filters:
                    if filters.get('source'):
                        query = query.filter(IndexLog.source.ilike(f'%{filters["source"]}%'))
                    if filters.get('source_type'):
                        query = query.filter(IndexLog.source_type == filters['source_type'])
                    if filters.get('status'):
                        query = query.filter(IndexLog.status == filters['status'])
                    if filters.get('created_by'):
                        query = query.filter(IndexLog.created_by.ilike(f'%{filters["created_by"]}%'))
                    if filters.get('created_at_from'):
                        query = query.filter(IndexLog.created_at >= filters['created_at_from'])
                    if filters.get('created_at_to'):
                        query = query.filter(IndexLog.created_at <= filters['created_at_to'])

                # Apply pagination
                offset = (page - 1) * page_size
                query = query.offset(offset).limit(page_size)
                results = query.all()

                return [self._create_detached_copy(result) for result in results]

            except SQLAlchemyError as e:
                self.logger.error(f'Error while finding all index logs: {str(e)}')
                raise DatabaseError(f"Database error: {str(e)}")

    def query(self):
        """Create a new query object for the model class."""
        with self.db_manager.session() as session:
            return session.query(self.model_class)
