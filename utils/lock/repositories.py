from datetime import datetime, UTC
from sqlalchemy.exc import SQLAlchemyError
from config.database.exceptions import DatabaseError
from config.database.repository import BaseRepository
from utils.lock import DistributedLock
from utils.id_util import get_id
from utils.logging_util import logger


class DistributedLockRepository(BaseRepository[DistributedLock]):
    def __init__(self, db_manager):
        super().__init__(db_manager, DistributedLock)
        self.logger = logger

    def _get_model_class(self) -> type:
        return DistributedLock

    def acquire_lock(self, lock_key: str, instance_name: str) -> bool:
        with self.db_manager.session() as session:
            try:
                lock = DistributedLock(
                    id=get_id(),
                    lock_key=lock_key,
                    instance_name=instance_name,
                    created_at=datetime.now(UTC)
                )
                session.add(lock)
                session.flush()
                session.commit()
                return True
            except SQLAlchemyError as e:
                self.logger.error(f"Error acquiring lock: {str(e)}")
                session.rollback()
                return False

    def release_lock(self, lock_key: str, instance_name: str) -> bool:
        with self.db_manager.session() as session:
            try:
                result = session.query(self.model_class).filter(
                    self.model_class.lock_key == lock_key,
                    self.model_class.instance_name == instance_name
                ).delete(synchronize_session=False)
                session.commit()
                return result > 0
            except SQLAlchemyError as e:
                self.logger.error(f"Error releasing lock: {str(e)}")
                session.rollback()
                raise DatabaseError(f"Error releasing lock: {str(e)}")