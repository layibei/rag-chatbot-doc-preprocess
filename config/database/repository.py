from typing import TypeVar, Generic, Optional, List
from sqlalchemy.orm import Session

T = TypeVar('T')


class BaseRepository(Generic[T]):
    def __init__(self, db_manager, model_class=None):
        self.db_manager = db_manager
        self.model_class = model_class or self._get_model_class()

    def _get_model_class(self) -> type:
        """Get the model class from the generic type parameter"""
        raise NotImplementedError

    def save(self, entity: T) -> T:
        with self.db_manager.session() as session:
            session.add(entity)
            session.flush()
            return entity

    def update(self, entity: T) -> T:
        with self.db_manager.session() as session:
            merged = session.merge(entity)
            session.flush()
            return merged

    def delete(self, entity: T):
        with self.db_manager.session() as session:
            session.delete(entity)

    def find_by_id(self, id: any) -> Optional[T]:
        with self.db_manager.session() as session:
            return session.query(self.model_class).get(id)

    def find_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        with self.db_manager.session() as session:
            return session.query(self.model_class) \
                .offset(skip) \
                .limit(limit) \
                .all()

    def find_by_filter(self, **filters):
        with self.db_manager.session() as session:
            result = session.query(self.model_class).filter_by(**filters).all()
            return [self._create_detached_copy(item) for item in result]

    def count(self, **filters) -> int:
        with self.db_manager.session() as session:
            return session.query(self.model_class) \
                .filter_by(**filters) \
                .count()

    def delete_by_filter(self, **filters) -> int:
        with self.db_manager.session() as session:
            return session.query(self.model_class) \
                .filter_by(**filters) \
                .delete()
