from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

from utils.logger_init import logger

Base = declarative_base()


class DatabaseManager:
    _instance = None

    def __new__(cls, uri: str = None):
        if cls._instance is None and uri:
            cls._instance = super().__new__(cls)
            cls._instance.init_db(uri)
        return cls._instance

    def init_db(self, uri: str):
        self.engine = create_engine(uri)
        self.SessionFactory = sessionmaker(bind=self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception as e:
            logger.error(f"Error: {e}")
            session.rollback()
            raise e
        finally:
            session.close()
