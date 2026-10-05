from collections.abc import Iterator
from functools import lru_cache
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from app.core.config import get_settings
from app.db.models import Base

@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False)

def get_db() -> Iterator[Session]:
    db = SessionLocal(bind=get_engine())
    try:
        yield db
    finally:
        db.close()

def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())
