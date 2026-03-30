import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.database import Base

@pytest.fixture(scope="session")
def engine():
    return create_engine(settings.DATABASE_URL)

@pytest.fixture(scope="session")
def tables(engine):
    Base.metadata.create_all(engine)
    yield
    # Don't drop tables — they're shared with the running app
    # Use transaction rollback in db_session for test isolation instead

@pytest.fixture
def db_session(engine, tables):
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()
    yield session
    session.close()
    transaction.rollback()
    connection.close()
