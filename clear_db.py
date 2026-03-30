"""Clear all data from the database. Run from backend directory with venv activated."""
from app.config import settings
from sqlalchemy import create_engine, text

engine = create_engine(settings.DATABASE_URL)
with engine.connect() as conn:
    conn.execute(text("DELETE FROM innovations"))
    conn.execute(text("DELETE FROM patents"))
    conn.execute(text("DELETE FROM analyses"))
    conn.commit()
    print("All tables cleared.")
