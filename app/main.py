from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.routes import router
from app.api.websocket import ws_router
from app.database import Base, engine
from app.models.tables import Analysis, Patent, Innovation  # noqa: F401 — registers models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables if they don't exist (safe — won't drop or alter existing tables)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="IdeaPoke Patent Intelligence", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://ideapoke-frnd.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.include_router(ws_router)
