# server.py
import os
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, String, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, Session

# -------------------------------------------------------------------
# 1) CONFIG DB (via variables d'env)
# -------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "fastapi_db")

DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    "?charset=utf8mb4"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=1800)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# -------------------------------------------------------------------
# 2) MODELE SQLAlchemy 
# -------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# -------------------------------------------------------------------
# 3) DEPENDENCY DB
# -------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------------------------------------------
# 4) APP + ROUTES MINIMALES
# -------------------------------------------------------------------
app = FastAPI(title="FastAPI + MySQL (simple)")

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
    return {"status": "ok"}
