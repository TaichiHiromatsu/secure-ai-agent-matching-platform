"""
SQLite database configuration and models
"""
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trusted_agent_store.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Submission(Base):
    """Agent submission tracking"""
    __tablename__ = "submissions"
    
    id = Column(String, primary_key=True)
    agent_card_url = Column(String, nullable=False)
    endpoint_url = Column(String, nullable=False)
    status = Column(String, default="precheck_pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    stages = Column(JSON, default=lambda: {})  # Security, Functional, Judge results


class AgentCard(Base):
    """Agent card metadata storage"""
    __tablename__ = "agent_cards"
    
    id = Column(String, primary_key=True)
    submission_id = Column(String, nullable=False)
    card_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
