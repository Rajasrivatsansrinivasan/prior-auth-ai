"""
Database models for Prior Authorization AI Engine
Uses SQLAlchemy with SQLite (dev) or PostgreSQL (prod)
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./prior_auth.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PriorAuthRequest(Base):
    __tablename__ = "prior_auth_requests"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(50), unique=True, index=True)
    
    # Patient info
    patient_id = Column(String(50))
    patient_age = Column(Integer)
    patient_gender = Column(String(10))
    
    # Clinical info
    diagnosis_codes = Column(String(500))   # Comma-separated ICD-10 codes
    cpt_codes = Column(String(200))          # Comma-separated CPT codes
    payer = Column(String(100))
    clinical_summary = Column(Text)
    
    # AI Decision
    decision = Column(String(20))            # APPROVED, DENIED, NEEDS_REVIEW
    confidence_score = Column(Float)
    reasoning = Column(Text)
    policy_citations = Column(JSON)          # List of cited policy sections
    retrieved_policy_ids = Column(JSON)      # List of policy doc IDs used
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time_ms = Column(Integer)
    model_used = Column(String(100))
    

class PolicyDocument(Base):
    __tablename__ = "policy_documents"

    id = Column(Integer, primary_key=True, index=True)
    policy_id = Column(String(100), unique=True, index=True)
    payer = Column(String(100))
    title = Column(String(500))
    content = Column(Text)
    effective_date = Column(String(20))
    cpt_codes_covered = Column(String(500))
    icd_codes_covered = Column(String(500))
    file_path = Column(String(500))
    indexed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(50), index=True)
    action = Column(String(100))
    details = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
