"""
Pydantic schemas for Prior Authorization AI Engine API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AuthDecision(str, Enum):
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    PENDING = "PENDING"


class PolicyCitation(BaseModel):
    policy_id: str
    payer: str
    section: str
    relevant_text: str
    supports_decision: bool


class PriorAuthRequestInput(BaseModel):
    patient_age: Optional[int] = Field(None, ge=0, le=130, description="Patient age in years")
    patient_gender: Optional[str] = Field(None, description="Patient gender")
    diagnosis_codes: List[str] = Field(..., description="ICD-10 diagnosis codes")
    cpt_codes: List[str] = Field(..., description="CPT procedure codes being requested")
    payer: str = Field(..., description="Insurance payer name")
    clinical_summary: str = Field(..., min_length=20, description="Clinical notes and patient history")

    class Config:
        json_schema_extra = {
            "example": {
                "patient_age": 52,
                "patient_gender": "Female",
                "diagnosis_codes": ["M17.11", "M17.31"],
                "cpt_codes": ["27447"],
                "payer": "UnitedHealthcare",
                "clinical_summary": "52-year-old female with severe right knee osteoarthritis confirmed on X-ray (Kellgren-Lawrence Grade 4). Has failed 6 months of conservative treatment including 12 sessions of physical therapy, NSAIDs (naproxen 500mg BID x 4 months), and 2 intra-articular corticosteroid injections. Significant functional limitation - unable to climb stairs or walk more than half a block. BMI 28. Cardiologist clearance obtained."
            }
        }


class PriorAuthResponse(BaseModel):
    request_id: str
    decision: AuthDecision
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    policy_citations: List[PolicyCitation]
    recommended_actions: List[str]
    processing_time_ms: int
    model_used: str
    timestamp: datetime


class PolicyDocumentInput(BaseModel):
    policy_id: str
    payer: str
    title: str
    content: str
    effective_date: Optional[str] = None
    cpt_codes_covered: Optional[List[str]] = None
    icd_codes_covered: Optional[List[str]] = None


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    faiss_index_loaded: bool
    llm_loaded: bool
    policy_count: int
    database_connected: bool
