"""
FastAPI Backend for Prior Authorization AI Engine
Endpoints for submitting PA requests and managing policies
"""

import uuid
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.database import get_db, create_tables, PriorAuthRequest, PolicyDocument
from models.schemas import (
    PriorAuthRequestInput,
    PriorAuthResponse,
    PolicyCitation,
    AuthDecision,
    PolicyDocumentInput,
    HealthCheckResponse,
)
from utils.vector_store import PolicyVectorStore
from utils.llm_engine import analyze_prior_auth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App Init ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Prior Authorization AI Engine",
    description="AI-powered prior authorization decision support grounded in payer policy documents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared vector store instance
_vector_store: Optional[PolicyVectorStore] = None


def get_vector_store() -> PolicyVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = PolicyVectorStore()
    return _vector_store


@app.on_event("startup")
async def startup_event():
    create_tables()
    get_vector_store()  # Pre-load
    logger.info("Prior Auth AI Engine started.")


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthCheckResponse)
def health_check(db: Session = Depends(get_db)):
    vs = get_vector_store()
    policy_count = db.query(PolicyDocument).count()
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        faiss_index_loaded=vs.total_chunks > 0,
        llm_loaded=True,
        policy_count=policy_count,
        database_connected=True,
    )


# ─── Core PA Endpoint ─────────────────────────────────────────────────────────

@app.post("/analyze", response_model=PriorAuthResponse)
def analyze_request(request: PriorAuthRequestInput, db: Session = Depends(get_db)):
    """
    Submit a prior authorization request for AI analysis.
    Retrieves relevant policy documents and returns structured decision.
    """
    request_id = f"PA-{uuid.uuid4().hex[:8].upper()}"
    logger.info(f"Processing request {request_id}: CPT={request.cpt_codes}, DX={request.diagnosis_codes}")

    # Build search query from clinical data
    search_query = (
        f"{request.payer} "
        f"CPT {' '.join(request.cpt_codes)} "
        f"ICD {' '.join(request.diagnosis_codes)} "
        f"{request.clinical_summary[:300]}"
    )

    # Retrieve relevant policies
    vs = get_vector_store()
    k = int(os.getenv("MAX_RETRIEVED_DOCS", 5))
    retrieved = vs.search(search_query, k=k)

    if not retrieved:
        logger.warning(f"No policy documents retrieved for {request_id}. Index may be empty.")

    # Run LLM reasoning
    decision_dict, model_used, processing_time_ms = analyze_prior_auth(
        clinical_summary=request.clinical_summary,
        diagnosis_codes=request.diagnosis_codes,
        cpt_codes=request.cpt_codes,
        payer=request.payer,
        retrieved_policies=retrieved,
    )

    # Build citations
    citations = [
        PolicyCitation(
            policy_id=c.get("policy_id", "UNKNOWN"),
            payer=c.get("payer", "Unknown"),
            section=c.get("section", ""),
            relevant_text=c.get("relevant_text", ""),
            supports_decision=c.get("supports_decision", True),
        )
        for c in decision_dict.get("policy_citations", [])
    ]

    # Persist to database
    db_record = PriorAuthRequest(
        request_id=request_id,
        patient_age=request.patient_age,
        patient_gender=request.patient_gender,
        diagnosis_codes=",".join(request.diagnosis_codes),
        cpt_codes=",".join(request.cpt_codes),
        payer=request.payer,
        clinical_summary=request.clinical_summary,
        decision=decision_dict["decision"],
        confidence_score=decision_dict["confidence_score"],
        reasoning=decision_dict["reasoning"],
        policy_citations=[c.dict() for c in citations],
        retrieved_policy_ids=[r["metadata"]["policy_id"] for r in retrieved],
        processing_time_ms=processing_time_ms,
        model_used=model_used,
    )
    db.add(db_record)
    db.commit()

    return PriorAuthResponse(
        request_id=request_id,
        decision=AuthDecision(decision_dict["decision"]),
        confidence_score=decision_dict["confidence_score"],
        reasoning=decision_dict["reasoning"],
        policy_citations=citations,
        recommended_actions=decision_dict.get("recommended_actions", []),
        processing_time_ms=processing_time_ms,
        model_used=model_used,
        timestamp=datetime.utcnow(),
    )


# ─── History Endpoint ─────────────────────────────────────────────────────────

@app.get("/requests")
def list_requests(limit: int = 20, db: Session = Depends(get_db)):
    """List recent prior auth requests."""
    records = (
        db.query(PriorAuthRequest)
        .order_by(PriorAuthRequest.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "request_id": r.request_id,
            "payer": r.payer,
            "cpt_codes": r.cpt_codes,
            "decision": r.decision,
            "confidence_score": r.confidence_score,
            "created_at": r.created_at,
        }
        for r in records
    ]


# ─── Policy Management ────────────────────────────────────────────────────────

@app.get("/policies")
def list_policies(db: Session = Depends(get_db)):
    """List all indexed policy documents."""
    docs = db.query(PolicyDocument).all()
    return [
        {
            "policy_id": d.policy_id,
            "payer": d.payer,
            "title": d.title,
            "indexed": d.indexed,
        }
        for d in docs
    ]


import os

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
