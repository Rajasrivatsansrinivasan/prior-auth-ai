"""
Policy Ingestion Script
Reads policy documents from ./data/policies/ and builds FAISS index.
Run this ONCE before starting the app.

Usage: python scripts/ingest_policies.py
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.vector_store import PolicyVectorStore
from models.database import create_tables, SessionLocal, PolicyDocument

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Map filenames to policy metadata
POLICY_METADATA = {
    "aetna_mri_policy.txt": {
        "policy_id": "AETNA-IMG-MRI-2024",
        "payer": "Aetna",
        "title": "Magnetic Resonance Imaging (MRI) - Prior Authorization Requirements",
    },
    "bcbs_oncology_policy.txt": {
        "policy_id": "BCBS-ONCO-CHEMO-2024",
        "payer": "BlueCross BlueShield",
        "title": "Oncology Drug Prior Authorization - Targeted Therapy & Immunotherapy",
    },
    "uhc_orthopedic_policy.txt": {
        "policy_id": "UHC-SURG-ORTHO-2024",
        "payer": "UnitedHealthcare",
        "title": "Orthopedic Surgery Prior Authorization Guidelines",
    },
    "cigna_behavioral_health_policy.txt": {
        "policy_id": "CIGNA-BH-PSYCH-2024",
        "payer": "Cigna",
        "title": "Behavioral Health - Psychiatric Medication Prior Authorization",
    },
}


def ingest_all_policies():
    logger.info("=== Prior Authorization AI — Policy Ingestion ===")

    # Create DB tables
    create_tables()
    logger.info("Database tables created/verified.")

    # Initialize vector store
    store = PolicyVectorStore()

    policies_dir = Path(__file__).parent.parent / "data" / "policies"
    if not policies_dir.exists():
        logger.error(f"Policies directory not found: {policies_dir}")
        sys.exit(1)

    policy_files = list(policies_dir.glob("*.txt"))
    if not policy_files:
        logger.error("No .txt policy files found in data/policies/")
        sys.exit(1)

    logger.info(f"Found {len(policy_files)} policy files.")

    db = SessionLocal()
    ingested = 0

    for policy_file in policy_files:
        filename = policy_file.name
        if filename not in POLICY_METADATA:
            logger.warning(f"No metadata defined for {filename}, skipping.")
            continue

        meta = POLICY_METADATA[filename]
        content = policy_file.read_text(encoding="utf-8")

        # Add to vector store
        store.add_policy(
            policy_id=meta["policy_id"],
            payer=meta["payer"],
            title=meta["title"],
            content=content,
            file_path=str(policy_file),
        )

        # Save to database
        existing = db.query(PolicyDocument).filter_by(policy_id=meta["policy_id"]).first()
        if existing:
            existing.content = content
            existing.indexed = True
        else:
            db_policy = PolicyDocument(
                policy_id=meta["policy_id"],
                payer=meta["payer"],
                title=meta["title"],
                content=content,
                file_path=str(policy_file),
                indexed=True,
            )
            db.add(db_policy)

        db.commit()
        ingested += 1
        logger.info(f"  ✓ Ingested: {meta['policy_id']} ({meta['payer']})")

    db.close()

    # Save FAISS index
    store.save_index()

    logger.info(f"\n=== Ingestion Complete ===")
    logger.info(f"Policies ingested: {ingested}")
    logger.info(f"Total chunks indexed: {store.total_chunks}")
    logger.info(f"Index saved to: {store.index_path}")
    logger.info("\nYou can now start the app:")
    logger.info("  Streamlit: streamlit run app/streamlit_app.py")
    logger.info("  FastAPI:   uvicorn app.api:app --reload")


if __name__ == "__main__":
    ingest_all_policies()
