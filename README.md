# 🏥 Prior Authorization Intelligence Engine

> AI-powered clinical decision support that analyzes prior authorization requests against payer policy documents. 100% local — no OpenAI, no paid APIs, no cloud dependency.

---

## What It Does

Prior authorization is one of the most painful bottlenecks in US healthcare. This system:

1. **Ingests** a prior auth request (ICD-10 codes, CPT codes, clinical notes)
2. **Retrieves** relevant payer policy sections from a FAISS vector store using semantic search
3. **Reasons** over the retrieved policies with a local LLM (or rule-based fallback)
4. **Returns** a structured `APPROVED / DENIED / NEEDS_REVIEW` decision with confidence score and policy citations

## Tech Stack (All Free & Local)

| Component | Technology |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector Store | FAISS (CPU) |
| LLM Reasoning | `microsoft/phi-2` via HuggingFace Transformers |
| Fallback Engine | Rule-based keyword analyzer |
| Frontend | Streamlit |
| Backend API | FastAPI |
| Database | SQLite (dev) / PostgreSQL (prod) |
| ORM | SQLAlchemy |

---

## Project Structure

```
prior-auth-ai/
├── app/
│   ├── streamlit_app.py      # Main Streamlit UI (the demo)
│   └── api.py                # FastAPI REST backend
├── data/
│   ├── policies/             # Payer policy .txt documents
│   │   ├── aetna_mri_policy.txt
│   │   ├── bcbs_oncology_policy.txt
│   │   ├── uhc_orthopedic_policy.txt
│   │   └── cigna_behavioral_health_policy.txt
│   ├── faiss_index/          # Built by ingest script (auto-created)
│   └── samples/
│       └── demo_scenarios.py # Three demo clinical cases
├── models/
│   ├── database.py           # SQLAlchemy models
│   └── schemas.py            # Pydantic request/response schemas
├── utils/
│   ├── vector_store.py       # FAISS index manager
│   └── llm_engine.py         # LLM reasoning chain + rule-based fallback
├── scripts/
│   └── ingest_policies.py    # One-time policy ingestion script
├── tests/
│   └── test_core.py          # Pytest test suite
├── requirements.txt
├── .env.example
├── docker-compose.yml
└── Dockerfile
```

---

## Quick Start (Local, No Docker)

### Prerequisites
- Python 3.10+
- ~4GB RAM (for rule-based mode) or ~8GB RAM (for Phi-2 LLM)
- No GPU required (runs on CPU)

### 1. Clone & Install

```bash
git clone <your-repo>
cd prior-auth-ai

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed — defaults work out of the box with SQLite
```

### 3. Ingest Policy Documents (run once)

```bash
python scripts/ingest_policies.py
```

This will:
- Read the 4 policy `.txt` files from `data/policies/`
- Chunk and embed them using `all-MiniLM-L6-v2`
- Build and save a FAISS index to `data/faiss_index/`
- Record policies in SQLite database

Expected output:
```
✓ Ingested: AETNA-IMG-MRI-2024 (Aetna)
✓ Ingested: BCBS-ONCO-CHEMO-2024 (BlueCross BlueShield)
✓ Ingested: UHC-SURG-ORTHO-2024 (UnitedHealthcare)
✓ Ingested: CIGNA-BH-PSYCH-2024 (Cigna)
Total chunks indexed: 87
```

### 4. Run the Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

Open → **http://localhost:8501**

### 5. (Optional) Run the FastAPI Backend

```bash
uvicorn app.api:app --reload --port 8000
```

API docs → **http://localhost:8000/docs**

---

## Demo Scenarios

The app includes three clinical cases in the **Three-Scenario Demo** tab:

| Case | Payer | CPT | Expected |
|---|---|---|---|
| Total Knee Replacement | UnitedHealthcare | 27447 | ✅ APPROVED — K-L Grade 4, failed PT+NSAIDs+injections |
| Pembrolizumab (NSCLC) | BlueCross BlueShield | J9271 | ✅ APPROVED — PD-L1 78%, EGFR/ALK/ROS1 negative |
| Brain MRI for Headaches | Aetna | 70553 | ❌ DENIED — No red flags, no conservative therapy trial |

---

## LLM Configuration

The system works in two modes, automatically selected:

### Mode 1: Local LLM (Better quality)
Set in `.env`:
```
LLM_MODEL_NAME=microsoft/phi-2        # ~2.7B params, needs ~6GB RAM
# or
LLM_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0   # Lighter, ~2GB RAM
```
First run downloads the model (~5GB for Phi-2). Subsequent runs use cache.

### Mode 2: Rule-Based Fallback (Always available)
If the LLM fails to load or parse, the system automatically falls back to a keyword-based analyzer. This is clearly labeled in results (`model: rule-based-fallback`).

To force rule-based mode (faster, no download):
```python
# In llm_engine.py, or pass force_rule_based=True in the API call
```

---

## REST API Usage

```bash
# Submit a prior auth request
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "patient_age": 61,
    "patient_gender": "Female",
    "diagnosis_codes": ["M17.11"],
    "cpt_codes": ["27447"],
    "payer": "UnitedHealthcare",
    "clinical_summary": "61yo female with K-L Grade 4 right knee OA. Failed 14 sessions PT, 7 months NSAIDs, 2 corticosteroid injections. BMI 27. Cardiac clearance obtained."
  }'
```

Response:
```json
{
  "request_id": "PA-A3F2B1C0",
  "decision": "APPROVED",
  "confidence_score": 0.81,
  "reasoning": "Patient meets all UHC criteria for TKA: Grade 4 OA confirmed, failed conservative therapy > 3 months including PT, NSAIDs, and corticosteroid injection, BMI within range, cardiac clearance documented.",
  "policy_citations": [...],
  "recommended_actions": ["Proceed with authorization", "Notify provider"],
  "processing_time_ms": 1823,
  "model_used": "microsoft/phi-2"
}
```

---

## Adding Your Own Policies

1. Add `.txt` policy files to `data/policies/`
2. Add metadata entry to `scripts/ingest_policies.py`'s `POLICY_METADATA` dict
3. Re-run ingestion: `python scripts/ingest_policies.py`

Policy format: Plain text. The system chunks and embeds automatically. No special formatting required.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover vector store, LLM engine (rule-based), database models, and schemas. No external services required.

---

## Docker (Optional)

```bash
# Uses SQLite (no Postgres needed)
docker build -t prior-auth-ai .
docker run -p 8501:8501 -p 8000:8000 prior-auth-ai

# With Postgres
docker-compose up
```

---

## Why This Is Interesting for Healthcare Companies

- **Zero dependency on OpenAI** — runs entirely on-premises, suitable for HIPAA environments
- **Grounded reasoning** — decisions cite specific policy sections, not hallucinated criteria
- **Auditable** — every decision is logged in the database with the retrieved policy chunks
- **Pluggable** — swap any payer's policy documents without code changes
- **Explainable** — confidence scores and policy citations make every decision reviewable

---

## License

MIT License. For educational and research purposes.
This tool is not a substitute for clinical judgment or official payer decisions.
