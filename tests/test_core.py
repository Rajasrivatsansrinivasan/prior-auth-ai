"""
Tests for Prior Authorization AI Engine
Run: pytest tests/ -v
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Vector Store Tests ───────────────────────────────────────────────────────

class TestPolicyVectorStore:
    """Tests for FAISS vector store."""

    def test_init(self, tmp_path):
        from utils.vector_store import PolicyVectorStore
        vs = PolicyVectorStore(index_path=str(tmp_path / "test_index"))
        assert vs is not None
        assert vs.total_chunks == 0

    def test_add_and_search(self, tmp_path):
        from utils.vector_store import PolicyVectorStore
        vs = PolicyVectorStore(index_path=str(tmp_path / "test_index"))

        policy_content = """
        LUMBAR SPINE MRI approval criteria:
        - Low back pain persisting for 6 weeks despite conservative treatment
        - Physical therapy minimum 6 sessions required
        - NSAIDs at therapeutic doses
        Denial: Acute low back pain less than 6 weeks without red flags.
        """
        vs.add_policy(
            policy_id="TEST-MRI-001",
            payer="TestPayer",
            title="MRI Policy Test",
            content=policy_content,
        )
        assert vs.total_chunks > 0

        results = vs.search("MRI lumbar spine physical therapy", k=3)
        assert len(results) > 0
        assert results[0]["score"] > 0.0
        assert results[0]["metadata"]["policy_id"] == "TEST-MRI-001"

    def test_save_and_load(self, tmp_path):
        from utils.vector_store import PolicyVectorStore
        index_path = str(tmp_path / "persist_index")

        vs1 = PolicyVectorStore(index_path=index_path)
        vs1.add_policy("PERSIST-001", "Payer", "Title", "Content about knee surgery arthroplasty.")
        vs1.save_index()
        chunks_before = vs1.total_chunks

        vs2 = PolicyVectorStore(index_path=index_path)
        assert vs2.total_chunks == chunks_before

    def test_empty_search_returns_empty(self, tmp_path):
        from utils.vector_store import PolicyVectorStore
        vs = PolicyVectorStore(index_path=str(tmp_path / "empty_index"))
        results = vs.search("anything")
        assert results == []

    def test_get_policy_ids(self, tmp_path):
        from utils.vector_store import PolicyVectorStore
        vs = PolicyVectorStore(index_path=str(tmp_path / "ids_index"))
        vs.add_policy("ID-001", "Payer1", "T1", "Some policy content about MRI.")
        vs.add_policy("ID-002", "Payer2", "T2", "Another policy about surgery.")
        ids = vs.get_policy_ids()
        assert "ID-001" in ids
        assert "ID-002" in ids


# ─── LLM Engine Tests ─────────────────────────────────────────────────────────

class TestLLMEngine:
    """Tests for the reasoning engine (uses rule-based fallback)."""

    MOCK_POLICIES = [
        {
            "text": "APPROVED: Low back pain persisting 6+ weeks with failed conservative treatment including PT and NSAIDs.",
            "metadata": {"policy_id": "TEST-001", "payer": "TestPayer", "title": "MRI Policy"},
            "score": 0.88,
        }
    ]

    def test_rule_based_approval(self):
        from utils.llm_engine import rule_based_decision
        result = rule_based_decision(
            clinical_summary="Patient has severe chronic back pain, failed conservative treatment with 8 sessions of physical therapy and 3 months of NSAIDs. Documented functional limitation. Spine surgeon recommends MRI.",
            diagnosis_codes=["M54.5"],
            cpt_codes=["72148"],
            retrieved_policies=self.MOCK_POLICIES,
        )
        assert result["decision"] in ("APPROVED", "NEEDS_REVIEW", "DENIED")
        assert 0.0 <= result["confidence_score"] <= 1.0
        assert len(result["reasoning"]) > 0

    def test_rule_based_denial(self):
        from utils.llm_engine import rule_based_decision
        result = rule_based_decision(
            clinical_summary="Patient presents with acute low back pain for 3 days. No prior treatment. First visit. Screening.",
            diagnosis_codes=["M54.5"],
            cpt_codes=["72148"],
            retrieved_policies=self.MOCK_POLICIES,
        )
        assert result["decision"] in ("DENIED", "NEEDS_REVIEW")

    def test_analyze_prior_auth_force_rule_based(self):
        from utils.llm_engine import analyze_prior_auth
        result, model_used, ms = analyze_prior_auth(
            clinical_summary="Patient has chronic knee pain with failed PT and NSAID therapy for 4 months. Documented severe osteoarthritis.",
            diagnosis_codes=["M17.11"],
            cpt_codes=["27447"],
            payer="UnitedHealthcare",
            retrieved_policies=self.MOCK_POLICIES,
            force_rule_based=True,
        )
        assert result["decision"] in ("APPROVED", "DENIED", "NEEDS_REVIEW")
        assert 0.0 <= result["confidence_score"] <= 1.0
        assert model_used == "rule-based-fallback"
        assert ms >= 0

    def test_parse_llm_json_valid(self):
        from utils.llm_engine import parse_llm_json
        raw = '{"decision": "APPROVED", "confidence_score": 0.87, "reasoning": "Meets all criteria.", "policy_citations": [], "recommended_actions": []}'
        result = parse_llm_json(raw)
        assert result is not None
        assert result["decision"] == "APPROVED"

    def test_parse_llm_json_with_preamble(self):
        from utils.llm_engine import parse_llm_json
        raw = 'Here is my analysis:\n\n{"decision": "DENIED", "confidence_score": 0.72, "reasoning": "Does not meet criteria.", "policy_citations": []}'
        result = parse_llm_json(raw)
        assert result is not None
        assert result["decision"] == "DENIED"

    def test_parse_llm_json_invalid(self):
        from utils.llm_engine import parse_llm_json
        result = parse_llm_json("This is not JSON at all.")
        assert result is None


# ─── Database Tests ───────────────────────────────────────────────────────────

class TestDatabase:
    """Tests for database models."""

    def test_create_tables(self):
        """Tables should create without error."""
        import os
        os.environ["DATABASE_URL"] = "sqlite:///./test_prior_auth.db"
        from models.database import create_tables, engine
        create_tables()

        # Cleanup
        import os as _os
        if _os.path.exists("./test_prior_auth.db"):
            _os.remove("./test_prior_auth.db")

    def test_save_and_retrieve_request(self):
        import os
        os.environ["DATABASE_URL"] = "sqlite:///./test_pa_2.db"

        from models.database import create_tables, SessionLocal, PriorAuthRequest
        create_tables()
        db = SessionLocal()

        record = PriorAuthRequest(
            request_id="TEST-PA-001",
            payer="TestPayer",
            diagnosis_codes="M17.11",
            cpt_codes="27447",
            clinical_summary="Test clinical summary",
            decision="APPROVED",
            confidence_score=0.85,
            reasoning="Test reasoning",
            policy_citations=[],
            retrieved_policy_ids=[],
            processing_time_ms=250,
            model_used="test-model",
        )
        db.add(record)
        db.commit()

        fetched = db.query(PriorAuthRequest).filter_by(request_id="TEST-PA-001").first()
        assert fetched is not None
        assert fetched.decision == "APPROVED"
        assert fetched.confidence_score == 0.85

        db.close()
        import os as _os
        if _os.path.exists("./test_pa_2.db"):
            _os.remove("./test_pa_2.db")


# ─── Schema Tests ─────────────────────────────────────────────────────────────

class TestSchemas:
    def test_prior_auth_request_input_valid(self):
        from models.schemas import PriorAuthRequestInput
        req = PriorAuthRequestInput(
            patient_age=52,
            patient_gender="Female",
            diagnosis_codes=["M17.11"],
            cpt_codes=["27447"],
            payer="UnitedHealthcare",
            clinical_summary="Patient with severe osteoarthritis failed conservative treatment for 6 months.",
        )
        assert req.payer == "UnitedHealthcare"
        assert "M17.11" in req.diagnosis_codes

    def test_prior_auth_request_short_summary_rejected(self):
        from models.schemas import PriorAuthRequestInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PriorAuthRequestInput(
                diagnosis_codes=["M17.11"],
                cpt_codes=["27447"],
                payer="TestPayer",
                clinical_summary="Too short",  # Under min_length=20
            )

    def test_auth_decision_enum(self):
        from models.schemas import AuthDecision
        assert AuthDecision.APPROVED == "APPROVED"
        assert AuthDecision.DENIED == "DENIED"
        assert AuthDecision.NEEDS_REVIEW == "NEEDS_REVIEW"
