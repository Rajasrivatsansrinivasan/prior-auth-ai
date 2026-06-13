"""
Prior Authorization AI Engine — Streamlit Demo App
Auto-ingests policies on startup so it works on Streamlit Cloud with no manual steps.
"""

import sys
import os
import uuid
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Prior Auth AI Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.vector_store import PolicyVectorStore
from utils.llm_engine import analyze_prior_auth
from models.database import create_tables, SessionLocal, PriorAuthRequest
from data.samples.demo_scenarios import DEMO_SCENARIOS

logging.basicConfig(level=logging.WARNING)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c75 100%);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 2rem;
    border: 1px solid #1e40af33;
}
.hero-title { font-size: 2.2rem; font-weight: 700; color: #f0f9ff; margin: 0; letter-spacing: -0.5px; }
.hero-subtitle { font-size: 1rem; color: #93c5fd; margin: 0.5rem 0 0 0; font-weight: 400; }
.decision-approved {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 2px solid #10b981; border-radius: 12px; padding: 1.5rem; text-align: center;
}
.decision-denied {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 2px solid #ef4444; border-radius: 12px; padding: 1.5rem; text-align: center;
}
.decision-review {
    background: linear-gradient(135deg, #451a03, #78350f);
    border: 2px solid #f59e0b; border-radius: 12px; padding: 1.5rem; text-align: center;
}
.decision-label { font-size: 2rem; font-weight: 700; letter-spacing: 2px; margin: 0; }
.decision-confidence { font-size: 0.9rem; opacity: 0.85; margin-top: 0.25rem; }
.citation-card {
    background: #1e293b; border: 1px solid #334155; border-left: 4px solid #3b82f6;
    border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;
    font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;
}
.citation-policy-id {
    color: #60a5fa; font-weight: 500; font-size: 0.78rem;
    letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 0.25rem;
}
.scenario-badge {
    display: inline-block; background: #1e3a5f; color: #93c5fd;
    font-size: 0.72rem; padding: 0.2rem 0.6rem; border-radius: 20px;
    font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Auto-ingest on startup ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="🔍 Building policy index (one-time setup)...")
def load_vector_store():
    """Load or build FAISS index automatically — works locally and on Streamlit Cloud."""
    vs = PolicyVectorStore()

    # If index is empty, auto-ingest all policy files
    if vs.total_chunks == 0:
        policies_dir = ROOT / "data" / "policies"
        POLICY_META = {
            "aetna_mri_policy.txt":              ("AETNA-IMG-MRI-2024",    "Aetna",                 "MRI Prior Authorization Requirements"),
            "bcbs_oncology_policy.txt":           ("BCBS-ONCO-CHEMO-2024", "BlueCross BlueShield",  "Oncology Drug Prior Authorization"),
            "uhc_orthopedic_policy.txt":          ("UHC-SURG-ORTHO-2024",  "UnitedHealthcare",      "Orthopedic Surgery Prior Authorization"),
            "cigna_behavioral_health_policy.txt": ("CIGNA-BH-PSYCH-2024",  "Cigna",                 "Behavioral Health Prior Authorization"),
        }
        for fname, (pid, payer, title) in POLICY_META.items():
            fpath = policies_dir / fname
            if fpath.exists():
                vs.add_policy(pid, payer, title, fpath.read_text(encoding="utf-8"), str(fpath))

        # Try to persist (works locally, silently skips if read-only on Cloud)
        try:
            vs.save_index()
        except Exception:
            pass

    return vs


@st.cache_resource(show_spinner=False)
def init_db():
    create_tables()
    return True


def run_analysis(vs, scenario_data: dict) -> dict:
    search_query = (
        f"{scenario_data['payer']} CPT {' '.join(scenario_data['cpt_codes'])} "
        f"ICD {' '.join(scenario_data['diagnosis_codes'])} "
        f"{scenario_data['clinical_summary'][:300]}"
    )
    retrieved = vs.search(search_query, k=5)

    decision_dict, model_used, processing_time_ms = analyze_prior_auth(
        clinical_summary=scenario_data["clinical_summary"],
        diagnosis_codes=scenario_data["diagnosis_codes"],
        cpt_codes=scenario_data["cpt_codes"],
        payer=scenario_data["payer"],
        retrieved_policies=retrieved,
        force_rule_based=True,   # Always use rule-based for fast/reliable cloud demo
    )

    request_id = f"PA-{uuid.uuid4().hex[:8].upper()}"

    # Persist to DB (best-effort)
    try:
        db = SessionLocal()
        db.add(PriorAuthRequest(
            request_id=request_id,
            patient_age=scenario_data.get("patient_age"),
            patient_gender=scenario_data.get("patient_gender"),
            diagnosis_codes=",".join(scenario_data["diagnosis_codes"]),
            cpt_codes=",".join(scenario_data["cpt_codes"]),
            payer=scenario_data["payer"],
            clinical_summary=scenario_data["clinical_summary"],
            decision=decision_dict["decision"],
            confidence_score=decision_dict["confidence_score"],
            reasoning=decision_dict["reasoning"],
            policy_citations=decision_dict.get("policy_citations", []),
            retrieved_policy_ids=[r["metadata"]["policy_id"] for r in retrieved],
            processing_time_ms=processing_time_ms,
            model_used=model_used,
        ))
        db.commit()
        db.close()
    except Exception:
        pass

    return {
        "request_id": request_id,
        "decision": decision_dict["decision"],
        "confidence_score": decision_dict["confidence_score"],
        "reasoning": decision_dict["reasoning"],
        "policy_citations": decision_dict.get("policy_citations", []),
        "recommended_actions": decision_dict.get("recommended_actions", []),
        "retrieved_policies": retrieved,
        "processing_time_ms": processing_time_ms,
        "model_used": model_used,
    }


def render_decision_card(result: dict):
    decision = result["decision"]
    conf_pct = int(result["confidence_score"] * 100)
    css_map = {
        "APPROVED":    ("decision-approved", "✅", "#10b981"),
        "DENIED":      ("decision-denied",   "❌", "#ef4444"),
        "NEEDS_REVIEW":("decision-review",   "⚠️", "#f59e0b"),
    }
    css_class, icon, color = css_map.get(decision, css_map["NEEDS_REVIEW"])
    conf_color = "#10b981" if conf_pct >= 75 else ("#f59e0b" if conf_pct >= 55 else "#ef4444")

    st.markdown(f"""
    <div class="{css_class}">
        <p class="decision-label" style="color:{color};">{icon} {decision}</p>
        <p class="decision-confidence" style="color:#e2e8f0;">
            Confidence: {conf_pct}% &nbsp;|&nbsp; ID: {result['request_id']}
        </p>
    </div>
    <div style="margin-top:0.75rem;">
        <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#94a3b8;margin-bottom:4px;">
            <span>Confidence</span><span style="color:{conf_color};font-weight:600;">{conf_pct}%</span>
        </div>
        <div style="background:#1e293b;border-radius:20px;height:8px;overflow:hidden;">
            <div style="background:{conf_color};width:{conf_pct}%;height:100%;border-radius:20px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_result_detail(result: dict):
    st.markdown("#### 📋 Clinical Reasoning")
    st.info(result["reasoning"])

    citations = result.get("policy_citations", [])
    if citations:
        st.markdown(f"#### 📜 Policy Citations ({len(citations)})")
        for c in citations:
            supports = c.get("supports_decision", True)
            badge_color = "#10b981" if supports else "#ef4444"
            badge_text = "SUPPORTS" if supports else "CONFLICTS"
            st.markdown(f"""
            <div class="citation-card">
                <div class="citation-policy-id">{c.get('policy_id','N/A')} — {c.get('payer','N/A')}
                    <span style="background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}44;
                    padding:1px 8px;border-radius:10px;margin-left:8px;font-size:0.7rem;">{badge_text}</span>
                </div>
                <div style="color:#e2e8f0;font-weight:500;margin-bottom:4px;">{c.get('section','')}</div>
                <div style="color:#94a3b8;">{c.get('relevant_text','')[:400]}</div>
            </div>
            """, unsafe_allow_html=True)

    actions = result.get("recommended_actions", [])
    if actions:
        st.markdown("#### ⚡ Recommended Actions")
        for a in actions:
            st.markdown(f"▸ {a}")

    retrieved = result.get("retrieved_policies", [])
    if retrieved:
        with st.expander(f"🔍 Retrieved Policy Chunks ({len(retrieved)})", expanded=False):
            for i, doc in enumerate(retrieved, 1):
                meta = doc["metadata"]
                st.markdown(f"**{i}. {meta['policy_id']}** — {meta['payer']} &nbsp; `score: {doc['score']:.2f}`")
                st.code(doc["text"][:400] + "...", language="text")
                st.divider()

    st.caption(f"⏱ {result['processing_time_ms']}ms &nbsp;|&nbsp; Engine: `{result['model_used']}`")


# ─── Load resources ───────────────────────────────────────────────────────────
init_db()
vs = load_vector_store()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 PA Intelligence Engine")
    st.markdown("---")
    chunks = vs.total_chunks
    policies = len(vs.get_policy_ids())
    if chunks > 0:
        st.success(f"✅ **{policies} policies** indexed\n\n**{chunks} chunks** in vector store")
    else:
        st.error("⚠️ No policies indexed")
    st.markdown("---")
    st.markdown("**Indexed Payers**")
    for p in ["Aetna", "BlueCross BlueShield", "UnitedHealthcare", "Cigna"]:
        st.markdown(f"&nbsp;&nbsp;🏦 {p}")
    st.markdown("---")
    st.markdown("**Engine**")
    st.markdown("🧠 Semantic search: `all-MiniLM-L6-v2`")
    st.markdown("⚙️ Reasoning: Rule-based + policy grounding")
    st.caption("Runs 100% locally. Zero paid APIs.")

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <p class="hero-title">🏥 Prior Authorization Intelligence Engine</p>
    <p class="hero-subtitle">AI-powered clinical decision support · Grounded in payer policy documents · Zero paid APIs</p>
</div>
""", unsafe_allow_html=True)

if vs.total_chunks == 0:
    st.error("Policy index failed to load. Check that `data/policies/*.txt` files exist.")
    st.stop()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Three-Scenario Demo", "✍️ Custom Analysis", "📊 Request History"])

# ═══════════════════════════════════════
# TAB 1 — Three-Scenario Demo
# ═══════════════════════════════════════
with tab1:
    st.markdown("### Side-by-Side Scenario Comparison")
    st.markdown("Three realistic prior authorization cases analyzed against payer policy documents.")
    st.markdown("---")

    if st.button("▶️  Run All Three Scenarios", type="primary", use_container_width=True):
        scenario_keys = list(DEMO_SCENARIOS.keys())
        cols = st.columns(3)
        for col, key in zip(cols, scenario_keys):
            scenario = DEMO_SCENARIOS[key]
            with col:
                with st.spinner(f"Analyzing {scenario['label']}..."):
                    result = run_analysis(vs, scenario)
                st.markdown(f"<div class='scenario-badge'>{scenario['label']}</div>", unsafe_allow_html=True)
                st.markdown(f"**Payer:** {scenario['payer']}")
                st.markdown(f"**CPT:** `{', '.join(scenario['cpt_codes'])}` | **ICD-10:** `{', '.join(scenario['diagnosis_codes'])}`")
                st.markdown("")
                render_decision_card(result)
                st.markdown("")
                render_result_detail(result)
    else:
        cols = st.columns(3)
        for col, (key, scenario) in zip(cols, DEMO_SCENARIOS.items()):
            with col:
                st.markdown(f"<div class='scenario-badge'>{scenario['label']}</div>", unsafe_allow_html=True)
                st.markdown(f"**Payer:** {scenario['payer']}")
                st.markdown(f"**CPT:** `{', '.join(scenario['cpt_codes'])}`")
                st.markdown(f"**ICD-10:** `{', '.join(scenario['diagnosis_codes'])}`")
                st.markdown(f"**Patient:** {scenario['patient_age']}yo {scenario['patient_gender']}")
                st.divider()
                st.caption(scenario["clinical_summary"][:200] + "...")
        st.info("👆 Click **Run All Three Scenarios** to analyze with policy-grounded AI.")

# ═══════════════════════════════════════
# TAB 2 — Custom Analysis
# ═══════════════════════════════════════
with tab2:
    st.markdown("### Custom Prior Authorization Analysis")

    with st.form("custom_pa_form"):
        col1, col2 = st.columns(2)
        with col1:
            payer = st.selectbox("Payer", ["Aetna", "BlueCross BlueShield", "UnitedHealthcare", "Cigna", "Other"])
            diagnosis_input = st.text_input("ICD-10 Codes (comma-separated)", placeholder="e.g. M17.11, M17.31")
            patient_age = st.number_input("Patient Age", min_value=0, max_value=130, value=50)
        with col2:
            cpt_input = st.text_input("CPT Codes (comma-separated)", placeholder="e.g. 27447")
            patient_gender = st.selectbox("Patient Gender", ["Not specified", "Male", "Female", "Other"])

        clinical_summary = st.text_area("Clinical Summary / Notes", height=250,
            placeholder="Paste clinical notes: diagnosis confirmation, prior treatment history, functional status, test results...")

        demo_choice = st.selectbox("Or load a demo scenario:", ["(none)"] + list(DEMO_SCENARIOS.keys()))
        submitted = st.form_submit_button("🔍 Analyze", type="primary", use_container_width=True)

    if submitted:
        if demo_choice != "(none)" and not clinical_summary.strip():
            active = DEMO_SCENARIOS[demo_choice]
        else:
            if not clinical_summary.strip():
                st.error("Please enter a clinical summary.")
                st.stop()
            dx_codes = [c.strip() for c in diagnosis_input.split(",") if c.strip()]
            cpt_codes_list = [c.strip() for c in cpt_input.split(",") if c.strip()]
            if not dx_codes or not cpt_codes_list:
                st.error("Please enter at least one ICD-10 code and one CPT code.")
                st.stop()
            active = {
                "payer": payer, "diagnosis_codes": dx_codes, "cpt_codes": cpt_codes_list,
                "patient_age": patient_age,
                "patient_gender": patient_gender if patient_gender != "Not specified" else None,
                "clinical_summary": clinical_summary,
            }

        with st.spinner("Retrieving policy documents and analyzing..."):
            result = run_analysis(vs, active)

        st.markdown("---")
        st.markdown("### 🧾 Analysis Result")
        render_decision_card(result)
        st.markdown("")
        render_result_detail(result)

# ═══════════════════════════════════════
# TAB 3 — History
# ═══════════════════════════════════════
with tab3:
    st.markdown("### Request History")
    try:
        db = SessionLocal()
        records = db.query(PriorAuthRequest).order_by(PriorAuthRequest.created_at.desc()).limit(50).all()
        db.close()
        if not records:
            st.info("No requests yet. Run the demo or submit a custom analysis.")
        else:
            import pandas as pd
            icons = {"APPROVED": "✅", "DENIED": "❌", "NEEDS_REVIEW": "⚠️"}
            data = [{
                "Request ID": r.request_id,
                "Payer": r.payer or "-",
                "CPT": r.cpt_codes or "-",
                "Decision": f"{icons.get(r.decision,'?')} {r.decision}",
                "Confidence": f"{int((r.confidence_score or 0)*100)}%",
                "Submitted": r.created_at.strftime("%m/%d %H:%M") if r.created_at else "-",
            } for r in records]
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"DB error: {e}")

st.markdown("---")
st.caption("Prior Authorization AI Engine · Semantic search via FAISS · Policy-grounded decisions · Zero paid APIs")