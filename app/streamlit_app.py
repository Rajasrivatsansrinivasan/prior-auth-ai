"""
Prior Authorization AI Engine — Streamlit Demo App
A clinical decision support tool that analyzes PA requests against payer policy documents.
"""

import sys
import os
import time
import uuid
import logging
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

# ─── Page Config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Prior Auth AI Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Now import project modules
from utils.vector_store import PolicyVectorStore
from utils.llm_engine import analyze_prior_auth
from models.database import create_tables, SessionLocal, PriorAuthRequest
from data.samples.demo_scenarios import DEMO_SCENARIOS

logging.basicConfig(level=logging.WARNING)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c75 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid #1e40af33;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f0f9ff;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1rem;
    color: #93c5fd;
    margin: 0.5rem 0 0 0;
    font-weight: 400;
}

/* Decision cards */
.decision-approved {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    border: 2px solid #10b981;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.decision-denied {
    background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.decision-review {
    background: linear-gradient(135deg, #451a03 0%, #78350f 100%);
    border: 2px solid #f59e0b;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.decision-label {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin: 0;
}
.decision-confidence {
    font-size: 0.9rem;
    opacity: 0.85;
    margin-top: 0.25rem;
}

/* Policy citation cards */
.citation-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #cbd5e1;
    line-height: 1.6;
}
.citation-policy-id {
    color: #60a5fa;
    font-weight: 500;
    font-size: 0.78rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

/* Stat metric boxes */
.metric-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.25rem;
}

/* Scenario tabs */
.scenario-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    font-size: 0.72rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Override Streamlit expander */
.streamlit-expanderHeader {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

/* Scrollable text area */
.stTextArea textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Cached Resources ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading policy vector store...")
def load_vector_store():
    vs = PolicyVectorStore()
    return vs


@st.cache_data(show_spinner=False)
def get_db_stats():
    create_tables()
    db = SessionLocal()
    try:
        total = db.query(PriorAuthRequest).count()
        approved = db.query(PriorAuthRequest).filter_by(decision="APPROVED").count()
        denied = db.query(PriorAuthRequest).filter_by(decision="DENIED").count()
        review = db.query(PriorAuthRequest).filter_by(decision="NEEDS_REVIEW").count()
        return {"total": total, "approved": approved, "denied": denied, "review": review}
    finally:
        db.close()


def run_analysis(vs, scenario_data: dict):
    """Run PA analysis and return structured result."""
    create_tables()

    search_query = (
        f"{scenario_data['payer']} "
        f"CPT {' '.join(scenario_data['cpt_codes'])} "
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
    )

    request_id = f"PA-{uuid.uuid4().hex[:8].upper()}"

    # Persist
    db = SessionLocal()
    try:
        db_record = PriorAuthRequest(
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
        )
        db.add(db_record)
        db.commit()
    finally:
        db.close()

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
    """Render the decision banner."""
    decision = result["decision"]
    conf = result["confidence_score"]
    conf_pct = int(conf * 100)

    if decision == "APPROVED":
        css_class = "decision-approved"
        icon = "✅"
        color = "#10b981"
    elif decision == "DENIED":
        css_class = "decision-denied"
        icon = "❌"
        color = "#ef4444"
    else:
        css_class = "decision-review"
        icon = "⚠️"
        color = "#f59e0b"

    conf_bar_color = "#10b981" if conf >= 0.75 else ("#f59e0b" if conf >= 0.55 else "#ef4444")

    st.markdown(f"""
    <div class="{css_class}">
        <p class="decision-label" style="color: {color};">{icon} {decision}</p>
        <p class="decision-confidence" style="color: #e2e8f0;">Confidence: {conf_pct}% &nbsp;|&nbsp; Request ID: {result['request_id']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bar
    st.markdown(f"""
    <div style="margin-top:0.75rem;">
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#94a3b8; margin-bottom:4px;">
            <span>Confidence Score</span><span style="color:{conf_bar_color}; font-weight:600;">{conf_pct}%</span>
        </div>
        <div style="background:#1e293b; border-radius:20px; height:8px; overflow:hidden;">
            <div style="background:{conf_bar_color}; width:{conf_pct}%; height:100%; border-radius:20px; transition:width 0.5s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_result_detail(result: dict):
    """Render reasoning + citations + actions."""
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
                <div class="citation-policy-id">{c.get('policy_id', 'N/A')} — {c.get('payer', 'N/A')}
                    <span style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}44;
                    padding:1px 8px; border-radius:10px; margin-left:8px; font-size:0.7rem;">{badge_text}</span>
                </div>
                <div style="color:#e2e8f0; font-weight:500; margin-bottom:4px; font-family:Inter,sans-serif;">
                    {c.get('section', '')}
                </div>
                <div style="color:#94a3b8;">{c.get('relevant_text', '')[:400]}{'...' if len(c.get('relevant_text',''))>400 else ''}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("#### 📜 Policy Citations")
        st.caption("No structured citations returned (may be using rule-based fallback)")

    actions = result.get("recommended_actions", [])
    if actions:
        st.markdown("#### ⚡ Recommended Actions")
        for action in actions:
            st.markdown(f"▸ {action}")

    retrieved = result.get("retrieved_policies", [])
    if retrieved:
        with st.expander(f"🔍 View Retrieved Policy Chunks ({len(retrieved)})", expanded=False):
            for i, doc in enumerate(retrieved, 1):
                meta = doc["metadata"]
                score_pct = int(doc["score"] * 100)
                st.markdown(f"**{i}. {meta['policy_id']}** — {meta['payer']} &nbsp; `similarity: {score_pct}%`")
                st.code(doc["text"][:500] + ("..." if len(doc["text"]) > 500 else ""), language="text")
                st.divider()

    st.caption(f"⏱ {result['processing_time_ms']}ms &nbsp;|&nbsp; Model: `{result['model_used']}`")


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 PA Intelligence Engine")
    st.markdown("---")

    # Vector store status
    try:
        vs = load_vector_store()
        chunks = vs.total_chunks
        policies = len(vs.get_policy_ids())
        if chunks > 0:
            st.success(f"✅ Index loaded: **{policies} policies**, **{chunks} chunks**")
        else:
            st.error("⚠️ FAISS index is empty. Run ingestion first.")
            st.code("python scripts/ingest_policies.py", language="bash")
    except Exception as e:
        vs = None
        st.error(f"Vector store error: {e}")

    st.markdown("---")
    st.markdown("**Indexed Payers**")
    payers = ["Aetna", "BlueCross BlueShield", "UnitedHealthcare", "Cigna"]
    for p in payers:
        st.markdown(f"&nbsp;&nbsp;🏦 {p}")

    st.markdown("---")
    st.markdown("**Quick Stats**")
    try:
        stats = get_db_stats()
        c1, c2 = st.columns(2)
        c1.metric("Total", stats["total"])
        c2.metric("Approved", stats["approved"])
        c3, c4 = st.columns(2)
        c3.metric("Denied", stats["denied"])
        c4.metric("Review", stats["review"])
    except Exception:
        st.caption("No requests yet")

    st.markdown("---")
    st.caption("All models run **locally**. No OpenAI or paid APIs required.")


# ─── Main Content ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <p class="hero-title">🏥 Prior Authorization Intelligence Engine</p>
    <p class="hero-subtitle">AI-powered clinical decision support · Grounded in payer policy documents · Zero paid APIs</p>
</div>
""", unsafe_allow_html=True)

if vs is None or vs.total_chunks == 0:
    st.error("""
    **Policy index not loaded.** Run ingestion first:
    ```bash
    python scripts/ingest_policies.py
    ```
    Then restart the Streamlit app.
    """)
    st.stop()

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🎯 Three-Scenario Demo", "✍️ Custom Analysis", "📊 Request History"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — Three-Scenario Demo
# ══════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Side-by-Side Scenario Comparison")
    st.markdown("Three realistic prior authorization cases analyzed simultaneously against payer policy documents.")
    st.markdown("---")

    if st.button("▶️  Run All Three Scenarios", type="primary", use_container_width=True):
        scenario_keys = list(DEMO_SCENARIOS.keys())
        cols = st.columns(3)

        for col, key in zip(cols, scenario_keys):
            scenario = DEMO_SCENARIOS[key]
            with col:
                with st.spinner(f"Analyzing: {scenario['label']}..."):
                    result = run_analysis(vs, scenario)

                st.markdown(f"<div class='scenario-badge'>{scenario['label']}</div>", unsafe_allow_html=True)
                st.markdown(f"**Payer:** {scenario['payer']}")
                st.markdown(f"**CPT:** `{', '.join(scenario['cpt_codes'])}` | **ICD-10:** `{', '.join(scenario['diagnosis_codes'])}`")
                st.markdown("")
                render_decision_card(result)
                st.markdown("")
                render_result_detail(result)

    else:
        # Preview cards
        cols = st.columns(3)
        for col, (key, scenario) in zip(cols, DEMO_SCENARIOS.items()):
            with col:
                st.markdown(f"<div class='scenario-badge'>{scenario['label']}</div>", unsafe_allow_html=True)
                st.markdown(f"**Payer:** {scenario['payer']}")
                st.markdown(f"**CPT:** `{', '.join(scenario['cpt_codes'])}`")
                st.markdown(f"**ICD-10:** `{', '.join(scenario['diagnosis_codes'])}`")
                st.markdown(f"**Patient:** {scenario['patient_age']}yo {scenario['patient_gender']}")
                st.markdown("---")
                st.caption(scenario["clinical_summary"][:200] + "...")
        
        st.info("👆 Click **Run All Three Scenarios** to analyze with AI and see policy-grounded decisions.")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — Custom Analysis
# ══════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Custom Prior Authorization Analysis")
    st.markdown("Enter clinical details below to get an AI-generated policy-grounded decision.")

    with st.form("custom_pa_form"):
        col1, col2 = st.columns(2)
        with col1:
            payer = st.selectbox("Payer", ["Aetna", "BlueCross BlueShield", "UnitedHealthcare", "Cigna", "Other"])
            diagnosis_input = st.text_input("ICD-10 Diagnosis Codes (comma-separated)", placeholder="e.g. M17.11, M17.31")
            patient_age = st.number_input("Patient Age", min_value=0, max_value=130, value=50)
        with col2:
            cpt_input = st.text_input("CPT Codes (comma-separated)", placeholder="e.g. 27447")
            patient_gender = st.selectbox("Patient Gender", ["Not specified", "Male", "Female", "Other"])

        clinical_summary = st.text_area(
            "Clinical Summary / Notes",
            height=250,
            placeholder="Paste clinical notes here. Include: diagnosis confirmation, prior treatment history, functional status, relevant test results, ordering provider info...",
        )

        # Load from demo
        st.markdown("Or load a demo scenario:")
        demo_choice = st.selectbox("Load demo scenario", ["(none)"] + list(DEMO_SCENARIOS.keys()))

        submitted = st.form_submit_button("🔍 Analyze Prior Auth Request", type="primary", use_container_width=True)

    # Handle demo load
    if demo_choice != "(none)" and demo_choice in DEMO_SCENARIOS:
        demo = DEMO_SCENARIOS[demo_choice]
        st.info(f"Demo loaded: **{demo['label']}** — Fill the form above or click Analyze to run it directly.")

    if submitted:
        # Merge demo or form input
        if demo_choice != "(none)" and demo_choice in DEMO_SCENARIOS and not clinical_summary.strip():
            active = DEMO_SCENARIOS[demo_choice]
        else:
            if not clinical_summary.strip():
                st.error("Please enter a clinical summary.")
                st.stop()
            dx_codes = [c.strip() for c in diagnosis_input.split(",") if c.strip()]
            cpt_codes = [c.strip() for c in cpt_input.split(",") if c.strip()]
            if not dx_codes or not cpt_codes:
                st.error("Please enter at least one ICD-10 code and one CPT code.")
                st.stop()
            active = {
                "payer": payer,
                "diagnosis_codes": dx_codes,
                "cpt_codes": cpt_codes,
                "patient_age": patient_age,
                "patient_gender": patient_gender if patient_gender != "Not specified" else None,
                "clinical_summary": clinical_summary,
            }

        with st.spinner("Retrieving policy documents and analyzing request..."):
            result = run_analysis(vs, active)

        st.markdown("---")
        st.markdown("### 🧾 Analysis Result")
        render_decision_card(result)
        st.markdown("")
        render_result_detail(result)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — Request History
# ══════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Prior Authorization Request History")
    
    if st.button("🔄 Refresh"):
        get_db_stats.clear()

    try:
        db = SessionLocal()
        records = db.query(PriorAuthRequest).order_by(PriorAuthRequest.created_at.desc()).limit(50).all()
        db.close()

        if not records:
            st.info("No requests submitted yet. Run the demo or submit a custom analysis.")
        else:
            import pandas as pd
            data = []
            for r in records:
                conf_pct = int(r.confidence_score * 100) if r.confidence_score else 0
                decision_icon = {"APPROVED": "✅", "DENIED": "❌", "NEEDS_REVIEW": "⚠️"}.get(r.decision, "?")
                data.append({
                    "Request ID": r.request_id,
                    "Payer": r.payer or "-",
                    "CPT": r.cpt_codes or "-",
                    "ICD-10": r.diagnosis_codes or "-",
                    "Decision": f"{decision_icon} {r.decision}",
                    "Confidence": f"{conf_pct}%",
                    "Model": r.model_used or "-",
                    "Submitted": r.created_at.strftime("%m/%d %H:%M") if r.created_at else "-",
                })
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Summary
            total = len(records)
            approved = sum(1 for r in records if r.decision == "APPROVED")
            denied = sum(1 for r in records if r.decision == "DENIED")
            review = sum(1 for r in records if r.decision == "NEEDS_REVIEW")
            
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Requests", total)
            m2.metric("Approved", approved, delta=f"{int(approved/total*100)}%" if total else "")
            m3.metric("Denied", denied, delta=f"{int(denied/total*100)}%" if total else "")
            m4.metric("Needs Review", review)

    except Exception as e:
        st.error(f"Database error: {e}")

# Footer
st.markdown("---")
st.caption("Prior Authorization AI Engine · Runs 100% locally · No paid APIs · Built with SentenceTransformers + FAISS + HuggingFace")
