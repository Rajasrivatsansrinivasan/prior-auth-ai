"""
Prior Authorization AI Engine — Streamlit App
Enhanced UI with metrics, indicators, and policy-grounded decisions
"""

import sys, os, uuid, logging
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
from utils.llm_engine import analyze_prior_auth, CPT_DESCRIPTIONS, ICD_DESCRIPTIONS
from models.database import create_tables, SessionLocal, PriorAuthRequest
from data.samples.demo_scenarios import DEMO_SCENARIOS

logging.basicConfig(level=logging.WARNING)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d2137 40%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, #1e40af22 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.8px;
}
.hero-sub {
    font-size: 0.95rem;
    color: #64748b;
    margin: 0;
}
.hero-badges {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.badge {
    background: #0f172a;
    border: 1px solid #1e293b;
    color: #94a3b8;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    letter-spacing: 0.3px;
}
.badge-blue { border-color: #1e40af44; color: #60a5fa; }
.badge-green { border-color: #06542244; color: #34d399; }

/* Decision cards */
.card-approved {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 1.5px solid #10b981;
    border-radius: 14px; padding: 1.5rem; text-align: center;
    box-shadow: 0 0 20px #10b98122;
}
.card-denied {
    background: linear-gradient(135deg, #3b0764, #500724);
    border: 1.5px solid #e879f9;
    border-radius: 14px; padding: 1.5rem; text-align: center;
    box-shadow: 0 0 20px #e879f922;
}
.card-review {
    background: linear-gradient(135deg, #1c1917, #292524);
    border: 1.5px solid #f59e0b;
    border-radius: 14px; padding: 1.5rem; text-align: center;
    box-shadow: 0 0 20px #f59e0b22;
}
.decision-text { font-size: 1.7rem; font-weight: 800; letter-spacing: 3px; margin: 0; }
.decision-meta { font-size: 0.8rem; color: #94a3b8; margin-top: 0.35rem; }

/* Stat cards */
.stat-row { display: flex; gap: 0.75rem; margin-bottom: 1rem; }
.stat-card {
    flex: 1;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.stat-value { font-size: 1.6rem; font-weight: 700; color: #f8fafc; line-height: 1; }
.stat-label { font-size: 0.68rem; color: #475569; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 0.2rem; }

/* Citation cards */
.citation {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.6rem;
}
.citation-approved { border-left-color: #10b981; }
.citation-denied { border-left-color: #e879f9; }
.citation-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}
.citation-id { font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; color: #3b82f6; font-weight: 500; }
.citation-badge-support {
    background: #05261433; color: #34d399;
    border: 1px solid #10b98133;
    font-size: 0.65rem; font-weight: 600;
    padding: 1px 8px; border-radius: 10px;
}
.citation-badge-conflict {
    background: #2d002033; color: #e879f9;
    border: 1px solid #e879f933;
    font-size: 0.65rem; font-weight: 600;
    padding: 1px 8px; border-radius: 10px;
}
.citation-section { font-size: 0.82rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.3rem; }
.citation-text { font-size: 0.8rem; color: #64748b; line-height: 1.55; }

/* Indicator pills */
.indicator-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.pill-green {
    background: #05261422; color: #34d399;
    border: 1px solid #10b98133;
    font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 12px;
}
.pill-red {
    background: #2d002022; color: #f87171;
    border: 1px solid #ef444433;
    font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 12px;
}

/* Section headers */
.section-head {
    font-size: 0.75rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e293b;
}

/* Scenario tag */
.scenario-tag {
    display: inline-block;
    background: #0f172a;
    color: #60a5fa;
    border: 1px solid #1e40af55;
    font-size: 0.7rem; font-weight: 700;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Timeline step */
.timeline-step {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid #0f172a;
}
.step-num {
    background: #1e293b;
    color: #60a5fa;
    font-size: 0.7rem;
    font-weight: 700;
    width: 22px; height: 22px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.step-text { font-size: 0.82rem; color: #94a3b8; line-height: 1.4; padding-top: 2px; }

/* Confidence bar */
.conf-bar-wrap { margin: 0.75rem 0; }
.conf-bar-top { display: flex; justify-content: space-between; font-size: 0.75rem; color: #475569; margin-bottom: 5px; }
.conf-bar-bg { background: #0f172a; border-radius: 20px; height: 6px; overflow: hidden; border: 1px solid #1e293b; }
.conf-bar-fill { height: 100%; border-radius: 20px; transition: width 0.6s ease; }

/* Override Streamlit */
div[data-testid="stForm"] { background: transparent; border: none; padding: 0; }
.stTextArea textarea { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ─── Resources ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⚙️ Initializing policy vector store...")
def load_vector_store():
    vs = PolicyVectorStore()
    if vs.total_chunks == 0:
        POLICY_META = {
            "aetna_mri_policy.txt":              ("AETNA-IMG-MRI-2024",    "Aetna",                "MRI Prior Authorization Requirements"),
            "bcbs_oncology_policy.txt":           ("BCBS-ONCO-CHEMO-2024", "BlueCross BlueShield", "Oncology Drug Prior Authorization"),
            "uhc_orthopedic_policy.txt":          ("UHC-SURG-ORTHO-2024",  "UnitedHealthcare",     "Orthopedic Surgery Prior Authorization"),
            "cigna_behavioral_health_policy.txt": ("CIGNA-BH-PSYCH-2024",  "Cigna",                "Behavioral Health Prior Authorization"),
        }
        policies_dir = ROOT / "data" / "policies"
        for fname, (pid, payer, title) in POLICY_META.items():
            fpath = policies_dir / fname
            if fpath.exists():
                vs.add_policy(pid, payer, title, fpath.read_text(encoding="utf-8"), str(fpath))
        try:
            vs.save_index()
        except Exception:
            pass
    return vs


@st.cache_resource(show_spinner=False)
def init_db():
    create_tables()
    return True


def run_analysis(vs, data: dict) -> dict:
    query = (f"{data['payer']} CPT {' '.join(data['cpt_codes'])} "
             f"ICD {' '.join(data['diagnosis_codes'])} {data['clinical_summary'][:400]}")
    retrieved = vs.search(query, k=6)

    result, model, ms = analyze_prior_auth(
        clinical_summary=data["clinical_summary"],
        diagnosis_codes=data["diagnosis_codes"],
        cpt_codes=data["cpt_codes"],
        payer=data["payer"],
        retrieved_policies=retrieved,
        force_rule_based=True,
    )

    request_id = f"PA-{uuid.uuid4().hex[:8].upper()}"

    try:
        db = SessionLocal()
        db.add(PriorAuthRequest(
            request_id=request_id,
            patient_age=data.get("patient_age"),
            patient_gender=data.get("patient_gender"),
            diagnosis_codes=",".join(data["diagnosis_codes"]),
            cpt_codes=",".join(data["cpt_codes"]),
            payer=data["payer"],
            clinical_summary=data["clinical_summary"],
            decision=result["decision"],
            confidence_score=result["confidence_score"],
            reasoning=result["reasoning"],
            policy_citations=result.get("policy_citations", []),
            retrieved_policy_ids=[r["metadata"]["policy_id"] for r in retrieved],
            processing_time_ms=ms,
            model_used=model,
        ))
        db.commit()
        db.close()
    except Exception:
        pass

    return {**result, "request_id": request_id, "retrieved_policies": retrieved,
            "processing_time_ms": ms, "model_used": model}


# ─── Render helpers ───────────────────────────────────────────────────────────

def conf_bar(pct: int, color: str):
    return f"""
    <div class="conf-bar-wrap">
        <div class="conf-bar-top"><span>Confidence Score</span>
            <span style="color:{color};font-weight:700;">{pct}%</span></div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div>
        </div>
    </div>"""


def render_decision(result: dict):
    d = result["decision"]
    pct = int(result["confidence_score"] * 100)

    cfg = {
        "APPROVED":     ("card-approved", "✅", "#10b981", "APPROVED"),
        "DENIED":       ("card-denied",   "🚫", "#e879f9", "DENIED"),
        "NEEDS_REVIEW": ("card-review",   "⚠️",  "#f59e0b", "NEEDS REVIEW"),
    }
    css, icon, color, label = cfg.get(d, cfg["NEEDS_REVIEW"])

    st.markdown(f"""
    <div class="{css}">
        <div class="decision-text" style="color:{color};">{icon} {label}</div>
        <div class="decision-meta">Request ID: {result['request_id']}</div>
    </div>
    {conf_bar(pct, color)}
    """, unsafe_allow_html=True)


def render_indicators(result: dict):
    approvals = result.get("matched_approval_indicators", [])
    denials = result.get("matched_denial_indicators", [])
    if approvals or denials:
        st.markdown('<div class="section-head">📊 Clinical Indicators Found</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if approvals:
                st.markdown("**✅ Approval indicators**")
                pills = "".join(f'<span class="pill-green">✓ {kw}</span>' for kw in approvals)
                st.markdown(f'<div class="indicator-row">{pills}</div>', unsafe_allow_html=True)
            else:
                st.markdown("**✅ Approval indicators**")
                st.caption("None found in clinical summary")
        with col2:
            if denials:
                st.markdown("**🚫 Denial indicators**")
                pills = "".join(f'<span class="pill-red">✗ {kw}</span>' for kw in denials)
                st.markdown(f'<div class="indicator-row">{pills}</div>', unsafe_allow_html=True)
            else:
                st.markdown("**🚫 Denial indicators**")
                st.caption("None found")


def render_citations(result: dict):
    citations = result.get("policy_citations", [])
    if not citations:
        return
    d = result["decision"]
    st.markdown(f'<div class="section-head">📜 Policy Citations ({len(citations)})</div>', unsafe_allow_html=True)
    for c in citations:
        supports = c.get("supports_decision", True)
        card_cls = "citation-approved" if supports else "citation-denied"
        badge_cls = "citation-badge-support" if supports else "citation-badge-conflict"
        badge_txt = "SUPPORTS" if supports else "CONFLICTS"
        sim = c.get("similarity_score", 0)
        st.markdown(f"""
        <div class="citation {card_cls}">
            <div class="citation-header">
                <span class="citation-id">{c.get('policy_id','N/A')} · {c.get('payer','')}</span>
                <span class="{badge_cls}">{badge_txt} · {int(sim*100)}% match</span>
            </div>
            <div class="citation-section">{c.get('section','')}</div>
            <div class="citation-text">{c.get('relevant_text','')}</div>
        </div>
        """, unsafe_allow_html=True)


def render_actions(result: dict):
    actions = result.get("recommended_actions", [])
    if not actions:
        return
    st.markdown('<div class="section-head">⚡ Recommended Actions</div>', unsafe_allow_html=True)
    steps_html = "".join(f"""
        <div class="timeline-step">
            <div class="step-num">{i+1}</div>
            <div class="step-text">{a}</div>
        </div>""" for i, a in enumerate(actions))
    st.markdown(steps_html, unsafe_allow_html=True)


def render_full_result(result: dict):
    render_decision(result)
    st.markdown('<div class="section-head">📋 Clinical Reasoning</div>', unsafe_allow_html=True)
    st.info(result["reasoning"])
    render_indicators(result)
    render_citations(result)
    render_actions(result)

    with st.expander(f"🔍 Retrieved Policy Chunks ({len(result.get('retrieved_policies',[]))})", expanded=False):
        for i, doc in enumerate(result.get("retrieved_policies", []), 1):
            m = doc["metadata"]
            st.markdown(f"**{i}. `{m['policy_id']}`** — {m['payer']} &nbsp; `{int(doc['score']*100)}% match`")
            st.code(doc["text"][:400], language="text")
            st.divider()

    st.caption(f"⏱ {result['processing_time_ms']}ms · Engine: `{result['model_used']}`")


# ─── Init ─────────────────────────────────────────────────────────────────────
init_db()
vs = load_vector_store()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 PA Intelligence Engine")
    st.markdown("---")

    chunks = vs.total_chunks
    policies_count = len(vs.get_policy_ids())

    if chunks > 0:
        st.success(f"✅ Index ready")
        col1, col2 = st.columns(2)
        col1.metric("Policies", policies_count)
        col2.metric("Chunks", chunks)
    else:
        st.error("⚠️ Index empty")

    st.markdown("---")
    st.markdown("**📋 Indexed Payers**")
    payer_info = [
        ("Aetna", "MRI / Imaging"),
        ("BlueCross BlueShield", "Oncology"),
        ("UnitedHealthcare", "Orthopedic Surgery"),
        ("Cigna", "Behavioral Health"),
    ]
    for payer, category in payer_info:
        st.markdown(f"🏦 **{payer}**  \n&nbsp;&nbsp;&nbsp;&nbsp;`{category}`")
        st.markdown("")

    st.markdown("---")

    # DB stats
    try:
        db = SessionLocal()
        total = db.query(PriorAuthRequest).count()
        approved = db.query(PriorAuthRequest).filter_by(decision="APPROVED").count()
        denied = db.query(PriorAuthRequest).filter_by(decision="DENIED").count()
        db.close()
        st.markdown("**📊 Session Stats**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("✅", approved)
        c3.metric("🚫", denied)
    except Exception:
        pass

    st.markdown("---")
    st.markdown("**🔧 Engine**")
    st.markdown("`all-MiniLM-L6-v2` embeddings")
    st.markdown("FAISS semantic retrieval")
    st.markdown("Policy-grounded reasoning")
    st.caption("Zero paid APIs · Runs locally")


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🏥 Prior Authorization Intelligence Engine</div>
    <div class="hero-sub">AI-powered clinical decision support grounded in real payer policy documents</div>
    <div class="hero-badges">
        <span class="badge badge-blue">⚡ Semantic Policy Retrieval</span>
        <span class="badge badge-green">✅ Policy-Grounded Decisions</span>
        <span class="badge">🔒 Zero Paid APIs</span>
        <span class="badge">📋 4 Payer Policies Indexed</span>
        <span class="badge">🏥 Aetna · BCBS · UHC · Cigna</span>
    </div>
</div>
""", unsafe_allow_html=True)

if vs.total_chunks == 0:
    st.error("Policy index failed to load. Ensure `data/policies/*.txt` files exist.")
    st.stop()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Three-Scenario Demo",
    "✍️ Custom Analysis",
    "📊 Analytics Dashboard",
    "📚 Policy Explorer",
])


# ════════════════════════════════════════
# TAB 1 — Demo
# ════════════════════════════════════════
with tab1:
    st.markdown("### Side-by-Side Scenario Comparison")
    st.markdown("Three realistic prior authorization cases — orthopedic surgery, oncology, and diagnostic imaging — analyzed against indexed payer policies.")

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.info("**🦴 Case 1:** Total knee replacement after failed conservative treatment")
    with col_info2:
        st.info("**🧬 Case 2:** Pembrolizumab for NSCLC with PD-L1 ≥50%")
    with col_info3:
        st.info("**🧠 Case 3:** Brain MRI for routine headaches — no red flags")

    st.markdown("---")

    if st.button("▶️  Run All Three Scenarios", type="primary", use_container_width=True):
        cols = st.columns(3)
        for col, (key, scenario) in zip(cols, DEMO_SCENARIOS.items()):
            with col:
                with st.spinner(f"Analyzing..."):
                    result = run_analysis(vs, scenario)

                st.markdown(f'<div class="scenario-tag">{scenario["label"]}</div>', unsafe_allow_html=True)
                st.markdown(f"**{scenario['payer']}** · Age {scenario.get('patient_age','?')} {scenario.get('patient_gender','')}")
                st.markdown(f"`CPT: {', '.join(scenario['cpt_codes'])}` · `ICD: {', '.join(scenario['diagnosis_codes'])}`")
                st.markdown("")
                render_full_result(result)
    else:
        cols = st.columns(3)
        for col, (key, scenario) in zip(cols, DEMO_SCENARIOS.items()):
            with col:
                st.markdown(f'<div class="scenario-tag">{scenario["label"]}</div>', unsafe_allow_html=True)
                st.markdown(f"**Payer:** {scenario['payer']}")
                st.markdown(f"**CPT:** `{', '.join(scenario['cpt_codes'])}`")
                st.markdown(f"**ICD-10:** `{', '.join(scenario['diagnosis_codes'])}`")
                st.markdown(f"**Patient:** {scenario['patient_age']}yo {scenario['patient_gender']}")
                st.divider()
                st.caption(scenario["clinical_summary"][:220] + "...")
        st.info("👆 Click **Run All Three Scenarios** to see live policy-grounded analysis")


# ════════════════════════════════════════
# TAB 2 — Custom
# ════════════════════════════════════════
with tab2:
    st.markdown("### Custom Prior Authorization Request")
    st.markdown("Enter any clinical scenario to get a policy-grounded AI decision.")

    with st.form("pa_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            payer = st.selectbox("Payer *", ["Aetna", "BlueCross BlueShield", "UnitedHealthcare", "Cigna", "Other"])
        with c2:
            diagnosis_input = st.text_input("ICD-10 Codes *", placeholder="M17.11, M17.31")
        with c3:
            cpt_input = st.text_input("CPT Codes *", placeholder="27447")

        c4, c5 = st.columns(2)
        with c4:
            patient_age = st.number_input("Patient Age", 0, 130, 55)
        with c5:
            patient_gender = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"])

        clinical_summary = st.text_area("Clinical Notes *", height=220,
            placeholder="Include: confirmed diagnosis, prior treatment history (PT sessions, medications, injections), functional status, relevant test results (imaging grade, biomarkers), ordering provider specialty...")

        st.markdown("**Or load a demo scenario:**")
        demo_choice = st.selectbox("Quick load", ["(none)"] + list(DEMO_SCENARIOS.keys()))
        submitted = st.form_submit_button("🔍 Analyze Prior Auth Request", type="primary", use_container_width=True)

    if submitted:
        if demo_choice != "(none)" and not clinical_summary.strip():
            active = DEMO_SCENARIOS[demo_choice]
        else:
            if not clinical_summary.strip():
                st.error("Please enter clinical notes.")
                st.stop()
            dx = [c.strip() for c in diagnosis_input.split(",") if c.strip()]
            cpts = [c.strip() for c in cpt_input.split(",") if c.strip()]
            if not dx or not cpts:
                st.error("Please enter at least one ICD-10 and one CPT code.")
                st.stop()
            active = {
                "payer": payer, "diagnosis_codes": dx, "cpt_codes": cpts,
                "patient_age": patient_age,
                "patient_gender": patient_gender if patient_gender != "Not specified" else None,
                "clinical_summary": clinical_summary,
            }

        with st.spinner("🔍 Retrieving policy documents and reasoning..."):
            result = run_analysis(vs, active)

        st.markdown("---")
        st.markdown("### 📋 Authorization Decision")
        render_full_result(result)


# ════════════════════════════════════════
# TAB 3 — Analytics
# ════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Analytics Dashboard")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

    try:
        import pandas as pd
        db = SessionLocal()
        records = db.query(PriorAuthRequest).order_by(PriorAuthRequest.created_at.desc()).limit(200).all()
        db.close()

        if not records:
            st.info("No requests yet. Run the demo or submit a custom analysis to see analytics.")
        else:
            total = len(records)
            approved = sum(1 for r in records if r.decision == "APPROVED")
            denied = sum(1 for r in records if r.decision == "DENIED")
            review = sum(1 for r in records if r.decision == "NEEDS_REVIEW")
            avg_conf = sum(r.confidence_score or 0 for r in records) / total

            # Top metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Requests", total)
            m2.metric("✅ Approved", approved, f"{int(approved/total*100)}%")
            m3.metric("🚫 Denied", denied, f"{int(denied/total*100)}%")
            m4.metric("⚠️ Needs Review", review, f"{int(review/total*100)}%")
            m5.metric("Avg Confidence", f"{int(avg_conf*100)}%")

            st.markdown("---")

            # Charts
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Decision Distribution**")
                dist_df = pd.DataFrame({
                    "Decision": ["Approved", "Denied", "Needs Review"],
                    "Count": [approved, denied, review]
                })
                st.bar_chart(dist_df.set_index("Decision"))

            with col_b:
                st.markdown("**Requests by Payer**")
                payer_counts = {}
                for r in records:
                    p = r.payer or "Unknown"
                    payer_counts[p] = payer_counts.get(p, 0) + 1
                payer_df = pd.DataFrame(list(payer_counts.items()), columns=["Payer", "Count"])
                st.bar_chart(payer_df.set_index("Payer"))

            st.markdown("---")
            st.markdown("**Recent Requests**")
            icons = {"APPROVED": "✅", "DENIED": "🚫", "NEEDS_REVIEW": "⚠️"}
            table_data = [{
                "ID": r.request_id,
                "Payer": r.payer or "-",
                "CPT": r.cpt_codes or "-",
                "ICD-10": r.diagnosis_codes or "-",
                "Decision": f"{icons.get(r.decision,'?')} {r.decision}",
                "Confidence": f"{int((r.confidence_score or 0)*100)}%",
                "Time": r.created_at.strftime("%m/%d %H:%M") if r.created_at else "-",
            } for r in records[:50]]
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading analytics: {e}")


# ════════════════════════════════════════
# TAB 4 — Policy Explorer
# ════════════════════════════════════════
with tab4:
    st.markdown("### 📚 Policy Explorer")
    st.markdown("Browse and search the indexed payer policy documents.")

    search_query = st.text_input("🔍 Search policies", placeholder="e.g. physical therapy MRI lumbar spine")

    policies_dir = ROOT / "data" / "policies"
    POLICY_META = {
        "aetna_mri_policy.txt":              ("AETNA-IMG-MRI-2024",    "Aetna",                "MRI Prior Authorization", "🏥"),
        "bcbs_oncology_policy.txt":           ("BCBS-ONCO-CHEMO-2024", "BlueCross BlueShield", "Oncology Drug PA",        "💊"),
        "uhc_orthopedic_policy.txt":          ("UHC-SURG-ORTHO-2024",  "UnitedHealthcare",     "Orthopedic Surgery PA",   "🦴"),
        "cigna_behavioral_health_policy.txt": ("CIGNA-BH-PSYCH-2024",  "Cigna",                "Behavioral Health PA",    "🧠"),
    }

    if search_query.strip():
        st.markdown(f"**Search results for:** `{search_query}`")
        results = vs.search(search_query, k=8)
        if results:
            for r in results:
                m = r["metadata"]
                score_pct = int(r["score"] * 100)
                color = "#10b981" if score_pct > 70 else ("#f59e0b" if score_pct > 50 else "#64748b")
                st.markdown(f"""
                <div class="citation" style="border-left-color:{color};">
                    <div class="citation-header">
                        <span class="citation-id">{m['policy_id']} · {m['payer']}</span>
                        <span style="color:{color};font-size:0.75rem;font-weight:700;">{score_pct}% match</span>
                    </div>
                    <div class="citation-section">{m['title']}</div>
                    <div class="citation-text">{r['text'][:350]}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No matching policy sections found.")
    else:
        for fname, (pid, payer, title, icon) in POLICY_META.items():
            fpath = policies_dir / fname
            with st.expander(f"{icon} **{payer}** — {title} `{pid}`"):
                if fpath.exists():
                    content = fpath.read_text(encoding="utf-8")
                    st.code(content, language="text")
                else:
                    st.warning("Policy file not found.")

st.markdown("---")
st.caption("Prior Authorization AI Engine · FAISS Semantic Search · Policy-Grounded Decisions · Zero Paid APIs")