import re, time, logging
from typing import List, Dict, Tuple
logger = logging.getLogger(__name__)

CPT_DESCRIPTIONS = {
    "27447": "Total Knee Arthroplasty",
    "27130": "Total Hip Arthroplasty",
    "29827": "Rotator Cuff Repair",
    "72148": "Lumbar Spine MRI",
    "70551": "Brain MRI without contrast",
    "70552": "Brain MRI with contrast",
    "70553": "Brain MRI with/without contrast",
    "73721": "Knee MRI",
    "J9271": "Pembrolizumab (Keytruda)",
    "J9355": "Trastuzumab (Herceptin)",
    "J0401": "Aripiprazole (Abilify)",
}

ICD_DESCRIPTIONS = {
    "M17.11": "Primary osteoarthritis right knee",
    "M17.12": "Primary osteoarthritis left knee",
    "M17.31": "Post-traumatic osteoarthritis right knee",
    "C34.12": "NSCLC upper lobe left lung",
    "C34.11": "NSCLC upper lobe right lung",
    "G43.909": "Migraine unspecified",
    "M54.5": "Low back pain",
    "F20.9": "Schizophrenia unspecified",
    "F32.9": "Major depressive disorder",
}

STRONG_APPROVAL = [
    "failed conservative treatment", "conservative treatment failure",
    "physical therapy completed", "physical therapy sessions",
    "failed physical therapy", "nsaid", "corticosteroid injection",
    "kellgren-lawrence grade 3", "kellgren-lawrence grade 4",
    "grade 4", "grade 3", "bone-on-bone", "severe osteoarthritis",
    "full thickness tear", "full-thickness tear",
    "pd-l1", "her2 positive", "her2-positive", "egfr negative",
    "alk negative", "biomarker testing", "pathology confirmed",
    "failed 2 antidepressants", "treatment resistant",
    "functional limitation", "unable to perform", "cannot walk",
    "cardiac clearance", "pre-operative", "specialist referral",
    "6 weeks", "3 months", "6 months", "12 sessions", "8 sessions",
    "failed prior", "inadequate response", "refractory",
]

STRONG_DENIAL = [
    "acute onset", "first visit", "first episode",
    "no prior treatment", "no conservative treatment",
    "no physical therapy", "no documentation",
    "less than 2 weeks", "less than 4 weeks",
    "routine screening", "without symptoms",
    "normal neurological exam", "no red flags",
    "no neurological deficit", "responsive to otc",
    "ibuprofen relieved", "advil relieved",
    "completely normal", "no red flag",
    "no formal", "no neurology consultation",
    "no trial of preventive", "no awakening from sleep",
    "not positional", "no prior workup",
    "tried advil", "tried ibuprofen",
    "worried about", "no family history",
    "no focal neurological", "normal fundoscopic",
]


def _extract_relevant_citations(retrieved_policies, decision):
    citations = []
    seen = set()
    approval_kw = ["approved", "approval criteria", "criteria met", "indication", "covered"]
    denial_kw = ["denied", "denial", "not covered", "excluded", "not indicated"]
    for doc in retrieved_policies[:4]:
        pid = doc["metadata"]["policy_id"]
        if pid in seen:
            continue
        seen.add(pid)
        text = doc["text"]
        sentences = [s.strip() for s in re.split(r"[.\n]", text) if len(s.strip()) > 30]
        best, best_score = "", -1
        kws = approval_kw if decision == "APPROVED" else denial_kw
        for sent in sentences:
            sl = sent.lower()
            score = sum(1 for kw in kws if kw in sl)
            if "approved:" in sl:
                score += 3
            if "denied:" in sl:
                score += 3
            if score > best_score:
                best_score = score
                best = sent
        if not best and sentences:
            best = sentences[0]
        best = re.sub(r"={3,}", "", best).strip()
        if len(best) > 20:
            citations.append({
                "policy_id": pid,
                "payer": doc["metadata"]["payer"],
                "section": doc["metadata"]["title"],
                "relevant_text": best[:300],
                "supports_decision": decision in ("APPROVED", "NEEDS_REVIEW"),
                "similarity_score": round(doc["score"], 3),
            })
    return citations


def rule_based_decision(clinical_summary, diagnosis_codes, cpt_codes, retrieved_policies, payer=""):
    sl = clinical_summary.lower()
    approval_count = sum(1 for kw in STRONG_APPROVAL if kw in sl)
    denial_count = sum(1 for kw in STRONG_DENIAL if kw in sl)
    matched_approval = [kw for kw in STRONG_APPROVAL if kw in sl]
    matched_denial = [kw for kw in STRONG_DENIAL if kw in sl]

    if denial_count >= 2 and approval_count < 2:
        decision, confidence = "DENIED", min(0.55 + denial_count * 0.06, 0.88)
    elif approval_count >= 4:
        decision, confidence = "APPROVED", min(0.60 + approval_count * 0.04, 0.92)
    elif approval_count >= 2 and denial_count == 0:
        decision, confidence = "APPROVED", min(0.58 + approval_count * 0.04, 0.82)
    elif approval_count >= 1 and denial_count == 0:
        decision, confidence = "NEEDS_REVIEW", 0.55
    else:
        decision, confidence = "NEEDS_REVIEW", 0.48

    cpt_desc = ", ".join(CPT_DESCRIPTIONS.get(c, c) for c in cpt_codes)
    icd_desc = ", ".join(ICD_DESCRIPTIONS.get(c, c) for c in diagnosis_codes)

    if decision == "APPROVED":
        reasoning = (
            f"Request for {cpt_desc} for {icd_desc} meets policy approval criteria. "
            f"Clinical documentation demonstrates: {', '.join(matched_approval[:4])}."
        )
        actions = [
            "Issue authorization approval notification to provider",
            "Set authorization valid period per payer policy",
            "Document approval in member record",
        ]
    elif decision == "DENIED":
        reasoning = (
            f"Request for {cpt_desc} for {icd_desc} does not meet policy criteria. "
            f"Denial indicators found: {', '.join(matched_denial[:4])}. "
            f"Provider may appeal within 60 days with additional clinical documentation."
        )
        actions = [
            "Send denial notice with specific policy citation to provider",
            "Include appeal rights and 60-day appeal window in notice",
            "Offer peer-to-peer review within 72 hours",
            "Document denial rationale in member record",
        ]
    else:
        reasoning = (
            f"Request for {cpt_desc} for {icd_desc} requires additional clinical review. "
            f"{'Partial approval criteria found: ' + ', '.join(matched_approval[:2]) + '.' if matched_approval else 'Insufficient clinical information provided.'} "
            f"Additional documentation needed to make a final determination."
        )
        actions = [
            "Request additional clinical documentation from ordering provider",
            "Schedule peer-to-peer clinical review",
            "Contact provider within 24 hours for missing information",
        ]

    return {
        "decision": decision,
        "confidence_score": round(confidence, 2),
        "reasoning": reasoning,
        "policy_citations": _extract_relevant_citations(retrieved_policies, decision),
        "recommended_actions": actions,
        "matched_approval_indicators": matched_approval[:6],
        "matched_denial_indicators": matched_denial[:6],
    }


def analyze_prior_auth(clinical_summary, diagnosis_codes, cpt_codes, payer,
                       retrieved_policies, force_rule_based=False):
    start = time.time()
    result = rule_based_decision(
        clinical_summary, diagnosis_codes, cpt_codes, retrieved_policies, payer
    )
    ms = int((time.time() - start) * 1000)
    return result, "semantic-rule-engine-v2", ms