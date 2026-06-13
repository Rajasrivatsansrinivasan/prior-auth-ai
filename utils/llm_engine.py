"""
LLM Reasoning Engine for Prior Authorization Decisions
Uses free local HuggingFace models (no API key needed)
Primary: microsoft/phi-2 | Fallback: rule-based when no GPU
"""

import os
import re
import json
import logging
import time
from typing import List, Dict, Tuple, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Lazy-load transformers to speed startup
_pipeline = None
_model_name = None


def _get_pipeline():
    """Lazy-load the LLM pipeline."""
    global _pipeline, _model_name

    if _pipeline is not None:
        return _pipeline

    model_name = os.getenv("LLM_MODEL_NAME", "microsoft/phi-2")
    logger.info(f"Loading LLM: {model_name} (first load may take 1-2 minutes)...")

    try:
        from transformers import pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1
        device_label = "GPU" if device == 0 else "CPU"
        logger.info(f"Using device: {device_label}")

        _pipeline = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            max_new_tokens=600,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=50256,
            trust_remote_code=True,
        )
        _model_name = model_name
        logger.info(f"LLM loaded successfully: {model_name}")
        return _pipeline

    except Exception as e:
        logger.warning(f"Could not load LLM ({e}). Will use rule-based fallback.")
        return None


# ─────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical prior authorization specialist AI. Your job is to analyze a prior authorization request against retrieved payer policy documents and produce a structured recommendation.

You must:
1. Read the clinical information carefully
2. Compare it against the retrieved policy sections
3. Determine if the request meets approval criteria
4. Cite specific policy sections that support your decision
5. Provide a confidence score between 0.0 and 1.0

Respond ONLY in valid JSON format with this structure:
{
  "decision": "APPROVED" | "DENIED" | "NEEDS_REVIEW",
  "confidence_score": <float 0.0-1.0>,
  "reasoning": "<clear clinical reasoning in 2-4 sentences>",
  "policy_citations": [
    {
      "policy_id": "<policy ID>",
      "payer": "<payer name>",
      "section": "<section name>",
      "relevant_text": "<the specific policy text that applies>",
      "supports_decision": <true|false>
    }
  ],
  "recommended_actions": ["<action 1>", "<action 2>"]
}"""


def build_prompt(
    clinical_summary: str,
    diagnosis_codes: List[str],
    cpt_codes: List[str],
    payer: str,
    retrieved_policies: List[Dict],
) -> str:
    policy_context = ""
    for i, doc in enumerate(retrieved_policies, 1):
        policy_context += f"\n--- Policy Document {i} ---\n"
        policy_context += f"Policy ID: {doc['metadata']['policy_id']}\n"
        policy_context += f"Payer: {doc['metadata']['payer']}\n"
        policy_context += f"Title: {doc['metadata']['title']}\n"
        policy_context += f"Relevance Score: {doc['score']:.3f}\n"
        policy_context += f"Content:\n{doc['text']}\n"

    prompt = f"""{SYSTEM_PROMPT}

=== PRIOR AUTHORIZATION REQUEST ===
Payer: {payer}
Diagnosis Codes (ICD-10): {', '.join(diagnosis_codes)}
Procedure Codes (CPT): {', '.join(cpt_codes)}

Clinical Summary:
{clinical_summary}

=== RETRIEVED PAYER POLICY SECTIONS ===
{policy_context}

=== YOUR STRUCTURED DECISION (JSON only) ===
"""
    return prompt


# ─────────────────────────────────────────────
# Rule-based fallback (no LLM needed)
# ─────────────────────────────────────────────

APPROVAL_KEYWORDS = [
    "conservative treatment", "failed", "documented", "physical therapy",
    "chronic", "persistent", "specialist", "imaging confirmed", "grade 3",
    "grade 4", "full thickness", "severe", "moderate to severe",
]

DENIAL_KEYWORDS = [
    "acute", "first visit", "no prior treatment", "screening",
    "without symptoms", "less than", "no documentation",
]


def rule_based_decision(
    clinical_summary: str,
    diagnosis_codes: List[str],
    cpt_codes: List[str],
    retrieved_policies: List[Dict],
) -> Dict:
    """Fallback rule-based engine when no LLM is available."""
    summary_lower = clinical_summary.lower()

    approval_hits = sum(1 for kw in APPROVAL_KEYWORDS if kw in summary_lower)
    denial_hits = sum(1 for kw in DENIAL_KEYWORDS if kw in summary_lower)

    if approval_hits >= 3 and denial_hits == 0:
        decision = "APPROVED"
        confidence = min(0.55 + approval_hits * 0.04, 0.82)
    elif denial_hits > approval_hits:
        decision = "DENIED"
        confidence = min(0.50 + denial_hits * 0.05, 0.78)
    else:
        decision = "NEEDS_REVIEW"
        confidence = 0.45

    citations = []
    for doc in retrieved_policies[:2]:
        citations.append({
            "policy_id": doc["metadata"]["policy_id"],
            "payer": doc["metadata"]["payer"],
            "section": doc["metadata"]["title"],
            "relevant_text": doc["text"][:300] + "...",
            "supports_decision": decision == "APPROVED",
        })

    actions = []
    if decision == "APPROVED":
        actions = ["Proceed with authorization", "Notify provider of approval"]
    elif decision == "DENIED":
        actions = ["Send denial notice with policy citation", "Inform provider of appeal rights"]
    else:
        actions = ["Request additional clinical documentation", "Schedule peer-to-peer review"]

    return {
        "decision": decision,
        "confidence_score": round(confidence, 2),
        "reasoning": f"Rule-based analysis: Found {approval_hits} approval indicators and {denial_hits} denial indicators in clinical summary. Policy documents retrieved for context.",
        "policy_citations": citations,
        "recommended_actions": actions,
    }


# ─────────────────────────────────────────────
# Main reasoning function
# ─────────────────────────────────────────────

def parse_llm_json(raw_output: str) -> Optional[Dict]:
    """Extract JSON from LLM output, handling common formatting issues."""
    # Try to find JSON block
    json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        # Try cleaning up common LLM JSON mistakes
        cleaned = json_match.group()
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            return None


def analyze_prior_auth(
    clinical_summary: str,
    diagnosis_codes: List[str],
    cpt_codes: List[str],
    payer: str,
    retrieved_policies: List[Dict],
    force_rule_based: bool = False,
) -> Tuple[Dict, str, int]:
    """
    Main reasoning function.
    Returns: (decision_dict, model_used, processing_time_ms)
    """
    start_time = time.time()

    if force_rule_based:
        result = rule_based_decision(clinical_summary, diagnosis_codes, cpt_codes, retrieved_policies)
        ms = int((time.time() - start_time) * 1000)
        return result, "rule-based-fallback", ms

    pipe = _get_pipeline()

    if pipe is None:
        result = rule_based_decision(clinical_summary, diagnosis_codes, cpt_codes, retrieved_policies)
        ms = int((time.time() - start_time) * 1000)
        return result, "rule-based-fallback", ms

    prompt = build_prompt(clinical_summary, diagnosis_codes, cpt_codes, payer, retrieved_policies)

    try:
        outputs = pipe(prompt)
        raw_text = outputs[0]["generated_text"]
        # Strip the prompt from output (some models return prompt + completion)
        if prompt in raw_text:
            raw_text = raw_text.replace(prompt, "")

        parsed = parse_llm_json(raw_text)
        if parsed:
            # Validate required fields
            for field in ["decision", "confidence_score", "reasoning"]:
                if field not in parsed:
                    raise ValueError(f"Missing field: {field}")
            if parsed["decision"] not in ("APPROVED", "DENIED", "NEEDS_REVIEW"):
                parsed["decision"] = "NEEDS_REVIEW"
            parsed.setdefault("policy_citations", [])
            parsed.setdefault("recommended_actions", [])

            ms = int((time.time() - start_time) * 1000)
            return parsed, _model_name, ms
        else:
            logger.warning("LLM output could not be parsed as JSON. Using rule-based fallback.")
            result = rule_based_decision(clinical_summary, diagnosis_codes, cpt_codes, retrieved_policies)
            ms = int((time.time() - start_time) * 1000)
            return result, "rule-based-fallback (parse error)", ms

    except Exception as e:
        logger.error(f"LLM inference error: {e}")
        result = rule_based_decision(clinical_summary, diagnosis_codes, cpt_codes, retrieved_policies)
        ms = int((time.time() - start_time) * 1000)
        return result, "rule-based-fallback (error)", ms

CPT_DESCRIPTIONS = {'27447': 'Total Knee Arthroplasty', '27130': 'Total Hip Arthroplasty', '29827': 'Rotator Cuff Repair', '72148': 'Lumbar Spine MRI', '70551': 'Brain MRI without contrast', '70552': 'Brain MRI with contrast', '70553': 'Brain MRI with/without contrast', '73721': 'Knee MRI', 'J9271': 'Pembrolizumab (Keytruda)', 'J9355': 'Trastuzumab (Herceptin)', 'J0401': 'Aripiprazole (Abilify)'}

ICD_DESCRIPTIONS = {'M17.11': 'Primary osteoarthritis right knee', 'M17.12': 'Primary osteoarthritis left knee', 'M17.31': 'Post-traumatic osteoarthritis right knee', 'C34.12': 'NSCLC upper lobe left lung', 'C34.11': 'NSCLC upper lobe right lung', 'G43.909': 'Migraine unspecified', 'M54.5': 'Low back pain', 'F20.9': 'Schizophrenia unspecified', 'F32.9': 'Major depressive disorder'}
