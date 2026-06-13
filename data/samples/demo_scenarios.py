"""
Demo clinical scenarios for the Prior Auth AI Engine Streamlit app.
Three realistic cases covering orthopedic, oncology, and behavioral health.
"""

DEMO_SCENARIOS = {
    "🦴 Total Knee Replacement (Likely Approved)": {
        "label": "Total Knee Replacement",
        "payer": "UnitedHealthcare",
        "diagnosis_codes": ["M17.11", "M17.31"],
        "cpt_codes": ["27447"],
        "patient_age": 61,
        "patient_gender": "Female",
        "clinical_summary": """61-year-old female presenting for prior authorization for right total knee arthroplasty (TKA).

DIAGNOSIS: Severe primary osteoarthritis of right knee, confirmed on weight-bearing X-ray showing Kellgren-Lawrence Grade 4 changes with bone-on-bone medial compartment narrowing, osteophyte formation, and subchondral sclerosis.

CONSERVATIVE TREATMENT HISTORY (documented):
- Physical therapy: 14 sessions over 4 months with licensed PT (session notes attached). Minimal improvement.
- NSAIDs: Naproxen 500mg BID × 5 months, then Meloxicam 15mg daily × 2 months. Inadequate pain relief.
- Intra-articular corticosteroid injections: 2 injections (March 2024, June 2024). Temporary relief < 4 weeks each.
- Viscosupplementation: Hyaluronic acid series (3 injections). Minimal benefit.

FUNCTIONAL STATUS: Patient reports inability to climb stairs without severe pain, walks less than one block before stopping due to pain (VAS 8/10 at rest, 10/10 with activity). Cannot perform ADLs independently. Has stopped working as elementary school teacher due to inability to stand.

COMORBIDITIES & CLEARANCE:
- BMI: 27.4 (within guidelines)
- No active infections
- Cardiologist clearance obtained (EF 60%, NYHA Class I)
- Pre-operative labs: normal CBC, BMP, coagulation

SURGEON: Dr. Sarah Johnson, MD, Board-Certified Orthopedic Surgery, NPI: 1234567890
FACILITY: Regional Medical Center, Joint Replacement Center of Excellence"""
    },

    "🧬 Pembrolizumab for NSCLC (Likely Approved)": {
        "label": "Pembrolizumab (Keytruda) for NSCLC",
        "payer": "BlueCross BlueShield",
        "diagnosis_codes": ["C34.12"],
        "cpt_codes": ["J9271"],
        "patient_age": 67,
        "patient_gender": "Male",
        "clinical_summary": """67-year-old male with newly diagnosed advanced non-small cell lung cancer (NSCLC), adenocarcinoma histology, requesting first-line pembrolizumab (Keytruda) monotherapy.

DIAGNOSIS: Stage IIIB/IVA NSCLC, adenocarcinoma (ICD-10: C34.12). CT chest/abdomen/pelvis and PET scan confirmed bilateral mediastinal lymph node involvement without distant organ metastasis. Brain MRI negative for CNS involvement.

BIOMARKER TESTING (all completed prior to this request):
- PD-L1 Tumor Proportion Score (TPS): 78% (HIGH EXPRESSION - QualTek Labs, July 2024)
- EGFR mutation status: NEGATIVE (cobas EGFR Mutation Test v2)
- ALK rearrangement: NEGATIVE (Vysis ALK Break Apart FISH)
- ROS1 rearrangement: NEGATIVE (RT-PCR)
- KRAS G12C: NEGATIVE
- NTRK fusion: NEGATIVE
- TMB: 14 mut/Mb (intermediate)

TREATMENT RATIONALE: Patient meets FDA-approved first-line indication for pembrolizumab monotherapy per KEYNOTE-024 trial data. PD-L1 TPS ≥ 50% with no targetable driver mutations. NCCN Category 1 preferred regimen. Patient PS ECOG 1.

ONCOLOGIST: Dr. Michael Chen, MD PhD, Medical Oncology, Board-Certified
INSTITUTION: University Cancer Center, NCI-Designated Comprehensive Cancer Center
INTENDED REGIMEN: Pembrolizumab 200mg IV Q3W"""
    },

    "🧠 MRI Brain for Headaches (Likely Denied)": {
        "label": "Brain MRI for Chronic Headaches",
        "payer": "Aetna",
        "diagnosis_codes": ["G43.909"],
        "cpt_codes": ["70553"],
        "patient_age": 34,
        "patient_gender": "Female",
        "clinical_summary": """34-year-old female requesting brain MRI with and without contrast for evaluation of headaches.

CHIEF COMPLAINT: Headaches occurring 3-4 times per month, described as bilateral throbbing, moderate severity (5-6/10). Duration 4-8 hours per episode.

HISTORY: Patient reports headaches for approximately 6 months. Denies visual disturbances, focal neurological symptoms, weakness, numbness, or cognitive changes. No fever, no neck stiffness. No head trauma. Headaches are not positional. No awakening from sleep.

NEUROLOGICAL EXAM: Completely normal. Cranial nerves II-XII intact. No papilledema on fundoscopic exam. Coordination and gait normal. DTRs symmetric and normal.

PRIOR WORKUP: None formally documented. Patient states she "tried Advil" for 2-3 episodes with some relief. No formal headache diary. No neurology consultation. No trial of preventive medications.

PATIENT REQUEST: Patient requests MRI as she is "worried about a brain tumor." No family history of brain tumors. No HIV, no malignancy, no immunosuppression.

ORDERING PHYSICIAN: Dr. Amy Park, MD, Internal Medicine (PCP)

NOTE: No red flag features identified (no thunderclap onset, no progressive worsening, no papilledema, no focal deficits, no systemic symptoms, no immunosuppression)."""
    },
}
