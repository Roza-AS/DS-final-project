"""
Generate synthetic T2D patients + clinical notes for local testing.

Usage:
    python -m data.generate_synthetic_data --n 80 --seed 42

Outputs:
    data/patients.json
    data/clinical_notes.json
"""
from __future__ import annotations
import argparse
import json
import os
import random
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
PATIENTS_PATH = DATA_DIR / "patients.json"
NOTES_PATH = DATA_DIR / "clinical_notes.json"

SEX = ["female", "male"]
SMOKING = ["never", "former", "current"]
DIAG_BASE = ["type 2 diabetes", "hypertension", "dyslipidemia", "obesity", "ckd", "asthma"]
MEDS = [
    "metformin", "insulin", "dpp-4 inhibitor", "sglt2 inhibitor",
    "glp-1 receptor agonist", "statin", "ace inhibitor", "arb"
]

def maybe(val, p_missing: float, rng: random.Random):
    return None if rng.random() < p_missing else val

def make_patient(i: int, rng: random.Random) -> dict:
    pid = f"P{str(i).zfill(4)}"

    # ages: broad, but biased toward adult middle
    age = int(rng.triangular(18, 85, 55))

    sex = rng.choice(SEX)

    # HbA1c plausible distribution: 6.0-12.5
    hba1c = round(rng.triangular(5.8, 12.5, 7.6), 1)

    bmi = round(rng.triangular(20, 55, 32), 1)

    # eGFR 15-120
    egfr = int(rng.triangular(15, 120, 85))

    # UACR 5-2000 mg/g (log-ish)
    uacr = int(10 ** rng.uniform(0.7, 3.2))  # ~5 to ~1600

    smoking = rng.choice(SMOKING)

    # Pregnant can only be female; keep rare
    pregnant = False
    if sex == "female" and 18 <= age <= 50:
        pregnant = rng.random() < 0.06

    # Create diagnoses
    diagnoses = ["type 2 diabetes"]
    for d in DIAG_BASE:
        if d != "type 2 diabetes" and rng.random() < 0.35:
            diagnoses.append(d)

    # Some patients are actually T1D
    if rng.random() < 0.03:
        diagnoses = ["type 1 diabetes"]
    type1 = "type 1 diabetes" in diagnoses

    # meds
    meds = []
    # metformin common in T2D
    if (not type1) and rng.random() < 0.78:
        meds.append("metformin")
    # insulin sometimes
    if rng.random() < (0.25 if not type1 else 0.7):
        meds.append("insulin")
    # add-ons
    for m in ["dpp-4 inhibitor", "sglt2 inhibitor", "glp-1 receptor agonist"]:
        if rng.random() < 0.22:
            meds.append(m)
    # other comorbidity meds
    if "dyslipidemia" in diagnoses and rng.random() < 0.75:
        meds.append("statin")
    if "hypertension" in diagnoses and rng.random() < 0.6:
        meds.append(rng.choice(["ace inhibitor", "arb"]))

    meds = sorted(set(meds))

    # metformin stable months (if present)
    metformin_stable = None
    if "metformin" in meds:
        metformin_stable = int(rng.triangular(1, 36, 10))

    # recent MI/stroke months: mostly none
    recent_mi_or_stroke_months = None
    if rng.random() < 0.08:
        recent_mi_or_stroke_months = int(rng.triangular(1, 60, 18))

    # Flags used by trials
    severe_renal_impairment = egfr < 30
    dialysis = severe_renal_impairment and (rng.random() < 0.05)
    kidney_transplant = rng.random() < 0.01
    eating_disorder = rng.random() < 0.03

    # Inject missingness into some fields
    p_missing = 0.08
    patient = {
        "patient_id": pid,
        "age_years": maybe(age, p_missing, rng),
        "sex": maybe(sex, 0.03, rng),
        "diagnoses": maybe(diagnoses, 0.02, rng),
        "hba1c_percent": maybe(hba1c, p_missing, rng),
        "bmi": maybe(bmi, p_missing, rng),
        "egfr": maybe(egfr, p_missing, rng),
        "uacr_mg_g": maybe(uacr, 0.12, rng),
        "smoking_status": maybe(smoking, 0.05, rng),
        "pregnant": maybe(pregnant, 0.08, rng),
        "medications": maybe(meds, p_missing, rng),
        "metformin_stable_months": maybe(metformin_stable, 0.15, rng),
        "recent_mi_or_stroke_months": maybe(recent_mi_or_stroke_months, 0.2, rng),
        "type1_diabetes": type1,
        "severe_renal_impairment": severe_renal_impairment,
        "dialysis": dialysis,
        "kidney_transplant": kidney_transplant,
        "eating_disorder": eating_disorder,
    }
    return patient

def make_note(patient: dict, rng: random.Random) -> str:
    """
    Create an unstructured clinical note that may include contradictory or missing info,
    to stress-test the LLM and rule-based layer.
    """
    pid = patient["patient_id"]
    age = patient.get("age_years")
    sex = patient.get("sex")
    hba1c = patient.get("hba1c_percent")
    egfr = patient.get("egfr")
    bmi = patient.get("bmi")
    meds = patient.get("medications") or []
    diags = patient.get("diagnoses") or []
    preg = patient.get("pregnant")

    lines = []
    lines.append(f"Patient {pid} seen in endocrinology clinic for diabetes follow-up.")
    if age is not None:
        lines.append(f"Age: {age} years.")
    if sex:
        lines.append(f"Sex: {sex}.")
    if "type 2 diabetes" in diags:
        lines.append("Diagnosis: Type 2 diabetes mellitus, long-standing.")
    elif "type 1 diabetes" in diags:
        lines.append("Diagnosis: Type 1 diabetes mellitus.")
    else:
        lines.append("Diabetes type not clearly documented in this note.")

    if hba1c is not None:
        lines.append(f"Most recent HbA1c: {hba1c}%.")
    else:
        lines.append("HbA1c not available in chart today.")

    if bmi is not None:
        lines.append(f"BMI around {bmi} kg/m2.")
    if egfr is not None:
        lines.append(f"Renal function: eGFR {egfr} mL/min/1.73m2.")
    else:
        lines.append("Renal labs pending (eGFR unknown).")

    if meds:
        lines.append("Current meds: " + ", ".join(meds) + ".")
    else:
        lines.append("Medication list not available in this note.")

    # pregnancy mention
    if preg is True:
        lines.append("Pregnancy: currently pregnant; needs OB coordination.")
    elif preg is False and sex == "female" and (age is not None and 18 <= age <= 50):
        if rng.random() < 0.15:
            lines.append("Pregnancy status: not discussed today.")
        else:
            lines.append("Pregnancy: denies pregnancy.")
    else:
        if rng.random() < 0.1:
            lines.append("Pregnancy status unknown.")

    # add comorbidities flavor
    if "hypertension" in diags:
        lines.append("Comorbidity: hypertension (controlled).")
    if "dyslipidemia" in diags:
        lines.append("Comorbidity: dyslipidemia.")
    if "ckd" in diags:
        lines.append("Comorbidity: CKD noted (stage unspecified).")

    # cardiovascular event mention (sometimes)
    if patient.get("recent_mi_or_stroke_months") is not None:
        m = patient["recent_mi_or_stroke_months"]
        lines.append(f"History: MI/stroke about {m} months ago.")

    return " ".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    patients = [make_patient(i+1, rng) for i in range(args.n)]
    notes = [{"patient_id": p["patient_id"], "note": make_note(p, rng)} for p in patients]

    PATIENTS_PATH.write_text(json.dumps(patients, indent=2), encoding="utf-8")
    NOTES_PATH.write_text(json.dumps(notes, indent=2), encoding="utf-8")

    print(f"Wrote {len(patients)} patients to {PATIENTS_PATH}")
    print(f"Wrote {len(notes)} notes to {NOTES_PATH}")

if __name__ == "__main__":
    main()
