"""
Deterministic rule-based screening engine (baseline).
The LLM should NOT replace this logic; it should explain it.

Outputs a status:
- "Eligible"
- "Not eligible"
- "Uncertain" (missing required data)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

Status = str  # Eligible / Not eligible / Uncertain

def _norm_list(x):
    if x is None:
        return None
    if isinstance(x, list):
        return [str(v).strip().lower() for v in x]
    return [str(x).strip().lower()]

def _has_any(hay: List[str], needles: List[str]) -> bool:
    hay_set = set(hay)
    return any(n.lower() in hay_set for n in needles)

def _has_all(hay: List[str], needles: List[str]) -> bool:
    hay_set = set(hay)
    return all(n.lower() in hay_set for n in needles)

def _missing(*values) -> bool:
    return any(v is None for v in values)

@dataclass
class ScreenResult:
    status: Status
    reasons: List[str]
    missing_fields: List[str]
    criteria_passed: List[str]
    criteria_failed: List[str]

def screen_patient_for_trial(patient: Dict[str, Any], trial: Dict[str, Any]) -> ScreenResult:
    """
    Screen a single patient against a single trial JSON.

    IMPORTANT: The trial JSON is intended to be readable and explainable.
    Rule checks are implemented for a limited set of criterion types used in this demo.
    """
    reasons: List[str] = []
    missing: List[str] = []
    passed: List[str] = []
    failed: List[str] = []

    inc = trial.get("inclusion", {})
    exc = trial.get("exclusion", {})

    # Pull patient fields
    age = patient.get("age_years")
    diagnoses = _norm_list(patient.get("diagnoses"))
    hba1c = patient.get("hba1c_percent")
    bmi = patient.get("bmi")
    egfr = patient.get("egfr")
    uacr = patient.get("uacr_mg_g")
    meds = _norm_list(patient.get("medications"))
    preg = patient.get("pregnant")
    met_stable = patient.get("metformin_stable_months")
    recent_evt = patient.get("recent_mi_or_stroke_months")
    type1 = patient.get("type1_diabetes")

    # === Inclusion checks ===
    # Age
    if "age_years" in inc:
        lo = inc["age_years"].get("min")
        hi = inc["age_years"].get("max")
        if age is None:
            missing.append("age_years")
        else:
            if lo is not None and age < lo:
                failed.append(f"Age {age} < {lo}")
            elif hi is not None and age > hi:
                failed.append(f"Age {age} > {hi}")
            else:
                passed.append(f"Age within [{lo},{hi}]")

    # Diagnosis (any)
    if "diagnoses_any" in inc:
        if diagnoses is None:
            missing.append("diagnoses")
        else:
            if not _has_any(diagnoses, inc["diagnoses_any"]):
                failed.append("Does not have required T2D diagnosis")
            else:
                passed.append("Has required T2D diagnosis")

    # HbA1c
    if "hba1c_percent" in inc:
        lo = inc["hba1c_percent"].get("min")
        hi = inc["hba1c_percent"].get("max")
        if hba1c is None:
            missing.append("hba1c_percent")
        else:
            if lo is not None and hba1c < lo:
                failed.append(f"HbA1c {hba1c}% < {lo}%")
            elif hi is not None and hba1c > hi:
                failed.append(f"HbA1c {hba1c}% > {hi}%")
            else:
                passed.append(f"HbA1c within [{lo},{hi}]")

    # BMI
    if "bmi" in inc:
        lo = inc["bmi"].get("min")
        hi = inc["bmi"].get("max")
        if bmi is None:
            missing.append("bmi")
        else:
            if lo is not None and bmi < lo:
                failed.append(f"BMI {bmi} < {lo}")
            elif hi is not None and bmi > hi:
                failed.append(f"BMI {bmi} > {hi}")
            else:
                passed.append(f"BMI within [{lo},{hi}]")

    # eGFR
    if "egfr" in inc:
        lo = inc["egfr"].get("min")
        hi = inc["egfr"].get("max")
        if egfr is None:
            missing.append("egfr")
        else:
            if lo is not None and egfr < lo:
                failed.append(f"eGFR {egfr} < {lo}")
            elif hi is not None and egfr > hi:
                failed.append(f"eGFR {egfr} > {hi}")
            else:
                passed.append("eGFR within range")

    # UACR
    if "uacr_mg_g" in inc:
        lo = inc["uacr_mg_g"].get("min")
        if uacr is None:
            missing.append("uacr_mg_g")
        else:
            if lo is not None and uacr < lo:
                failed.append(f"UACR {uacr} < {lo}")
            else:
                passed.append("UACR meets minimum")

    # Medications: all
    if "medications_all" in inc:
        if meds is None:
            missing.append("medications")
        else:
            if not _has_all(meds, inc["medications_all"]):
                failed.append("Missing required meds: " + ", ".join(inc["medications_all"]))
            else:
                passed.append("Has all required meds")

    # Medications: any
    if "medications_any" in inc:
        if meds is None:
            missing.append("medications")
        else:
            if not _has_any(meds, inc["medications_any"]):
                failed.append("Does not use any of the allowed background meds")
            else:
                passed.append("Has an allowed background medication")

    # Metformin stable months
    if "metformin_stable_months" in inc:
        lo = inc["metformin_stable_months"].get("min")
        if met_stable is None:
            missing.append("metformin_stable_months")
        else:
            if lo is not None and met_stable < lo:
                failed.append(f"Metformin not stable >= {lo} months")
            else:
                passed.append("Metformin stable long enough")

    # === Exclusion checks ===
    # Pregnant
    if exc.get("pregnant") is True:
        if preg is None:
            missing.append("pregnant")
        elif preg is True:
            failed.append("Pregnant (exclusion)")
        else:
            passed.append("Not pregnant")

    # Any of these medications exclude
    if "medications_any" in exc:
        if meds is None:
            missing.append("medications")
        else:
            if _has_any(meds, exc["medications_any"]):
                failed.append("Uses excluded meds: " + ", ".join(exc["medications_any"]))
            else:
                passed.append("No excluded meds")

    # Recent MI/stroke
    if "recent_mi_or_stroke_months" in exc:
        max_m = exc["recent_mi_or_stroke_months"].get("max")
        if recent_evt is None:
            # It's okay to be None; it means no known event.
            passed.append("No documented recent MI/stroke")
        else:
            if max_m is not None and recent_evt <= max_m:
                failed.append(f"Recent MI/stroke within {max_m} months")
            else:
                passed.append("MI/stroke not within exclusion window")

    # Type1 diabetes
    if exc.get("type1_diabetes") is True:
        if type1 is True:
            failed.append("Type 1 diabetes (exclusion)")
        else:
            passed.append("Not type 1 diabetes")

    # Trial-specific boolean exclusions
    for flag in ["severe_renal_impairment", "eating_disorder", "dialysis", "kidney_transplant"]:
        if exc.get(flag) is True:
            v = patient.get(flag)
            if v is True:
                failed.append(f"{flag} (exclusion)")
            else:
                passed.append(f"{flag} not present")

    # === Decide status ===
    # If ANY required inclusion field is missing, we return Uncertain (even if other things fail/pass),
    # because we don't want to guess.
    # However: if there is a definitive exclusion or inclusion failure, we can still declare Not eligible
    # even with other missing fields, if the failure is decisive.
    decisive_failure = len(failed) > 0

    if not decisive_failure and len(missing) > 0:
        status = "Uncertain"
        reasons.append("Missing required information: " + ", ".join(sorted(set(missing))))
    elif decisive_failure:
        status = "Not eligible"
        reasons.append("One or more criteria failed: " + "; ".join(failed))
        if len(missing) > 0:
            reasons.append("Also missing info: " + ", ".join(sorted(set(missing))))
    else:
        status = "Eligible"
        reasons.append("All checked criteria passed, no exclusions triggered.")

    # Deduplicate lists
    missing = sorted(set(missing))
    passed = sorted(set(passed))
    failed = sorted(set(failed))

    return ScreenResult(
        status=status,
        reasons=reasons,
        missing_fields=missing,
        criteria_passed=passed,
        criteria_failed=failed,
    )
