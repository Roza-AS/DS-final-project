"""
Streamlit UI for hybrid clinical trial eligibility.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
from math import inf
from pathlib import Path

import pandas as pd
import streamlit as st

from eligibility import screen_patient_for_trial
from llm.gemini_agent import explain_eligibility_with_gemini

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

PATIENTS_PATH = DATA_DIR / "patients.json"
NOTES_PATH = DATA_DIR / "clinical_notes.json"
TRIALS_PATH = DATA_DIR / "trials.json"

st.set_page_config(page_title="Clinical Trial Eligibility (T2D)", layout="wide")


@st.cache_data
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def get_note(notes_by_pid: dict, pid: str) -> str:
    return notes_by_pid.get(pid, "")


# ---------- Ranking helpers ----------
def _range_distance(value, lo=None, hi=None) -> float:
    """0 if within range; positive distance to nearest boundary if outside; inf if missing."""
    if value is None:
        return inf
    if lo is not None and value < lo:
        return float(lo - value)
    if hi is not None and value > hi:
        return float(value - hi)
    return 0.0


def _trial_closeness(patient: dict, trial: dict) -> float:
    """Lower is better. Uses numeric inclusion ranges from the trial JSON."""
    inc = trial.get("inclusion", {})
    dist = 0.0

    if "age_years" in inc:
        dist += _range_distance(
            patient.get("age_years"),
            inc["age_years"].get("min"),
            inc["age_years"].get("max"),
        )

    if "hba1c_percent" in inc:
        dist += _range_distance(
            patient.get("hba1c_percent"),
            inc["hba1c_percent"].get("min"),
            inc["hba1c_percent"].get("max"),
        )

    if "bmi" in inc:
        dist += _range_distance(
            patient.get("bmi"),
            inc["bmi"].get("min"),
            inc["bmi"].get("max"),
        )

    if "egfr" in inc:
        dist += _range_distance(
            patient.get("egfr"),
            inc["egfr"].get("min"),
            inc["egfr"].get("max"),
        )

    if "uacr_mg_g" in inc:
        dist += _range_distance(
            patient.get("uacr_mg_g"),
            inc["uacr_mg_g"].get("min"),
            None,
        )

    return dist


def _phase_rank(phase: str) -> int:
    """Lower is better. Prefers Phase 3 over Phase 2 over Phase 1."""
    p = (phase or "").strip().lower()
    if "phase 3" in p:
        return 0
    if "phase 2" in p:
        return 1
    if "phase 1" in p:
        return 2
    return 9


def rank_key(patient: dict, trial: dict, res) -> tuple:
    """Lower tuple is better."""
    status_priority = {"Eligible": 0, "Uncertain": 1, "Not eligible": 2}
    return (
        status_priority.get(res.status, 99),
        len(res.criteria_failed or []),      # fewer failures better
        len(res.missing_fields or []),       # fewer missing fields better
        _trial_closeness(patient, trial),    # closer to inclusion ranges better
        _phase_rank(trial.get("phase", "")), # optional
        -len(res.criteria_passed or []),     # more passes better
    )


@st.cache_data
def screen_and_rank_trials_for_patient(patient: dict, trials: list[dict]) -> list[dict]:
    ranked = []
    for t in trials:
        res = screen_patient_for_trial(patient, t)
        ranked.append(
            {
                "trial_id": t.get("trial_id"),
                "title": t.get("title"),
                "phase": t.get("phase"),
                "status": res.status,
                "missing_fields_count": len(res.missing_fields or []),
                "failed_criteria_count": len(res.criteria_failed or []),
                "passed_checks_count": len(res.criteria_passed or []),
                "closeness": _trial_closeness(patient, t),
                "_trial": t,
                "_res": res,
            }
        )

    ranked.sort(key=lambda r: rank_key(patient, r["_trial"], r["_res"]))
    return ranked


# ---------- UI ----------
st.title("🧪 Clinical Trial Eligibility (Type 2 Diabetes)")
st.caption(
    "Hybrid approach: **Rule-based screening** + **Gemini explanation layer** "
    "(no guessing; missing info → Uncertain)."
)

if not PATIENTS_PATH.exists():
    st.warning("No synthetic data found yet. Generate it first:\n\n`python -m data.generate_synthetic_data --n 80`")
    st.stop()

patients = load_json(PATIENTS_PATH)
notes = load_json(NOTES_PATH) if NOTES_PATH.exists() else []
trials = load_json(TRIALS_PATH)

notes_by_pid = {n.get("patient_id"): n.get("note", "") for n in notes} if notes else {}

# Sidebar controls
st.sidebar.header("Controls")

mode = st.sidebar.radio(
    "View mode",
    ["Trial → Patients", "Patient → Trial ranking"],
    index=0,
)

status_filter = st.sidebar.multiselect(
    "Show statuses",
    ["Eligible", "Not eligible", "Uncertain"],
    default=["Eligible", "Not eligible", "Uncertain"],
)

st.sidebar.markdown("---")
st.sidebar.info("Tip: If you deploy on Streamlit Cloud, add `GEMINI_API_KEY` in **Secrets**.")

# ---------- Mode 1: Trial → Patients (your original view) ----------
if mode == "Trial → Patients":
    trial_title_map = {f'{t["trial_id"]} — {t["title"]}': t for t in trials}
    trial_label = st.sidebar.selectbox("Select clinical trial", list(trial_title_map.keys()))
    trial = trial_title_map[trial_label]

    rows = []
    for p in patients:
        res = screen_patient_for_trial(p, trial)
        rows.append(
            {
                "patient_id": p.get("patient_id"),
                "age": p.get("age_years"),
                "sex": p.get("sex"),
                "hba1c": p.get("hba1c_percent"),
                "bmi": p.get("bmi"),
                "egfr": p.get("egfr"),
                "status": res.status,
                "why_rule_based": " | ".join(res.reasons),
                "_res": res,
            }
        )

    df = pd.DataFrame(rows)
    df_view = df[df["status"].isin(status_filter)].copy()

    col1, col2 = st.columns([1.15, 0.85], gap="large")

    with col1:
        st.subheader("📋 Patients")
        st.dataframe(
            df_view.drop(columns=["_res"]),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.subheader("👤 Patient Snapshot")
        pid_list = df_view["patient_id"].tolist() if len(df_view) else df["patient_id"].tolist()
        pid = st.selectbox("Select patient", pid_list, key="trial_mode_pid")
        p = next(x for x in patients if x.get("patient_id") == pid)
        note = get_note(notes_by_pid, pid)
        res = screen_patient_for_trial(p, trial)

        a, b = st.columns(2)
        with a:
            st.metric("Rule-based status", res.status)
        with b:
            st.metric("Missing fields", str(len(res.missing_fields)))

        st.write("**Structured data**")
        st.json(p, expanded=False)

        st.write("**Clinical note (unstructured)**")
        st.text_area("note", value=note, height=180, key="trial_mode_note")

        st.divider()
        st.write("**Rule-based explanation**")
        st.write("- " + "\n- ".join(res.reasons))
        if res.criteria_failed:
            st.write("**Failed criteria:**")
            st.write("- " + "\n- ".join(res.criteria_failed))
        if res.criteria_passed:
            st.write("**Passed checks:**")
            st.write("- " + "\n- ".join(res.criteria_passed))
        if res.missing_fields:
            st.write("**Missing fields:**")
            st.write("- " + "\n- ".join(res.missing_fields))

        st.divider()
        st.write("### 🤖 Gemini explanation")
        if st.button("Explain with Gemini (LLM)", type="primary", key="trial_mode_gemini"):
            with st.spinner("Calling Gemini..."):
                out = explain_eligibility_with_gemini(
                    patient=p,
                    clinical_note=note,
                    trial=trial,
                    rule_based=res,
                )
            st.success("Done")
            st.json(out, expanded=True)

# ---------- Mode 2: Patient → Trial ranking (NEW) ----------
else:
    st.subheader("🎯 Patient → Trial ranking")

    pid = st.selectbox("Select patient", [p.get("patient_id") for p in patients], key="rank_mode_pid")
    patient = next(p for p in patients if p.get("patient_id") == pid)
    note = get_note(notes_by_pid, pid)

    ranked = screen_and_rank_trials_for_patient(patient, trials)

    df_ranked = pd.DataFrame(
        [
            {
                "trial_id": r["trial_id"],
                "title": r["title"],
                "phase": r["phase"],
                "status": r["status"],
                "failed_criteria": r["failed_criteria_count"],
                "missing_fields": r["missing_fields_count"],
                "passed_checks": r["passed_checks_count"],
                "closeness": r["closeness"],
            }
            for r in ranked
        ]
    )
    df_ranked = df_ranked[df_ranked["status"].isin(status_filter)].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Eligible", int((df_ranked["status"] == "Eligible").sum()))
    c2.metric("Uncertain", int((df_ranked["status"] == "Uncertain").sum()))
    c3.metric("Not eligible", int((df_ranked["status"] == "Not eligible").sum()))

    colA, colB = st.columns([1.2, 0.8], gap="large")

    with colA:
        st.write("### 🏁 Ranked trials (best → worst)")
        st.dataframe(df_ranked, use_container_width=True, hide_index=True)

    with colB:
        st.write("### 🔍 Trial details")

        # Use the *full* ranked list for drill-down (even if filtered),
        # otherwise the index mapping becomes confusing.
        options = [f'{r["trial_id"]} — {r["title"]} ({r["status"]})' for r in ranked]
        chosen = st.selectbox("Select trial", options, key="rank_mode_trial_pick")
        r = ranked[options.index(chosen)]
        trial = r["_trial"]
        res = r["_res"]

        st.metric("Rule-based status", res.status)
        st.metric("Missing fields", str(len(res.missing_fields or [])))

        st.write("**Clinical note (unstructured)**")
        st.text_area("note", value=note, height=180, key="rank_mode_note")

        st.divider()
        st.write("**Rule-based explanation**")
        st.write("- " + "\n- ".join(res.reasons))
        if res.criteria_failed:
            st.write("**Failed criteria:**")
            st.write("- " + "\n- ".join(res.criteria_failed))
        if res.criteria_passed:
            st.write("**Passed checks:**")
            st.write("- " + "\n- ".join(res.criteria_passed))
        if res.missing_fields:
            st.write("**Missing fields:**")
            st.write("- " + "\n- ".join(res.missing_fields))

        st.divider()
        st.write("### 🤖 Gemini explanation (selected trial)")
        if st.button("Explain with Gemini (LLM)", type="primary", key="rank_mode_gemini"):
            with st.spinner("Calling Gemini..."):
                out = explain_eligibility_with_gemini(
                    patient=patient,
                    clinical_note=note,
                    trial=trial,
                    rule_based=res,
                )
            st.success("Done")
            st.json(out, expanded=True)

