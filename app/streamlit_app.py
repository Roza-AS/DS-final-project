"""
Streamlit UI for hybrid clinical trial eligibility.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import json
from pathlib import Path

import streamlit as st
import pandas as pd

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

def get_note(notes, pid: str) -> str:
    for n in notes:
        if n.get("patient_id") == pid:
            return n.get("note", "")
    return ""

st.title("🧪 Clinical Trial Eligibility (Type 2 Diabetes)")
st.caption("Hybrid approach: **Rule-based screening** + **Gemini explanation layer** (no guessing; missing info → Uncertain).")

if not PATIENTS_PATH.exists():
    st.warning("No synthetic data found yet. Generate it first:\n\n`python -m data.generate_synthetic_data --n 80`")
    st.stop()

patients = load_json(PATIENTS_PATH)
notes = load_json(NOTES_PATH) if NOTES_PATH.exists() else []
trials = load_json(TRIALS_PATH)

# Sidebar controls
st.sidebar.header("Controls")
trial_title_map = {f'{t["trial_id"]} — {t["title"]}': t for t in trials}
trial_label = st.sidebar.selectbox("Select clinical trial", list(trial_title_map.keys()))
trial = trial_title_map[trial_label]

status_filter = st.sidebar.multiselect(
    "Show statuses",
    ["Eligible", "Not eligible", "Uncertain"],
    default=["Eligible", "Not eligible", "Uncertain"]
)

# Build overview table
rows = []
for p in patients:
    res = screen_patient_for_trial(p, trial)
    rows.append({
        "patient_id": p.get("patient_id"),
        "age": p.get("age_years"),
        "sex": p.get("sex"),
        "hba1c": p.get("hba1c_percent"),
        "bmi": p.get("bmi"),
        "egfr": p.get("egfr"),
        "status": res.status,
        "why_rule_based": " | ".join(res.reasons),
        "_res": res,  # keep object
    })

df = pd.DataFrame(rows)
df_view = df[df["status"].isin(status_filter)].copy()

col1, col2 = st.columns([1.15, 0.85], gap="large")

with col1:
    st.subheader("📋 Patients")
    st.dataframe(
        df_view.drop(columns=["_res"]),
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.subheader("👤 Patient Snapshot")
    pid = st.selectbox("Select patient", df_view["patient_id"].tolist() if len(df_view) else df["patient_id"].tolist())
    p = next(x for x in patients if x["patient_id"] == pid)
    note = get_note(notes, pid)
    res = screen_patient_for_trial(p, trial)

    a, b = st.columns(2)
    with a:
        st.metric("Rule-based status", res.status)
    with b:
        st.metric("Missing fields", str(len(res.missing_fields)))

    st.write("**Structured data**")
    st.json(p, expanded=False)

    st.write("**Clinical note (unstructured)**")
    st.text_area("note", value=note, height=180)

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
    if st.button("Explain with Gemini (LLM)", type="primary"):
        with st.spinner("Calling Gemini..."):
            out = explain_eligibility_with_gemini(
                patient=p,
                clinical_note=note,
                trial=trial,
                rule_based=res,
            )
        st.success("Done")
        st.json(out, expanded=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "Tip: If you deploy on Streamlit Cloud, add `GEMINI_API_KEY` in **Secrets**."
)
