# Clinical Trial Eligibility (Type 2 Diabetes) — Hybrid Rule-Based + Gemini (LLM) Reasoning

This project is a **hybrid eligibility engine** for **Type 2 Diabetes clinical trials**:

1) **Rule-based screening (deterministic, safe baseline)**  
   A transparent rules engine checks clear medical criteria (age, diagnosis, HbA1c, meds, pregnancy, etc.).

2) **LLM reasoning layer (Gemini)**  
   Gemini is used **only to explain and justify** eligibility decisions (Eligible / Not eligible / Uncertain),
   referencing the trial criteria and patient data.  
   The LLM **does not replace** the deterministic medical logic.

3) **Missing data handling**  
   If required fields are missing, the system returns **Uncertain** (never guesses).

---

## Repository Structure

```
/app
  streamlit_app.py        # Streamlit UI

/data
  patients.json           # synthetic structured patients
  clinical_notes.json     # synthetic unstructured notes
  trials.json             # clinical trial criteria
  generate_synthetic_data.py

/llm
  gemini_agent.py         # prompt + Gemini API calls + robust JSON parsing

eligibility.py            # deterministic screening engine (rule-based baseline)
requirements.txt
```

---

## Quickstart (Local)

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Set your Gemini API key
Recommended: set environment variable:

```bash
export GEMINI_API_KEY="YOUR_KEY"
# or: export GOOGLE_API_KEY="YOUR_KEY"
```

> The Google GenAI SDK automatically reads `GEMINI_API_KEY` or `GOOGLE_API_KEY`.  
> See Google docs: API key setup. 

### 3) Generate synthetic data
```bash
python -m data.generate_synthetic_data --n 80
```

### 4) Run Streamlit
```bash
streamlit run app/streamlit_app.py
```

---

## Deploy (Streamlit Community Cloud)

1) Push this repo to GitHub.
2) In Streamlit Community Cloud, create a new app from the repo.
3) Add your API key in **App settings → Secrets**:
```toml
GEMINI_API_KEY = "YOUR_KEY"
```
Streamlit will expose it via `st.secrets`, and we also read environment variables.

Docs: Streamlit secrets management.

---

## Notes

- This project is for **education/demo** using **synthetic data** only.
- Do not use in production without clinical validation and governance.
