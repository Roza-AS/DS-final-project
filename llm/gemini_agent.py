"""
Gemini reasoning layer: produces an *explanation* for a trial eligibility decision.

Key design goals:
- The deterministic `eligibility.py` remains the decision baseline.
- Gemini is used to explain, cross-check, and label "Uncertain" when data is missing.
- Robust parsing: request JSON output, parse carefully, and degrade gracefully.
"""
from __future__ import annotations
from dataclasses import asdict
import json
import os
import re
from typing import Any, Dict, Optional

from google import genai
from google.genai import types

from eligibility import ScreenResult

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # fast & cheap default

JSON_SCHEMA_HINT = {
    "type": "object",
    "properties": {
        "final_status": {"type": "string", "enum": ["Eligible", "Not eligible", "Uncertain"]},
        "summary": {"type": "string"},
        "criteria_matched": {"type": "array", "items": {"type": "string"}},
        "criteria_violated": {"type": "array", "items": {"type": "string"}},
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "recommended_next_questions": {"type": "array", "items": {"type": "string"}},
        "consistency_check_with_rule_based": {
            "type": "object",
            "properties": {
                "rule_based_status": {"type": "string"},
                "llm_agrees": {"type": "boolean"},
                "notes": {"type": "string"},
            },
            "required": ["rule_based_status", "llm_agrees", "notes"],
        },
        "safety_note": {"type": "string"},
    },
    "required": [
        "final_status",
        "summary",
        "criteria_matched",
        "criteria_violated",
        "missing_information",
        "recommended_next_questions",
        "consistency_check_with_rule_based",
        "safety_note",
    ],
}

SYSTEM_INSTRUCTIONS = """You are a clinical-trial eligibility explanation assistant.
You do NOT make medical decisions. You explain eligibility decisions based on explicit trial criteria and given patient data.
If required information is missing, you MUST return final_status="Uncertain" and list missing_information. Do NOT guess.
Be precise, cite which criteria were met or violated using the exact words/numbers from the criteria whenever possible.
Return ONLY valid JSON matching the requested schema. Do not include markdown.
"""

def _get_api_key_from_streamlit_secrets_if_present() -> Optional[str]:
    # Avoid importing streamlit unless needed.
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets"):
            # Common keys
            for k in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
                if k in st.secrets:
                    return st.secrets[k]
    except Exception:
        pass
    return None

def _make_client() -> genai.Client:
    """
    Create a Gemini client.
    - Prefers environment variables (GEMINI_API_KEY or GOOGLE_API_KEY).
    - If running under Streamlit Community Cloud, it may be stored in st.secrets.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or _get_api_key_from_streamlit_secrets_if_present()
    if api_key:
        return genai.Client(api_key=api_key)
    # fall back to env auto-detection (may still work if set in environment)
    return genai.Client()

def _extract_json(text: str) -> str:
    """
    Extract JSON from a model response that may contain extra text.
    """
    text = text.strip()
    # If already looks like JSON, return
    if text.startswith("{") and text.endswith("}"):
        return text
    # Try to find first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return m.group(0)
    return text

def explain_eligibility_with_gemini(
    patient: Dict[str, Any],
    clinical_note: str,
    trial: Dict[str, Any],
    rule_based: ScreenResult,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Returns a JSON dict (already parsed) describing the explanation.
    """
    client = _make_client()

    payload = {
        "patient_structured": patient,
        "patient_note_unstructured": clinical_note,
        "trial": trial,
        "rule_based_result": asdict(rule_based),
        "required_output_schema": JSON_SCHEMA_HINT,
    }

    # Using a single prompt for simplicity; could be split into system+user.
    prompt = (
        SYSTEM_INSTRUCTIONS
        + "\n\nINPUT:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nOUTPUT: Return ONLY JSON, no extra text."
    )

    # Try to request JSON output (best-effort)
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    text = getattr(resp, "text", None) or str(resp)
    raw = _extract_json(text)

    try:
        data = json.loads(raw)
    except Exception:
        # Degrade gracefully
        data = {
            "final_status": rule_based.status if rule_based else "Uncertain",
            "summary": "Model returned a non-JSON response; showing rule-based result instead.",
            "criteria_matched": rule_based.criteria_passed if rule_based else [],
            "criteria_violated": rule_based.criteria_failed if rule_based else [],
            "missing_information": rule_based.missing_fields if rule_based else ["unknown"],
            "recommended_next_questions": [
                "Provide missing required fields and re-run the screening."
            ],
            "consistency_check_with_rule_based": {
                "rule_based_status": rule_based.status if rule_based else "Uncertain",
                "llm_agrees": True,
                "notes": "Fallback mode due to parsing failure.",
            },
            "safety_note": "This is a demo; decisions must be validated clinically.",
            "_raw_model_text": text[:4000],
        }

    return data
