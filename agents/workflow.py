"""
agents/workflow.py
==================
Workflow Orchestrator (simplified, stateless version)

Runs all 5 agents sequentially for each patient.
Does NOT use LangGraph's shared state to avoid cross-patient contamination.

Flow:
  demographic → clinical → left_eye → right_eye → fusion → prevention
"""

import os
from typing import Dict, Any
from agents.pipeline import (
    agent_demographic,
    agent_clinical_nlp,
    agent_image,
    agent_fusion,
    agent_prevention,
)
from config import DEFAULT_MODEL, IMAGE_DIR


def run_assessment(row: Dict, image_dir: str = IMAGE_DIR, model_key: str = DEFAULT_MODEL, verbose: bool = True) -> Dict:
    """
    Run the full 5-agent multimodal assessment on a single patient.

    Sequential execution (no shared state between patients):
      1. Demographic Risk (age, sex)
      2. Clinical NLP (keywords)
      3. Left Eye Vision (fundus image)
      4. Right Eye Vision (fundus image)
      5. Risk Fusion (combine all)
      6. Prevention (tier-based plan)

    Args:
        row:       Patient data dict (from OIA-ODIR CSV)
        image_dir: Path to directory containing fundus images
        model_key: LLM to use ('gpt4o_mini', 'gpt4o', 'gemini_2')
        verbose:   Print progress messages

    Returns:
        Complete assessment dict with all agent outputs
    """
    # Extract patient data
    age       = int(row.get("Patient Age", 0))
    sex       = str(row.get("Patient Sex", "Unknown"))
    left_file = str(row.get("Left-Fundus", ""))
    right_file = str(row.get("Right-Fundus", ""))

    # Combine keywords from both eyes
    left_kw  = str(row.get("Left-Diagnostic Keywords",  "") or "")
    right_kw = str(row.get("Right-Diagnostic Keywords", "") or "")
    keywords = f"{left_kw} {right_kw}".strip()

    # Full image paths
    left_path  = os.path.join(image_dir, left_file)  if left_file  else ""
    right_path = os.path.join(image_dir, right_file) if right_file else ""

    try:
        # Agent 1 — Demographic
        if verbose: print("  👤 [Agent 1] Demographic Risk Scorer...")
        demographic = agent_demographic(age=age, sex=sex, model_key=model_key)
        if verbose:
            print(f"     → Score: {demographic.get('demographic_risk_score', '?')} | Level: {demographic.get('age_risk_level', '?')}")

        # Agent 2 — Clinical NLP
        if verbose: print("  📝 [Agent 2] Clinical NLP Extractor...")
        clinical = agent_clinical_nlp(keywords=keywords, model_key=model_key)
        if verbose:
            print(f"     → Clinical Score: {clinical.get('text_derived_risk_score', '?')}")
            print(f"     → Findings: {clinical.get('diabetic_findings', [])}")

        # Agent 3a — Left Eye
        if verbose: print("  🔵 [Agent 3a] Left Eye Vision Analysis...")
        left_eye = agent_image(image_path=left_path, eye_side="left", model_key=model_key)
        if verbose:
            print(f"     → DR: {left_eye.get('dr', '?')} | Severity: {left_eye.get('dr_severity', '?')} | Quality: {left_eye.get('image_quality', '?')}")

        # Agent 3b — Right Eye
        if verbose: print("  🔴 [Agent 3b] Right Eye Vision Analysis...")
        right_eye = agent_image(image_path=right_path, eye_side="right", model_key=model_key)
        if verbose:
            print(f"     → DR: {right_eye.get('dr', '?')} | Severity: {right_eye.get('dr_severity', '?')} | Quality: {right_eye.get('image_quality', '?')}")

        # Agent 4 — Risk Fusion
        if verbose: print("  🔗 [Agent 4] Risk Fusion (combining all modalities)...")
        fusion = agent_fusion(
            demographic=demographic,
            clinical=clinical,
            left_eye=left_eye,
            right_eye=right_eye,
            age=age,
            sex=sex,
            model_key=model_key,
        )
        if verbose:
            print(f"     → Risk Level: {fusion.get('overall_diabetes_risk_level', '?')}")
            print(f"     → Risk Score: {fusion.get('diabetes_risk_score', '?')}/100")
            print(f"     → Confidence: {fusion.get('confidence', '?')}")

        # Agent 5 — Prevention
        if verbose: print("  💊 [Agent 5] Prevention Recommender...")
        has_dr = (left_eye.get("dr", 0) == 1) or (right_eye.get("dr", 0) == 1)
        prevention = agent_prevention(
            fusion=fusion,
            age=age,
            sex=sex,
            has_dr=has_dr,
            model_key=model_key,
        )
        if verbose:
            print(f"     → Tier: {prevention.get('tier_priority', '?')} ({prevention.get('tier_name', '?')})")
            print("  ✅ Assessment complete!")

        # Package final result
        return {
            "patient":     {"age": age, "sex": sex},
            "model_used":  model_key,
            "demographic": demographic,
            "clinical":    clinical,
            "left_eye":    left_eye,
            "right_eye":   right_eye,
            "fusion":      fusion,
            "prevention":  prevention,
        }

    except Exception as e:
        if verbose:
            print(f"❌ Assessment error: {e}")
            import traceback
            traceback.print_exc()
        return {"error": str(e)}
