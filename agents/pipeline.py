"""
agents/pipeline.py
===================
5-Agent Multimodal Pipeline for Diabetes Risk & Prevention

Each agent handles one specific modality or task:
  Agent 1 - Demographic Risk    : age + sex → risk score
  Agent 2 - Clinical NLP        : text keywords → findings
  Agent 3a - Left Eye Vision    : fundus image → DR status
  Agent 3b - Right Eye Vision   : fundus image → DR status
  Agent 4 - Risk Fusion         : all signals → unified risk
  Agent 5 - Prevention          : risk → personalized plan
"""

import os
import json
import base64
import re
from openai import OpenAI
from typing import Dict, Any, Optional
from config import MODELS, DEFAULT_MODEL, HIGH_RISK_THRESHOLD, MOD_RISK_THRESHOLD


# =============================================================================
# Client & Utilities
# =============================================================================

def get_client() -> OpenAI:
    """Initialize OpenAI-compatible client pointing to OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def resolve_model(model_key: str) -> str:
    """Convert short model key ('gpt4o_mini') to full OpenRouter model string."""
    return MODELS.get(model_key, model_key)


def call_llm(model_key: str, messages: list, max_tokens: int = 2000) -> str:
    """
    Unified LLM call via OpenRouter.
    Uses higher max_tokens (2000) to avoid truncated JSON.
    """
    client   = get_client()
    model    = resolve_model(model_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def image_to_base64(image_path: str) -> Optional[str]:
    """Convert local image file to base64 data URL for API transmission."""
    if not os.path.exists(image_path):
        return None
    ext = os.path.splitext(image_path)[1].replace(".", "").lower()
    if ext == "jpg":
        ext = "jpeg"
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
            return f"data:image/{ext};base64,{encoded}"
    except Exception as e:
        print(f"Warning: Image conversion failed for {image_path}: {e}")
        return None


def parse_json(text: str) -> Dict:
    """
    Robust JSON parser that handles various LLM response formats:
    - Raw JSON
    - JSON in ```json ... ``` code blocks (Gemini style)
    - JSON in ``` ... ``` code blocks
    - JSON with surrounding text
    """
    if not text:
        return {"error": "Empty response"}

    text = text.strip()

    # Strategy 1: Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract content from ```json ... ``` or ``` ... ```
    code_block_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]
    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                start = cleaned.find('{')
                end   = cleaned.rfind('}') + 1
                if start != -1 and end > start:
                    try:
                        return json.loads(cleaned[start:end])
                    except json.JSONDecodeError:
                        pass

    # Strategy 3: Find JSON object bounds in raw text
    start = text.find('{')
    end   = text.rfind('}') + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return {"error": "JSON parse failed", "raw": text[:300]}


def normalize_risk_score(score: Any) -> float:
    """
    Normalize risk score to the 0-100 scale.

    Some LLMs (e.g. Gemma, Qwen) sometimes return risk on a 0-1 scale
    instead of 0-100, even when the prompt asks for 0-100. This helper
    detects the scale automatically:
      - If score <= 1.0  → assume 0-1 scale, multiply by 100
      - If score > 1.0   → assume 0-100 scale, keep as-is
      - Clamp final value to [0, 100]
    """
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.0

    # Detect 0-1 scale and rescale to 0-100
    if 0.0 < s <= 1.0:
        s = s * 100.0

    # Clamp to [0, 100]
    return max(0.0, min(100.0, s))


# =============================================================================
# AGENT 1 — Demographic Risk Scorer
# =============================================================================

def agent_demographic(age: int, sex: str, model_key: str = DEFAULT_MODEL) -> Dict:
    """Agent 1: Assess baseline diabetes risk from patient demographics (age, sex)."""
    prompt = f"""You are a diabetes risk specialist using epidemiological data.

Patient Demographics:
- Age: {age} years
- Sex: {sex}

Assess diabetes risk based ONLY on age and sex.
Key facts: risk increases significantly after age 45, males have slightly higher risk.

Return ONLY valid JSON, no markdown, no explanation:
{{
    "demographic_risk_score": 0.35,
    "age_risk_level": "low|moderate|high",
    "age_risk_reason": "brief explanation of age-based risk",
    "sex_risk_factor": "brief note about sex-related risk",
    "key_factors": ["factor1", "factor2"]
}}"""

    try:
        content = call_llm(model_key, [{"role": "user", "content": prompt}])
        result  = parse_json(content)
        result["agent"] = "demographic"
        return result
    except Exception as e:
        return {
            "agent": "demographic", "error": str(e),
            "demographic_risk_score": 0.0, "age_risk_level": "unknown"
        }


# =============================================================================
# AGENT 2 — Clinical NLP Extractor
# =============================================================================

def agent_clinical_nlp(keywords: str, model_key: str = DEFAULT_MODEL) -> Dict:
    """Agent 2: Extract diabetes risk signals from clinical text keywords."""
    prompt = f"""You are a medical NLP specialist extracting diabetes risk signals.

Clinical keywords/description:
{keywords if keywords.strip() else "No clinical description available"}

Identify any findings related to diabetes or diabetic retinopathy.
Examples: microaneurysms, hemorrhages, hard exudates, hypertension, high glucose,
retinal changes, neovascularization.

Return ONLY valid JSON, no markdown:
{{
    "diabetic_findings": ["finding1", "finding2"],
    "metabolic_indicators": ["indicator1"],
    "text_derived_risk_score": 0.2,
    "clinical_summary": "one sentence summary of findings",
    "confidence": 0.8
}}"""

    try:
        content = call_llm(model_key, [{"role": "user", "content": prompt}])
        result  = parse_json(content)
        result["agent"] = "clinical_nlp"
        return result
    except Exception as e:
        return {
            "agent": "clinical_nlp", "error": str(e),
            "text_derived_risk_score": 0.0, "diabetic_findings": []
        }


# =============================================================================
# AGENT 3 — Fundus Image Analyzer
# =============================================================================

def agent_image(image_path: str, eye_side: str, model_key: str = DEFAULT_MODEL) -> Dict:
    """Agent 3: Analyze a fundus image for diabetic retinopathy."""
    if not image_path or not os.path.exists(image_path):
        return {
            "agent": "image_analysis", "eye_side": eye_side,
            "error": f"Image not found: {image_path}",
            "dr": -1, "dr_confidence": 0.0, "image_quality": "unknown"
        }

    data_url = image_to_base64(image_path)
    if not data_url:
        return {
            "agent": "image_analysis", "eye_side": eye_side,
            "error": "Image conversion failed",
            "dr": -1, "dr_confidence": 0.0
        }

    prompt = f"""You are an expert ophthalmologist examining a {eye_side} eye fundus photograph.

Carefully analyze the retinal image for signs of Diabetic Retinopathy (DR):
1. Microaneurysms — small red dots (earliest DR sign)
2. Hemorrhages — larger red blots or flame-shaped lesions
3. Hard exudates — bright yellow deposits from leaking vessels
4. Cotton wool spots — white fluffy patches (nerve fiber infarcts)
5. Neovascularization — abnormal new vessel growth (proliferative DR)

Grade the image:
- No DR, Mild, Moderate, Severe, or Proliferative

Return ONLY valid JSON, no markdown:
{{
    "dr": 0,
    "dr_severity": "no_dr|mild|moderate|severe|proliferative",
    "dr_confidence": 0.85,
    "key_findings": ["finding1", "finding2"],
    "image_quality": "good|fair|poor",
    "clinical_note": "brief one-sentence interpretation"
}}

dr=0 means No DR. dr=1 means DR present."""

    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        }]
        content = call_llm(model_key, messages)
        result  = parse_json(content)

        # Normalize dr to int (0 or 1)
        try:
            result["dr"] = int(result.get("dr", -1))
        except (TypeError, ValueError):
            result["dr"] = -1
        result["agent"]    = "image_analysis"
        result["eye_side"] = eye_side
        return result

    except Exception as e:
        return {
            "agent": "image_analysis", "eye_side": eye_side,
            "error": str(e), "dr": -1, "dr_confidence": 0.0
        }


# =============================================================================
# AGENT 4 — Risk Fusion
# =============================================================================

def agent_fusion(
    demographic: Dict, clinical: Dict, left_eye: Dict, right_eye: Dict,
    age: int, sex: str, model_key: str = DEFAULT_MODEL,
) -> Dict:
    """Agent 4: Fuse all modality signals into one unified diabetes risk assessment."""
    context = f"""You are an AI physician performing multimodal diabetes risk fusion.

Synthesize ALL data sources below into ONE unified assessment:

--- MODALITY 1: STRUCTURED DATA (Demographics) ---
Age: {age} | Sex: {sex}
Demographic Risk Score: {demographic.get('demographic_risk_score', 0.0)}
Risk Level: {demographic.get('age_risk_level', 'unknown')}

--- MODALITY 2: TEXT DATA (Clinical Keywords) ---
Summary: {clinical.get('clinical_summary', 'N/A')}
Diabetic Findings: {clinical.get('diabetic_findings', [])}
Text Risk Score: {clinical.get('text_derived_risk_score', 0.0)}

--- MODALITY 3: VISION DATA (Left Eye Fundus) ---
DR Present: {left_eye.get('dr', -1)}
Severity: {left_eye.get('dr_severity', 'unknown')}
Confidence: {left_eye.get('dr_confidence', 0.0)}
Findings: {left_eye.get('key_findings', [])}

--- MODALITY 4: VISION DATA (Right Eye Fundus) ---
DR Present: {right_eye.get('dr', -1)}
Severity: {right_eye.get('dr_severity', 'unknown')}
Confidence: {right_eye.get('dr_confidence', 0.0)}
Findings: {right_eye.get('key_findings', [])}

FUSION RULES:
1. DR in either eye = strong diabetes indicator → high risk score
2. Age > 45 + DR = high risk almost certainly
3. Clinical keywords mentioning diabetes reinforce DR findings

IMPORTANT: The diabetes_risk_score MUST be on a 0-100 scale (NOT 0-1).
- 0-39   = low risk
- 40-69  = moderate risk
- 70-100 = high risk

Return ONLY valid JSON, no markdown:
{{
    "overall_diabetes_risk_level": "low|moderate|high",
    "diabetes_risk_score": 45.0,
    "confidence": 0.8,
    "risk_reasoning": "2-3 sentence explanation",
    "key_risk_factors": ["factor1", "factor2"],
    "dr": 0,
    "needs_urgent_assessment": false,
    "multimodal_summary": "one sentence synthesis"
}}"""

    try:
        content = call_llm(model_key, [{"role": "user", "content": context}])
        result  = parse_json(content)

        # Normalize risk score to 0-100 scale (fixes Gemma/Qwen returning 0-1)
        if "diabetes_risk_score" in result:
            result["diabetes_risk_score"] = normalize_risk_score(
                result["diabetes_risk_score"]
            )

        result["agent"] = "risk_fusion"
        return result
    except Exception as e:
        return {
            "agent": "risk_fusion", "error": str(e),
            "overall_diabetes_risk_level": "unknown",
            "diabetes_risk_score": 0.0, "dr": -1
        }


# =============================================================================
# AGENT 5 — Prevention Recommender
# =============================================================================

def agent_prevention(
    fusion: Dict, age: int, sex: str, has_dr: bool,
    model_key: str = DEFAULT_MODEL,
) -> Dict:
    """Agent 5: Generate personalized prevention recommendations based on risk tier."""
    risk_level = fusion.get("overall_diabetes_risk_level", "unknown")

    # Normalize risk score to 0-100 scale (handles LLMs that return 0-1)
    risk_score = normalize_risk_score(fusion.get("diabetes_risk_score", 0.0))
    reasoning  = fusion.get("risk_reasoning", "")

    # Also consider the text-based risk level as a fallback
    # If the LLM says "high" but score is low (due to formatting issues), trust the level
    level_lower = str(risk_level).lower()

    # Assign tier using BOTH score AND risk level for robustness
    if risk_score >= HIGH_RISK_THRESHOLD or "high" in level_lower:
        tier, tier_name = 1, "URGENT"
    elif risk_score >= MOD_RISK_THRESHOLD or "moderate" in level_lower or "medium" in level_lower:
        tier, tier_name = 2, "MODERATE"
    else:
        tier, tier_name = 3, "PREVENTIVE"

    prompt = f"""You are a diabetes prevention specialist creating a personalized plan.

PATIENT: Age={age}, Sex={sex}
RISK: {risk_level} (Score: {risk_score:.1f}/100)
HAS DR: {has_dr}
REASONING: {reasoning}
TIER: {tier} — {tier_name}

Tier 1 URGENT   (>=70): Daily monitoring, immediate referrals
Tier 2 MODERATE (>=40): Weekly monitoring, scheduled specialists
Tier 3 PREVENTIVE (<40): Monthly checks, lifestyle guidance

Return ONLY valid JSON, no markdown:
{{
    "tier_priority": {tier},
    "tier_name": "{tier_name}",
    "prevention_plan": {{
        "lifestyle_modifications": ["rec1", "rec2"],
        "medical_screening": {{
            "screening_frequency": "every X months",
            "recommended_tests": ["HbA1c", "glucose"],
            "specialist_referrals": ["endo"]
        }},
        "risk_interventions": ["int1", "int2"]
    }},
    "monitoring_plan": {{
        "self_monitoring_frequency": "daily|weekly|monthly",
        "follow_up_schedule": "description",
        "urgent_warning_signs": ["sign1"]
    }},
    "education_topics": ["topic1"],
    "implementation_summary": "2-3 sentence plan"
}}"""

    try:
        content = call_llm(model_key, [{"role": "user", "content": prompt}])
        result  = parse_json(content)

        # FORCE the correct tier (don't trust the LLM to get it right)
        result["tier_priority"] = tier
        result["tier_name"]     = tier_name

        result["agent"] = "prevention"
        return result
    except Exception as e:
        return {
            "agent": "prevention", "error": str(e),
            "tier_priority": tier, "tier_name": tier_name
        }
