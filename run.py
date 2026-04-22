"""
run.py
======
Quick entry point for testing the system on a single patient.
Run this first to verify everything works before the full benchmark.

Usage:
    python run.py                    # test first patient with default model (qwen)
    python run.py 5 llama            # test patient #5 with llama
    python run.py 10 gemma           # test patient #10 with gemma
"""

import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv

# Load API key from .env file if present
load_dotenv()

from agents.workflow import run_assessment
from config import CSV_PATH, IMAGE_DIR, DEFAULT_MODEL, MODELS


def main():
    # Parse arguments
    patient_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model_key   = sys.argv[2]      if len(sys.argv) > 2 else DEFAULT_MODEL

    # Validate model key
    if model_key not in MODELS:
        print(f"❌ Unknown model: {model_key}")
        print(f"   Available: {list(MODELS.keys())}")
        sys.exit(1)

    # Check dataset exists
    if not os.path.exists(CSV_PATH):
        print(f"❌ Dataset not found: {CSV_PATH}")
        print("   Please copy OIA-ODIR-Merged/ into Final_Project/")
        sys.exit(1)

    # Load patient
    df  = pd.read_csv(CSV_PATH)
    row = df.iloc[patient_idx]

    print("\n" + "="*60)
    print("🏥 MULTIMODAL AGENTIC DIABETES SCREENING")
    print("="*60)
    print(f"Patient #{patient_idx}: Age={row.get('Patient Age','?')} | Sex={row.get('Patient Sex','?')}")
    print(f"Model: {model_key.upper()} ({MODELS[model_key]})")
    print(f"Left:  {row.get('Left-Diagnostic Keywords','N/A')}")
    print(f"Right: {row.get('Right-Diagnostic Keywords','N/A')}")
    print("="*60)

    # Run assessment
    result = run_assessment(
        row       = row.to_dict(),
        image_dir = IMAGE_DIR,
        model_key = model_key,
    )

    # Print key results
    fusion     = result.get("fusion", {})
    prevention = result.get("prevention", {})

    print("\n📊 RESULT:")
    print(f"  Risk Level : {fusion.get('overall_diabetes_risk_level', '?').upper()}")
    print(f"  Risk Score : {fusion.get('diabetes_risk_score', '?')}/100")
    print(f"  Confidence : {fusion.get('confidence', '?')}")
    print(f"  DR Left    : {result.get('left_eye',{}).get('dr','?')}")
    print(f"  DR Right   : {result.get('right_eye',{}).get('dr','?')}")
    print(f"  Tier       : {prevention.get('tier_priority','?')} — {prevention.get('tier_name','?')}")
    print(f"  Summary    : {fusion.get('multimodal_summary','N/A')}")

    # Save to file
    out_path = f"results/test_patient{patient_idx}_{model_key}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n✅ Full result saved to: {out_path}")


if __name__ == "__main__":
    main()
