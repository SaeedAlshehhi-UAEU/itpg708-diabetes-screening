"""
run_benchmark.py - STANDALONE benchmark (no package imports to avoid cache)
Run from Final_Project folder:
    python run_benchmark.py 3
    python run_benchmark.py 200
"""

import sys
import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

# Import our modules fresh
from agents.workflow import run_assessment
from config import MODELS, CSV_PATH, IMAGE_DIR, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("="*70)
    print(f"🏥 BENCHMARK: {num_samples} patients × {len(MODELS)} models")
    print("="*70)

    # Load stratified dataset
    df = pd.read_csv(CSV_PATH)
    df['has_dr'] = df['D'].astype(int)
    print(f"Dataset: {len(df)} | DR+: {df['has_dr'].sum()} | DR-: {(df['has_dr']==0).sum()}")

    half = num_samples // 2
    dr_pos = df[df['has_dr']==1].sample(n=min(half, df['has_dr'].sum()), random_state=42)
    dr_neg = df[df['has_dr']==0].sample(n=num_samples - len(dr_pos), random_state=42)
    sample = pd.concat([dr_pos, dr_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Sample: {len(sample)} | DR+: {sample['has_dr'].sum()} | DR-: {(sample['has_dr']==0).sum()}\n")

    # Results storage
    all_results = {mk: [] for mk in MODELS.keys()}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process each patient with each model
    for idx, (_, row) in enumerate(sample.iterrows()):
        print(f"\n{'─'*70}")
        print(f"Patient {idx+1}/{len(sample)} | Age={row['Patient Age']} | Sex={row['Patient Sex']} | GT_DR={row['has_dr']}")
        print(f"{'─'*70}")

        for model_key in MODELS.keys():
            print(f"\n🤖 Model: {model_key}")
            t0 = time.time()
            try:
                result = run_assessment(
                    row=row.to_dict(),
                    image_dir=IMAGE_DIR,
                    model_key=model_key,
                    verbose=True,
                )
                elapsed = time.time() - t0

                fusion = result.get("fusion", {})
                risk_level = fusion.get("overall_diabetes_risk_level", "unknown")
                risk_score = float(fusion.get("diabetes_risk_score", 0.0))
                pred_dr = 1 if "high" in str(risk_level).lower() else 0

                # Check if ANY agent had errors
                has_errors = any(
                    "error" in result.get(k, {})
                    for k in ["demographic", "clinical", "left_eye", "right_eye", "fusion"]
                )

                all_results[model_key].append({
                    "patient_id": int(row.get("ID", idx)),
                    "model": model_key,
                    "ground_truth": int(row['has_dr']),
                    "predicted_risk_level": risk_level,
                    "predicted_risk_score": risk_score,
                    "predicted_dr": pred_dr,
                    "confidence": float(fusion.get("confidence", 0.0)),
                    "success": not has_errors,
                    "time_seconds": round(elapsed, 2),
                    "error": "agent errors" if has_errors else None,
                })

                print(f"     Result: Risk={risk_level} Score={risk_score:.1f} Pred={pred_dr} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"     ❌ {e}")
                all_results[model_key].append({
                    "patient_id": int(row.get("ID", idx)),
                    "model": model_key,
                    "ground_truth": int(row['has_dr']),
                    "predicted_risk_level": "error",
                    "predicted_risk_score": 0.0,
                    "predicted_dr": 0,
                    "confidence": 0.0,
                    "success": False,
                    "time_seconds": round(time.time()-t0, 2),
                    "error": str(e),
                })

        # Incremental save every 5 patients
        if (idx+1) % 5 == 0:
            save_results(all_results, timestamp, final=False)

    # Final save & metrics
    save_results(all_results, timestamp, final=True)


def save_results(all_results, timestamp, final=False):
    """Save predictions + compute metrics."""
    # Flatten all predictions into CSV
    rows = []
    for mk, results in all_results.items():
        for r in results:
            r2 = dict(r)
            r2["model_key"] = mk
            rows.append(r2)

    suffix = "" if final else "_INCR"
    csv_path = os.path.join(OUTPUT_DIR, f"predictions_{timestamp}{suffix}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n💾 Saved: {csv_path}")

    if not final:
        return

    # Compute metrics per model
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)

    all_metrics = {}
    for mk, results in all_results.items():
        successful = [r for r in results if r["success"]]
        if not successful:
            print(f"\n{mk}: No successful predictions")
            all_metrics[mk] = {"error": "No success", "total": len(results)}
            continue

        y_true = [r["ground_truth"] for r in successful]
        y_pred = [r["predicted_dr"] for r in successful]
        scores = [r["predicted_risk_score"] for r in successful]

        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()

        try:
            auc = roc_auc_score(y_true, scores)
        except:
            auc = 0.0

        metrics = {
            "accuracy":         round(accuracy_score(y_true, y_pred), 4),
            "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score":         round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc":          round(auc, 4),
            "true_positives":   int(tp),
            "true_negatives":   int(tn),
            "false_positives":  int(fp),
            "false_negatives":  int(fn),
            "total":            len(results),
            "successful":       len(successful),
            "avg_time_seconds": round(float(np.mean([r["time_seconds"] for r in successful])), 2),
        }
        all_metrics[mk] = metrics

        print(f"\n{mk.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1_score']:.4f}")
        print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
        print(f"  Success: {len(successful)}/{len(results)}")

    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{timestamp}.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n✅ All results in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
