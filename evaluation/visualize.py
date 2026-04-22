"""
evaluation/visualize.py
========================
Generate charts and visualizations from benchmark results.
Run after benchmark.py to create charts for the report.

Usage:
    python -m evaluation.visualize
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from config import OUTPUT_DIR

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)


def load_latest_results():
    """Load the most recent predictions CSV and metrics JSON from results/."""
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ Results directory not found: {OUTPUT_DIR}")
        return None, None

    csv_files  = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("predictions_") and f.endswith(".csv")])
    json_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("metrics_") and f.endswith(".json")])

    if not csv_files:
        print("❌ No prediction CSV files found. Run benchmark first.")
        return None, None

    df      = pd.read_csv(os.path.join(OUTPUT_DIR, csv_files[-1]))
    metrics = None
    if json_files:
        with open(os.path.join(OUTPUT_DIR, json_files[-1])) as f:
            metrics = json.load(f)

    print(f"✓ Loaded: {csv_files[-1]} ({len(df)} records)")
    return df, metrics


def plot_metrics_comparison(metrics: dict):
    """Bar chart comparing accuracy, precision, recall, F1 across models."""
    if not metrics:
        return

    models       = list(metrics.keys())
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    colors       = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for idx, metric in enumerate(metric_names):
        values = [metrics[m].get(metric, 0) for m in models]
        bars   = axes[idx].bar(models, values, color=colors[:len(models)], alpha=0.85, edgecolor="black")
        axes[idx].set_title(metric.replace("_", " ").upper(), fontweight="bold")
        axes[idx].set_ylim(0, 1.1)
        axes[idx].set_ylabel("Score")
        axes[idx].tick_params(axis="x", rotation=30)
        for bar in bars:
            h = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.3f}", ha="center", fontweight="bold", fontsize=10)

    plt.suptitle("Model Performance Comparison — Diabetes DR Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "01_metrics_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {path}")
    plt.close()


def plot_confusion_matrices(df: pd.DataFrame):
    """Confusion matrix heatmap for each model."""
    if "model_key" not in df.columns:
        print("⚠ No model_key column found")
        return

    models  = df["model_key"].unique()
    n       = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        sub    = df[df["model_key"] == model]
        y_true = sub["ground_truth"]
        y_pred = sub["predicted_dr"]
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=["No DR", "DR"], yticklabels=["No DR", "DR"],
                    cbar=False, annot_kws={"size": 14, "weight": "bold"})
        axes[idx].set_title(model.upper(), fontweight="bold", fontsize=12)
        axes[idx].set_ylabel("Ground Truth")
        axes[idx].set_xlabel("Predicted")

    plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "02_confusion_matrices.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {path}")
    plt.close()


def plot_risk_distribution(df: pd.DataFrame):
    """Distribution of predicted risk levels per model."""
    if "model_key" not in df.columns or "predicted_risk_level" not in df.columns:
        return

    models = df["model_key"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    colors = {"low": "#2ecc71", "moderate": "#f39c12", "high": "#e74c3c", "unknown": "#95a5a6"}

    for idx, model in enumerate(models):
        sub    = df[df["model_key"] == model]
        counts = sub["predicted_risk_level"].value_counts()
        bar_colors = [colors.get(str(k).lower(), "#95a5a6") for k in counts.index]
        axes[idx].bar(counts.index, counts.values, color=bar_colors, edgecolor="black", alpha=0.85)
        axes[idx].set_title(model.upper(), fontweight="bold")
        axes[idx].set_ylabel("Count")
        axes[idx].tick_params(axis="x", rotation=30)

    plt.suptitle("Predicted Risk Level Distribution — All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "03_risk_distribution.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {path}")
    plt.close()


def plot_model_agreement(df: pd.DataFrame):
    """Pie chart showing how often all 3 models agree."""
    if "model_key" not in df.columns:
        return

    pivot = df.pivot_table(index="patient_id", columns="model_key", values="predicted_dr", aggfunc="first")
    if len(pivot.columns) < 2:
        return

    agree    = (pivot.nunique(axis=1) == 1).sum()
    disagree = len(pivot) - agree
    total    = len(pivot)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        [agree, disagree],
        labels=[f"Agree\n({agree}/{total})", f"Disagree\n({disagree}/{total})"],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
        textprops={"fontsize": 12},
    )
    for a in autotexts:
        a.set_color("white")
        a.set_fontweight("bold")
        a.set_fontsize(13)

    ax.set_title("Model Agreement on DR Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "04_model_agreement.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {path}")
    plt.close()


def plot_timing(metrics: dict):
    """Bar chart of average inference time per model."""
    if not metrics:
        return

    models = list(metrics.keys())
    times  = [metrics[m].get("avg_time_seconds", 0) for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, times, color=["#4C72B0", "#DD8452", "#55A868"][:len(models)], alpha=0.85, edgecolor="black")
    ax.set_ylabel("Average Time (seconds)")
    ax.set_title("Average Inference Time per Patient — All Models", fontsize=13, fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.2f}s", ha="center", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "05_inference_time.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {path}")
    plt.close()


def run_visualizations():
    """Run all visualizations from latest benchmark results."""
    print("\n" + "="*60)
    print("📊 GENERATING VISUALIZATIONS FROM BENCHMARK RESULTS")
    print("="*60)

    df, metrics = load_latest_results()
    if df is None:
        return

    plot_metrics_comparison(metrics)
    plot_confusion_matrices(df)
    plot_risk_distribution(df)
    plot_model_agreement(df)
    plot_timing(metrics)

    print(f"\n✅ All charts saved to: {VIZ_DIR}/")


if __name__ == "__main__":
    run_visualizations()
