"""
app/app.py
==========
Streamlit Web Application
Multimodal Agentic Diabetes Screening System
ITPG 708 — Spring 2026 | UAEU

Four tabs:
  1. Demo Patient    — Select from OIA-ODIR dataset
  2. New Patient     — Upload custom fundus images
  3. Benchmark       — View/compare model performance
  4. Analytics       — Session assessment history

Run with:
  streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import tempfile
from datetime import datetime
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & imports
# ---------------------------------------------------------------------------
load_dotenv()

# Make sure project root is on sys.path so we can import agents/ and config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.workflow import run_assessment
from config import MODELS, CSV_PATH, IMAGE_DIR, OUTPUT_DIR, DEFAULT_MODEL

# Friendly display names for model keys
MODEL_DISPLAY_NAMES = {
    "claude_sonnet": "Claude Sonnet 4.6 (Anthropic)",
    "gemini_25":     "Gemini 2.5 Flash (Google)",
    "gpt4o":         "GPT-4o (OpenAI)",
    "gemma_4":       "Gemma 4 31B (Google, open-weight)",
    "qwen_vl":       "Qwen 3 VL 8B (Alibaba, open-weight)",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Screening AI — ITPG 708",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: bold;
        background: linear-gradient(135deg, #1f77b4 0%, #0d47a1 100%);
        color: white;
        text-align: center;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stAlert { border-radius: 8px; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "assessment_history" not in st.session_state:
    st.session_state.assessment_history = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ System Info")

    st.info(
        "**5-Agent Pipeline**\n\n"
        "1. 👤 Demographic Risk\n"
        "2. 📝 Clinical NLP\n"
        "3. 🔵 Left Eye Vision\n"
        "4. 🔴 Right Eye Vision\n"
        "5. 🔗 Risk Fusion\n"
        "6. 💊 Prevention Plan"
    )

    # API key status
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        st.success("✅ API key loaded")
    else:
        st.error("❌ No API key — set `OPENROUTER_API_KEY` in `.env`")

    # Dataset status
    if os.path.exists(CSV_PATH):
        st.success("✅ Dataset found")
    else:
        st.warning(f"⚠️ Dataset missing at {CSV_PATH}")

    st.warning(
        "⚠️ **DISCLAIMER**\n\n"
        "For **educational use only**.\n"
        "NOT suitable for medical diagnosis."
    )

    st.markdown("---")
    st.markdown(f"**Session Assessments:** {len(st.session_state.assessment_history)}")
    if st.button("🗑️ Clear History"):
        st.session_state.assessment_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="main-header">🏥 Multimodal Agentic Diabetes Screening</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "**ITPG 708 — Spring 2026 | UAEU** | "
    "Vision-Language AI for Diabetic Retinopathy Detection"
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_risk_emoji(risk_level: str) -> str:
    """Return an emoji matching the risk level text."""
    text = str(risk_level or "").lower()
    if "high" in text:
        return "🔴"
    if "moderate" in text or "medium" in text:
        return "🟡"
    if "low" in text:
        return "🟢"
    return "⚪"


def fmt_number(value, decimals: int = 2, fallback: str = "N/A") -> str:
    """Safely format a value as a number, or return fallback."""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return fallback


def display_results(assessment: dict, patient_id: str = "N/A", model_key: str = "?") -> None:
    """Render a full assessment report in the UI."""
    if not assessment or not isinstance(assessment, dict) or "error" in assessment:
        st.error(f"❌ Assessment failed: {assessment.get('error', 'Unknown error') if isinstance(assessment, dict) else 'Invalid response'}")
        return

    # Save to session history
    st.session_state.assessment_history.append({
        "timestamp": datetime.now(),
        "patient_id": patient_id,
        "model": model_key,
        "assessment": assessment,
    })

    fusion     = assessment.get("fusion", {}) or {}
    prevention = assessment.get("prevention", {}) or {}
    left_eye   = assessment.get("left_eye", {}) or {}
    right_eye  = assessment.get("right_eye", {}) or {}
    clinical   = assessment.get("clinical", {}) or {}
    demo       = assessment.get("demographic", {}) or {}

    risk_level = fusion.get("overall_diabetes_risk_level", "unknown")
    risk_score = fusion.get("diabetes_risk_score", 0)
    confidence = fusion.get("confidence", 0)

    # ========= Top summary =========
    st.markdown("## 📊 Assessment Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Risk Level", f"{get_risk_emoji(risk_level)} {str(risk_level).upper()}")
    with c2:
        st.metric("Risk Score", f"{fmt_number(risk_score, 1)}/100")
    with c3:
        # Confidence may be 0-1 or 0-100 depending on model
        try:
            conf_val = float(confidence)
            conf_display = f"{conf_val * 100:.0f}%" if conf_val <= 1 else f"{conf_val:.0f}%"
        except (TypeError, ValueError):
            conf_display = "N/A"
        st.metric("Confidence", conf_display)
    with c4:
        st.metric("Patient", patient_id)

    if fusion.get("multimodal_summary"):
        st.info(f"📋 **Summary:** {fusion['multimodal_summary']}")
    if fusion.get("risk_reasoning"):
        with st.expander("🧠 Clinical Reasoning"):
            st.write(fusion["risk_reasoning"])

    st.markdown("---")

    # ========= Agent outputs =========
    st.markdown("## 🤖 Agent Pipeline Results")

    col_left, col_right = st.columns(2)

    with col_left:
        with st.expander("👤 Agent 1 — Demographic Risk", expanded=False):
            st.write(f"**Score:** {demo.get('demographic_risk_score', 'N/A')}")
            st.write(f"**Age Level:** {demo.get('age_risk_level', 'N/A')}")
            if demo.get("age_risk_reason"):
                st.write(f"**Reasoning:** {demo['age_risk_reason']}")
            factors = demo.get("key_factors") or []
            if factors:
                st.write(f"**Key Factors:** {', '.join(factors)}")

        with st.expander("📝 Agent 2 — Clinical NLP", expanded=False):
            st.write(f"**Clinical Score:** {clinical.get('text_derived_risk_score', 'N/A')}")
            if clinical.get("clinical_summary"):
                st.write(f"**Summary:** {clinical['clinical_summary']}")
            findings = clinical.get("diabetic_findings") or []
            if findings:
                st.write(f"**Findings:** {', '.join(findings)}")

    with col_right:
        with st.expander("🔵 Agent 3a — Left Eye", expanded=False):
            dr = left_eye.get("dr", -1)
            dr_label = "✅ YES" if dr == 1 else ("❌ NO" if dr == 0 else "❓ Unknown")
            st.write(f"**DR Detected:** {dr_label}")
            st.write(f"**Severity:** {left_eye.get('dr_severity', 'N/A')}")
            st.write(f"**Confidence:** {fmt_number(left_eye.get('dr_confidence'), 2)}")
            st.write(f"**Quality:** {left_eye.get('image_quality', 'N/A')}")
            findings = left_eye.get("key_findings") or []
            if findings:
                st.write(f"**Findings:** {', '.join(findings)}")

        with st.expander("🔴 Agent 3b — Right Eye", expanded=False):
            dr = right_eye.get("dr", -1)
            dr_label = "✅ YES" if dr == 1 else ("❌ NO" if dr == 0 else "❓ Unknown")
            st.write(f"**DR Detected:** {dr_label}")
            st.write(f"**Severity:** {right_eye.get('dr_severity', 'N/A')}")
            st.write(f"**Confidence:** {fmt_number(right_eye.get('dr_confidence'), 2)}")
            st.write(f"**Quality:** {right_eye.get('image_quality', 'N/A')}")
            findings = right_eye.get("key_findings") or []
            if findings:
                st.write(f"**Findings:** {', '.join(findings)}")

    st.markdown("---")

    # ========= Prevention plan =========
    st.markdown("## 💊 Prevention Plan")

    tier      = prevention.get("tier_priority", "?")
    tier_name = prevention.get("tier_name", "?")
    plan      = prevention.get("prevention_plan", {}) or {}
    monitor   = prevention.get("monitoring_plan", {}) or {}

    st.markdown(f"### Tier {tier} — {tier_name}")

    if prevention.get("implementation_summary"):
        st.success(f"**Action Plan:** {prevention['implementation_summary']}")

    col_a, col_b = st.columns(2)
    with col_a:
        lifestyle = plan.get("lifestyle_modifications") or []
        if lifestyle:
            st.markdown("**🏃 Lifestyle Modifications:**")
            for item in lifestyle:
                st.markdown(f"- {item}")

        interventions = plan.get("risk_interventions") or []
        if interventions:
            st.markdown("**🎯 Risk Interventions:**")
            for item in interventions:
                st.markdown(f"- {item}")

    with col_b:
        screening = plan.get("medical_screening", {}) or {}
        if screening:
            st.markdown("**🏥 Medical Screening:**")
            st.write(f"Frequency: {screening.get('screening_frequency', 'N/A')}")
            tests = screening.get("recommended_tests") or []
            if tests:
                st.write(f"Tests: {', '.join(tests)}")
            refs = screening.get("specialist_referrals") or []
            if refs:
                st.write(f"Referrals: {', '.join(refs)}")

        if monitor:
            st.markdown("**📅 Monitoring:**")
            st.write(f"Self-monitoring: {monitor.get('self_monitoring_frequency', 'N/A')}")
            st.write(f"Follow-up: {monitor.get('follow_up_schedule', 'N/A')}")

    # ========= Export options =========
    st.markdown("---")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        json_str = json.dumps(assessment, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"assessment_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
    with exp_col2:
        csv_df = pd.DataFrame([{
            "Patient ID": patient_id,
            "Model": model_key,
            "Risk Level": risk_level,
            "Risk Score": risk_score,
            "Confidence": confidence,
            "DR Left": left_eye.get("dr"),
            "DR Right": right_eye.get("dr"),
            "Tier": tier,
            "Timestamp": datetime.now().isoformat(),
        }])
        st.download_button(
            label="📊 Download CSV",
            data=csv_df.to_csv(index=False),
            file_name=f"assessment_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    with st.expander("🔧 Raw JSON Data"):
        st.json(assessment)


# ---------------------------------------------------------------------------
# TAB 1 — Demo Patient
# ---------------------------------------------------------------------------
def tab_demo() -> None:
    st.markdown("### 📂 Demo Patient — Select from OIA-ODIR Dataset")
    st.info(
        "Choose a patient from the 5,000-patient OIA-ODIR dataset. "
        "The system will load their bilateral fundus images and clinical "
        "keywords and run the full 5-agent assessment."
    )

    if not os.path.exists(CSV_PATH):
        st.error(
            f"❌ Dataset not found at `{CSV_PATH}`. "
            f"Please download OIA-ODIR and place it under `OIA-ODIR-Merged/`."
        )
        return

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        st.error(f"❌ Failed to read dataset: {e}")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        patient_idx = st.slider(
            "Select Patient Index",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            help="Any patient from the 5,000-patient OIA-ODIR dataset"
        )
    with col2:
        model_key = st.selectbox(
            "Select LLM Model",
            options=list(MODELS.keys()),
            format_func=lambda k: MODEL_DISPLAY_NAMES.get(k, k),
            index=list(MODELS.keys()).index(DEFAULT_MODEL) if DEFAULT_MODEL in MODELS else 0,
            key="demo_model",
        )

    row = df.iloc[patient_idx]
    patient_id = str(row.get("ID", f"idx_{patient_idx}"))

    # Patient info card
    st.markdown("**📋 Patient Information**")
    info_c1, info_c2, info_c3 = st.columns(3)
    with info_c1:
        st.write(f"**ID:** {patient_id}")
        st.write(f"**Age:** {row.get('Patient Age', 'N/A')}")
        st.write(f"**Sex:** {row.get('Patient Sex', 'N/A')}")
    with info_c2:
        st.write(f"**Left Keywords:**")
        st.caption(str(row.get("Left-Diagnostic Keywords", "N/A")))
    with info_c3:
        st.write(f"**Right Keywords:**")
        st.caption(str(row.get("Right-Diagnostic Keywords", "N/A")))

    # Ground truth
    if "D" in df.columns:
        gt = int(row.get("D", 0))
        st.markdown(
            f"**Ground Truth (DR):** "
            f"{'🔴 YES (DR present)' if gt == 1 else '🟢 NO (No DR)'}"
        )

    # Images
    left_path  = os.path.join(IMAGE_DIR, str(row.get("Left-Fundus", "")))
    right_path = os.path.join(IMAGE_DIR, str(row.get("Right-Fundus", "")))

    img_c1, img_c2 = st.columns(2)
    with img_c1:
        if os.path.exists(left_path):
            st.image(left_path, caption="👁️ Left Eye Fundus", width=300)
        else:
            st.warning(f"Left image not found: {left_path}")
    with img_c2:
        if os.path.exists(right_path):
            st.image(right_path, caption="👁️ Right Eye Fundus", width=300)
        else:
            st.warning(f"Right image not found: {right_path}")

    # Run button
    if st.button("🔍 Run 5-Agent Assessment", type="primary", use_container_width=True, key="demo_run"):
        if not os.getenv("OPENROUTER_API_KEY"):
            st.error("❌ No API key configured. Please set `OPENROUTER_API_KEY` in `.env`.")
            return

        with st.spinner(f"⏳ Running 5-agent pipeline with {MODEL_DISPLAY_NAMES.get(model_key, model_key)}..."):
            try:
                assessment = run_assessment(
                    row=row.to_dict(),
                    image_dir=IMAGE_DIR,
                    model_key=model_key,
                    verbose=False,
                )
                display_results(assessment, patient_id=f"Patient_{patient_id}", model_key=model_key)
            except Exception as e:
                st.error(f"❌ Pipeline error: {e}")


# ---------------------------------------------------------------------------
# TAB 2 — New Patient
# ---------------------------------------------------------------------------
def tab_new_patient() -> None:
    st.markdown("### 📸 New Patient — Upload Your Own Fundus Images")
    st.info(
        "Upload bilateral fundus images (JPG/PNG) and enter patient demographics. "
        "The system will run the full 5-agent multimodal assessment."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**👤 Patient Information**")
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
        sex = st.selectbox("Sex", ["Male", "Female"])
        left_keywords = st.text_input(
            "Left Eye Keywords (optional)",
            placeholder="e.g., microaneurysms, mild hemorrhage",
        )
        right_keywords = st.text_input(
            "Right Eye Keywords (optional)",
            placeholder="e.g., moderate non-proliferative DR",
        )
        model_key = st.selectbox(
            "LLM Model",
            options=list(MODELS.keys()),
            format_func=lambda k: MODEL_DISPLAY_NAMES.get(k, k),
            index=list(MODELS.keys()).index(DEFAULT_MODEL) if DEFAULT_MODEL in MODELS else 0,
            key="new_model",
        )

    with col2:
        st.markdown("**👁️ Fundus Images**")
        left_upload  = st.file_uploader(
            "Left Eye Image",
            type=["jpg", "jpeg", "png"],
            key="left_upload"
        )
        right_upload = st.file_uploader(
            "Right Eye Image",
            type=["jpg", "jpeg", "png"],
            key="right_upload"
        )
        if left_upload:
            st.image(left_upload, caption="Left Eye", use_container_width=True)
        if right_upload:
            st.image(right_upload, caption="Right Eye", use_container_width=True)

    if st.button("🔍 Run 5-Agent Assessment", type="primary", use_container_width=True, key="new_run"):
        if not left_upload or not right_upload:
            st.error("❌ Please upload both left and right eye images.")
            return
        if not os.getenv("OPENROUTER_API_KEY"):
            st.error("❌ No API key configured. Please set `OPENROUTER_API_KEY` in `.env`.")
            return

        with st.spinner(f"⏳ Running pipeline with {MODEL_DISPLAY_NAMES.get(model_key, model_key)}..."):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    left_path  = os.path.join(tmpdir, "left.jpg")
                    right_path = os.path.join(tmpdir, "right.jpg")
                    with open(left_path, "wb") as f:
                        f.write(left_upload.getvalue())
                    with open(right_path, "wb") as f:
                        f.write(right_upload.getvalue())

                    row = {
                        "Patient Age": age,
                        "Patient Sex": sex,
                        "Left-Fundus": "left.jpg",
                        "Right-Fundus": "right.jpg",
                        "Left-Diagnostic Keywords": left_keywords,
                        "Right-Diagnostic Keywords": right_keywords,
                    }

                    assessment = run_assessment(
                        row=row,
                        image_dir=tmpdir,
                        model_key=model_key,
                        verbose=False,
                    )
                    display_results(assessment, patient_id=f"Custom_{age}yo", model_key=model_key)
            except Exception as e:
                st.error(f"❌ Pipeline error: {e}")


# ---------------------------------------------------------------------------
# TAB 3 — Benchmark
# ---------------------------------------------------------------------------
def tab_benchmark() -> None:
    st.markdown("### 📊 Model Benchmark Results")
    st.info(
        "Compare the diagnostic performance of all 5 LLMs on the stratified "
        "OIA-ODIR sample (100 DR-positive + 100 DR-negative patients)."
    )

    if not os.path.exists(OUTPUT_DIR):
        st.warning("⚠️ No results directory yet. Run the benchmark first.")
        st.code("python run_benchmark.py 200", language="bash")
        return

    # Find metrics files (look in results/ and subdirs)
    metrics_files = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for fname in files:
            if fname.startswith("metrics_") and fname.endswith(".json"):
                metrics_files.append(os.path.join(root, fname))

    if not metrics_files:
        st.warning("⚠️ No benchmark metrics found yet.")
        st.code("python run_benchmark.py 200", language="bash")
        return

    # Let user pick which run to view
    metrics_files = sorted(metrics_files, reverse=True)
    if len(metrics_files) > 1:
        selected = st.selectbox(
            "Select benchmark run",
            metrics_files,
            format_func=lambda p: os.path.basename(p),
        )
    else:
        selected = metrics_files[0]

    try:
        with open(selected) as f:
            metrics = json.load(f)
    except Exception as e:
        st.error(f"Failed to load {selected}: {e}")
        return

    st.success(f"✅ Loaded: `{os.path.basename(selected)}`")

    # Build metrics table
    rows = []
    for model_key, m in metrics.items():
        if not isinstance(m, dict) or "error" in m:
            continue
        rows.append({
            "Model":      MODEL_DISPLAY_NAMES.get(model_key, model_key),
            "Accuracy":   m.get("accuracy", 0),
            "Precision":  m.get("precision", 0),
            "Recall":     m.get("recall", 0),
            "F1-Score":   m.get("f1_score", 0),
            "ROC-AUC":    m.get("roc_auc", 0),
            "Success":    f"{m.get('successful', 0)}/{m.get('total', 0)}",
            "Avg Time":   f"{m.get('avg_time_seconds', 0):.1f}s",
            "TP":         m.get("true_positives", 0),
            "TN":         m.get("true_negatives", 0),
            "FP":         m.get("false_positives", 0),
            "FN":         m.get("false_negatives", 0),
        })

    if not rows:
        st.warning("No successful model results in this file.")
        return

    df_display = pd.DataFrame(rows)

    # Format numbers for display
    numeric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    for col in numeric_cols:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")

    st.dataframe(df_display, use_container_width=True)

    # Bar chart of core metrics
    st.markdown("#### 📈 Metrics Comparison")
    chart_df = pd.DataFrame([
        {
            "Model": r["Model"].split(" (")[0],  # Short name
            "Accuracy":  float(r["Accuracy"]),
            "Precision": float(r["Precision"]),
            "Recall":    float(r["Recall"]),
            "F1-Score":  float(r["F1-Score"]),
        }
        for r in rows
    ]).set_index("Model")
    st.bar_chart(chart_df)

    # Confusion matrix highlights
    st.markdown("#### 🎯 Per-Model Confusion")
    conf_df = pd.DataFrame([
        {"Model": r["Model"].split(" (")[0], "TP": r["TP"], "TN": r["TN"], "FP": r["FP"], "FN": r["FN"]}
        for r in rows
    ]).set_index("Model")
    st.bar_chart(conf_df)


# ---------------------------------------------------------------------------
# TAB 4 — Analytics
# ---------------------------------------------------------------------------
def tab_analytics() -> None:
    st.markdown("### 📈 Session Analytics")

    history = st.session_state.assessment_history
    if not history:
        st.info(
            "No assessments run in this session yet. "
            "Go to the **Demo Patient** or **New Patient** tab to start."
        )
        return

    records = []
    for rec in history:
        fusion = rec["assessment"].get("fusion", {}) or {}
        records.append({
            "Timestamp":  rec["timestamp"].strftime("%H:%M:%S"),
            "Patient":    rec["patient_id"],
            "Model":      MODEL_DISPLAY_NAMES.get(rec.get("model", "?"), rec.get("model", "?")),
            "Risk Level": str(fusion.get("overall_diabetes_risk_level", "N/A")).upper(),
            "Score":      fusion.get("diabetes_risk_score", 0),
            "Confidence": fusion.get("confidence", 0),
        })

    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True)

    # Risk distribution
    if len(df) > 0:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Risk Level Distribution**")
            st.bar_chart(df["Risk Level"].value_counts())
        with c2:
            st.markdown("**Score Distribution**")
            st.line_chart(df["Score"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Demo Patient",
    "📸 New Patient",
    "📊 Benchmark",
    "📈 Analytics",
])

with tab1:
    tab_demo()

with tab2:
    tab_new_patient()

with tab3:
    tab_benchmark()

with tab4:
    tab_analytics()

# Footer
st.markdown("---")
st.markdown(
    "*ITPG 708 — Multimodal Agentic Diabetes Screening | Spring 2026 | UAEU*  \n"
    "*Saeed M. Alshehhi, Maitha A. Al Hayyas, Fatima M. Alnuaimi*  \n"
    "**⚠️ Educational/research use only. NOT suitable for medical diagnosis.**"
)
