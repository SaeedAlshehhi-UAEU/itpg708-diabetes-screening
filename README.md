# Multimodal Agentic AI for Diabetic Retinopathy Screening: Benchmarking Closed-Source and Open-Weight Vision-Language Models

**ITPG 708 Final Project – Spring 2026 | United Arab Emirates University**

*Educational and research use only. This system is not a certified medical device and must not be used for clinical diagnosis. Always consult qualified healthcare professionals.*

---

## Project Description

A multimodal agentic AI system for diabetes risk assessment and diabetic retinopathy (DR) detection using vision-language large language models (LLMs). The system replaces traditional CNN and tabular classifiers with a pipeline of five specialized LLM agents, orchestrated via LangGraph. It integrates three data modalities from the **same patient** using the OIA-ODIR dataset:

- **Vision** — Bilateral fundus images (left and right eye)
- **Text** — Clinical diagnostic keywords
- **Structured** — Age and sex demographics

The system outputs a unified diabetes risk score, a binary DR prediction, and a personalized prevention plan across three clinical tiers (Urgent, Moderate, Preventive).

**Key contributions:**

1. Unified multimodal pipeline — all modalities originate from one patient (no unpaired datasets).
2. Five-agent architecture orchestrated through LangGraph.
3. Comparative benchmark of five LLMs from four providers (closed-source and open-weight).
4. Tier-based prevention planning that converts probabilistic risk scores into actionable interventions.

---

## Software Requirements

- **Python:** 3.10 or higher (tested on 3.14)
- **Operating System:** Windows, macOS, or Linux
- **Internet:** Required (for API calls to OpenRouter)
- **OpenRouter API Key:** Required (account at https://openrouter.ai/)

### Python Libraries
All listed in `requirements.txt`:

- `openai>=1.0.0` — OpenAI-compatible API client
- `langgraph>=0.0.20` — Agent orchestration
- `langchain>=0.1.0` — LLM framework
- `streamlit>=1.28.0` — Web UI
- `pandas>=2.0.0` — Data handling
- `numpy>=1.24.0` — Numerical computing
- `scikit-learn>=1.3.0` — Evaluation metrics
- `matplotlib>=3.8.0` — Visualization
- `seaborn>=0.13.0` — Statistical plots
- `Pillow>=10.0.0` — Image processing
- `python-dotenv>=1.0.0` — Environment variable loading
- `tqdm>=4.66.0` — Progress bars
- `openpyxl>=3.1.0` — Excel file support (for dataset merging)

---

## Hardware Requirements

- **CPU:** Any modern x86 or ARM processor (no GPU required)
- **RAM:** 8 GB minimum, 16 GB recommended
- **Storage:** ~2 GB for the OIA-ODIR dataset plus ~500 MB for code and outputs
- **Network:** Stable internet connection (the full benchmark makes approximately 6,000 API calls)

No GPU is required — all model inference is performed via cloud APIs.

---

## Installation Instructions

### Step 1 — Clone the Repository
```bash
git clone https://github.com/SaeedAlshehhi-UAEU/itpg708-diabetes-screening.git
cd itpg708-diabetes-screening
```

### Step 2 — Create a Virtual Environment (Recommended)
```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows Command Prompt
python -m venv venv
venv\Scripts\activate.bat
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Configure the API Key
Copy `.env.example` to `.env` and add your OpenRouter API key:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```
Get an API key at https://openrouter.ai/.

### Step 5 — Download and Prepare the OIA-ODIR Dataset

The dataset is available from the official repository: **https://github.com/nkicsl/OIA-ODIR**

The official OIA-ODIR distribution is split into three separate subsets — **Training Set**, **On-site Test Set**, and **Off-site Test Set** — each with its own Excel annotation file and image folder. This project merges these subsets into a single unified dataset to simplify stratified sampling and evaluation.

**Step 5.1 — Download the raw dataset** from the official repository and extract it so the local layout looks like this:

```
Final_Project/
└── OIA-ODIR/
    ├── Training Set/
    │   ├── Annotation/
    │   │   └── ODIR-5K_training(English).xlsx
    │   └── Images/
    ├── Off-site Test Set/
    │   ├── Annotation/
    │   │   └── ODIR-5K_Offsite(English).xlsx
    │   └── Images/
    └── On-site Test Set/
        ├── Annotation/
        │   └── ODIR-5K_Onsite(English).xlsx
        └── Images/
```

**Step 5.2 — Run the provided merge script** to consolidate all subsets into a single folder:

```bash
python merge_dataset.py
```

This script will:

1. Copy all 10,000 fundus images from the three subsets into one `OIA-ODIR-Merged/Images/` folder.
2. Load each subset's English annotation file (`.xlsx`) and merge them into one unified annotation file.
3. Add a `dataset` column to the merged CSV indicating the origin subset (Training / On-site / Off-site) so provenance is preserved.
4. Produce both `all_annotations.xlsx` and `all_annotations.csv` (CSV is used by the benchmark code).


After the merge, the layout will be:

```
Final_Project/OIA-ODIR-Merged/
├── all_annotations.csv        # Merged annotations used by the pipeline
├── all_annotations.xlsx       # Same data in Excel format
├── train_annotations.xlsx     # 80% split (optional use)
├── test_annotations.xlsx      # 20% split (optional use)
└── Images/                    # All 10,000 fundus images
    ├── 937_left.jpg
    ├── 937_right.jpg
    └── ...
```

The merged CSV contains the following columns:

- `ID` — Patient identifier
- `Patient Age` — Age in years
- `Patient Sex` — Male / Female
- `Left-Fundus` — Filename of the left fundus image
- `Right-Fundus` — Filename of the right fundus image
- `Left-Diagnostic Keywords` — Free-text clinical keywords for the left eye
- `Right-Diagnostic Keywords` — Free-text clinical keywords for the right eye
- `N`, `D`, `G`, `C`, `A`, `H`, `M`, `O` — Eight binary disease labels
- `dataset` — Origin subset (`Training`, `On-site`, or `Off-site`)

---

## Build Instructions

This is a pure Python project — no compilation or build step is required. After installation, the code runs directly.

To verify the installation and that the required LLM models are accessible on your OpenRouter account:
```bash
python test_models.py
```

---

## How to Run the Project

### Option A — Single-Patient Test (quick verification)
Run the full pipeline on a single patient from the dataset:
```bash
python run.py 0 gemini_25
```
Arguments: `<patient_index> <model_key>`

Available model keys (defined in `config.py`):

- `claude_sonnet` — Claude Sonnet 4.6
- `gemini_25` — Gemini 2.5 Flash
- `gpt4o` — GPT-4o
- `gemma_4` — Gemma 4 31B (open-weight)
- `qwen_vl` — Qwen 3 VL 8B (open-weight)

### Option B — Full Benchmark (all five models on 200 patients)
```bash
python run_benchmark.py 200
```
Approximate runtime: 5–6 hours. Approximate cost: USD 13 in OpenRouter credits. Progress is saved incrementally every 5 patients to the `results/` directory.

### Option C — Streamlit Web Application (interactive demonstration)
```bash
streamlit run app/app.py
```
The app opens at `http://localhost:8501` and provides four tabs:

1. **Demo Patient** — Select any patient from OIA-ODIR and run the full assessment pipeline.
2. **New Patient** — Upload custom fundus images and clinical data.
3. **Benchmark** — View the saved metrics and comparison charts.
4. **Analytics** — Inspect the session assessment history.

### Option D — Generate Visualizations
After running the benchmark, generate charts and confusion matrices:
```bash
python -m evaluation.visualize
```
The outputs are written to `results/visualizations/`.

---

## Project Structure

```
Final_Project/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── config.py                       # Global configuration (models, paths, thresholds)
├── .env.example                    # API key template
├── .env                            # Local API key (gitignored)
├── .gitignore                      # Git exclusions
│
├── merge_dataset.py                # Merge official OIA-ODIR subsets into unified dataset
├── run.py                          # Single-patient entry point
├── run_benchmark.py                # Multi-patient benchmark runner
├── test_models.py                  # Verify which models are accessible
│
├── agents/                         # Five-agent pipeline
│   ├── __init__.py
│   ├── pipeline.py                 # Agent 1–5 implementations
│   └── workflow.py                 # Sequential orchestration
│
├── app/                            # Streamlit web application
│   └── app.py                      # Four-tab interactive UI
│
├── evaluation/                     # Benchmark and analysis
│   ├── __init__.py
│   ├── benchmark.py                # (legacy wrapper; see run_benchmark.py)
│   └── visualize.py                # Chart generation
│
├── results/                        # Benchmark outputs
│   ├── metrics_*.json              # Per-model performance metrics
│   ├── predictions_*.csv           # Individual patient predictions
│   └── visualizations/             # Generated charts
│
└── OIA-ODIR-Merged/                # Dataset (not tracked in git, built via merge_dataset.py)
    ├── all_annotations.csv
    └── Images/
```

---

## Dataset Information

**Name:** OIA-ODIR (Ocular Disease Intelligent Recognition)

**Source (official repository):** https://github.com/nkicsl/OIA-ODIR

**Citation:** Li, T. et al. (2019). *Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening.* Information Sciences, 501, 511–522.

**Size:** 5,000 patients — 10,000 fundus images (bilateral).

**Per-patient data:**

- Patient ID, Age, Sex (Male / Female)
- Left eye fundus image (JPG)
- Right eye fundus image (JPG)
- Left eye diagnostic keywords (free text)
- Right eye diagnostic keywords (free text)
- Eight binary disease labels: N (Normal), **D (Diabetic Retinopathy)**, G (Glaucoma), C (Cataract), A (AMD), H (Hypertension), M (Myopia), O (Other)

**Class distribution:**

- DR-positive (D = 1): 1,618 patients (32.4%)
- DR-negative (D = 0): 3,382 patients (67.6%)

**Preprocessing steps:**

1. **Dataset consolidation** — The official OIA-ODIR distribution is split into three subsets (Training Set, On-site Test Set, Off-site Test Set). The provided `merge_dataset.py` script combines all images into a single `Images/` folder and all English annotation files into a single unified CSV (`all_annotations.csv`) and Excel file (`all_annotations.xlsx`). A `dataset` column records each patient's origin subset.
2. **CSV loading** — The unified annotation file is loaded with pandas and validated for the required columns.
3. **Ground-truth extraction** — The binary DR label is read directly from the `D` column (`1` = DR present, `0` = no DR).
4. **Image encoding** — Each fundus image is read from disk and encoded as a base64 data URL (MIME type `image/jpeg`) for transmission to vision-language models via the OpenAI-compatible multimodal message format. No resizing or normalization is applied — each LLM handles its own preprocessing internally.
5. **Text preparation** — The left and right clinical keyword fields are concatenated with a space separator; empty strings are replaced with a placeholder.
6. **Stratified sampling** — For evaluation, 100 DR-positive and 100 DR-negative patients are drawn using a fixed random seed (42) for reproducibility. This ensures non-degenerate precision and recall estimates despite the natural class imbalance in OIA-ODIR.

**Why OIA-ODIR:** Unlike DDR (images only) or Pima Indians (tabular only), OIA-ODIR provides all modalities for the same patient, which eliminates the unpaired-dataset limitation identified in our mid-project report.

---

## Sample Input and Expected Output

### Sample Input (from a CSV row):
```
Patient ID: 937
Patient Age: 60
Patient Sex: Female
Left-Fundus: 937_left.jpg
Right-Fundus: 937_right.jpg
Left-Diagnostic Keywords: hypertensive retinopathy
Right-Diagnostic Keywords: hypertensive retinopathy, suspected diabetic retinopathy
Ground Truth D (DR): 1
```

### Expected Output (JSON produced by the 5-agent pipeline):
```json
{
  "patient": {"age": 60, "sex": "Female"},
  "model_used": "gemini_25",
  "demographic": {
    "demographic_risk_score": 0.4,
    "age_risk_level": "moderate",
    "key_factors": ["age over 45", "female sex"]
  },
  "clinical": {
    "diabetic_findings": ["suspected diabetic retinopathy"],
    "text_derived_risk_score": 0.3,
    "clinical_summary": "Signs of suspected DR with hypertensive retinopathy"
  },
  "left_eye": {
    "dr": 0,
    "dr_severity": "no_dr",
    "dr_confidence": 0.95,
    "image_quality": "good"
  },
  "right_eye": {
    "dr": 1,
    "dr_severity": "mild",
    "dr_confidence": 0.80,
    "image_quality": "good"
  },
  "fusion": {
    "overall_diabetes_risk_level": "moderate",
    "diabetes_risk_score": 45.0,
    "confidence": 0.8,
    "risk_reasoning": "60-year-old female with suspected DR and hypertensive retinopathy..."
  },
  "prevention": {
    "tier_priority": 2,
    "tier_name": "MODERATE",
    "prevention_plan": {
      "lifestyle_modifications": ["150 minutes moderate aerobic activity weekly", "..."],
      "medical_screening": {
        "screening_frequency": "every 6 months",
        "recommended_tests": ["HbA1c", "fasting glucose", "lipid panel"]
      }
    }
  }
}
```

---

## Models Evaluated

| Model | Provider | License | OpenRouter ID |
|-------|----------|---------|---------------|
| Claude Sonnet 4.6 | Anthropic | Closed-source | `anthropic/claude-sonnet-4.6` |
| Gemini 2.5 Flash | Google DeepMind | Closed-source | `google/gemini-2.5-flash` |
| GPT-4o | OpenAI | Closed-source | `openai/gpt-4o` |
| Gemma 4 31B | Google DeepMind | Open-weight | `google/gemma-4-31b-it` |
| Qwen 3 VL 8B | Alibaba | Open-weight | `qwen/qwen3-vl-8b-instruct` |

---

## Evaluation Metrics

- **Accuracy** — (TP + TN) / (TP + TN + FP + FN)
- **Precision** — TP / (TP + FP)
- **Recall (Sensitivity)** — TP / (TP + FN)
- **F1-Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Area under the Receiver Operating Characteristic curve
- **Success Rate** — Fraction of patients with a valid JSON prediction
- **Average Inference Time** — Seconds per patient per model

---

## Team

| Student | UAEU ID | Email |
|---------|---------|-------|
| Saeed Mohammed Alshehhi | 201604533 | 201604533@uaeu.ac.ae |
| Maitha Abdulla Al Hayyas | 201403194 | 201403194@uaeu.ac.ae |
| Fatima Mohammed Alnuaimi | 201302209 | 201302209@uaeu.ac.ae |

**Instructor:** Dr. Leila Ismail

---

## GitHub Repository

**Repository Link:** https://github.com/SaeedAlshehhi-UAEU/itpg708-diabetes-screening

The GitHub repository:

- Contains the same version as the submitted `.zip` file.
- Includes a complete commit history demonstrating development progress.
- Is publicly accessible.

---

## Citation

```
Alshehhi, S. M., Al Hayyas, M. A., Alnuaimi, F. M. (2026).
Multimodal Agentic AI for Diabetic Retinopathy Screening: Benchmarking
Closed-Source and Open-Weight Vision-Language Models.
ITPG 708 Final Project, United Arab Emirates University, Spring 2026.
```

---

## Disclaimer

This system is for **educational and research purposes only**. It is not a certified medical device, and its outputs must not be used for actual clinical decision-making. Always consult qualified ophthalmologists and endocrinologists for medical diagnosis and treatment. The authors and UAEU assume no liability for any use of this system beyond academic demonstration.
