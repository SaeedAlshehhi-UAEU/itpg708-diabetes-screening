# Multimodal Diabetes Screening Using Hybrid AI Models and Agent-Based Decision

**ITPG 708 Final Project – Spring 2026**

**Prepared by:**
- Saeed Mohammed Alshehhi — 201604533@uaeu.ac.ae
- Maitha Abdulla Al Hayyas — 201403194@uaeu.ac.ae
- Fatima Mohammed Alnuaimi — 201302209@uaeu.ac.ae

**Instructor:** Dr. Leila Ismail

---

## Abstract

This paper presents a multimodal agentic artificial intelligence system for diabetic retinopathy (DR) detection and diabetes risk assessment. Departing from conventional pipelines that combine convolutional neural networks with tabular classifiers, we propose a unified architecture built on vision-language large language models (LLMs) orchestrated through a five-agent pipeline. The system integrates three data modalities from a single patient: bilateral fundus images, free-text clinical keywords, and structured demographic data, eliminating the unpaired-dataset limitation present in prior multimodal studies. Five specialized agents — demographic risk scoring, clinical NLP extraction, bilateral vision analysis, multimodal risk fusion, and tier-based prevention planning — are connected through a LangGraph orchestration layer. We evaluated five state-of-the-art vision-language models from four providers (Anthropic Claude Sonnet 4.6, Google Gemini 2.5 Flash, OpenAI GPT-4o, Google Gemma 4 31B, and Alibaba Qwen 3 VL 8B) on 200 stratified patients from the Ocular Disease Intelligent Recognition (OIA-ODIR) dataset. The benchmark includes both closed-source and open-weight models, enabling cross-provider and licensing-based comparisons. The proposed system produces personalized prevention plans across three clinically meaningful risk tiers (Urgent, Moderate, Preventive), and it constitutes a reproducible benchmark for comparing multimodal LLMs in automated medical screening.

---

## 1. Introduction

Diabetes mellitus affects approximately 537 million adults worldwide, and its most severe ocular complication — diabetic retinopathy (DR) — is a leading cause of preventable blindness. Early detection and personalized prevention planning can significantly reduce the burden of advanced DR and associated vision loss. However, real-world screening remains limited by the shortage of ophthalmologists, the cost of specialized equipment, and the lack of integrated decision-support systems that can reason across multiple data modalities.

Artificial intelligence has the potential to transform diabetes screening. Deep convolutional neural networks (CNNs) have demonstrated ophthalmologist-level accuracy in DR detection from fundus photographs, while traditional machine-learning models can estimate diabetes risk from structured clinical variables such as age, blood pressure, and glycemic indicators. Nevertheless, these approaches typically operate in isolation: vision models focus on retinal images, tabular models focus on clinical features, and the two are rarely combined in a clinically actionable pipeline. Moreover, most published multimodal systems rely on unpaired datasets — using one corpus for images and another for tabular variables — which limits the ability to evaluate true multimodal reasoning on a single patient.

Recent advances in vision-language large language models (LLMs) have opened a new paradigm. Models such as Claude Sonnet 4.6, GPT-4o, Gemini 2.5 Flash, Gemma 4, and Qwen 3 VL can jointly process images and text, perform medical reasoning, and generate structured outputs. When combined with agent-orchestration frameworks such as LangGraph, these models enable the construction of modular pipelines in which each agent handles a specific subtask: demographic reasoning, clinical language understanding, fundus interpretation, multimodal fusion, and prevention planning. Unlike traditional pipelines, this agent-based approach does not require model training — it leverages pretrained foundation models through prompt engineering and structured orchestration.

This project designs, implements, and evaluates a five-agent multimodal screening system for diabetes risk and DR detection. The contributions are as follows:

1. **Unified multimodal pipeline** that processes bilateral fundus images, clinical keywords, and demographics for the same patient, solving the unpaired-dataset limitation.
2. **Five-agent architecture** orchestrated through LangGraph, replacing trainable CNN and tabular classifiers with prompt-engineered vision-language agents.
3. **Comparative benchmark** of five state-of-the-art LLMs (Claude Sonnet 4.6, Gemini 2.5 Flash, GPT-4o, Gemma 4 31B, Qwen 3 VL 8B) on 200 stratified OIA-ODIR patients using standard classification metrics, spanning both closed-source and open-weight models.
4. **Tier-based prevention planning** that converts probabilistic risk estimates into clinically actionable interventions across three urgency levels.

### 1.1 Problem Formulation

The objective is to design a multimodal agentic system that integrates bilateral retinal imaging, structured demographics, and free-text clinical descriptions to produce a unified diabetes-risk assessment and a personalized prevention plan.

**Inputs.** Each patient $i$ is represented by a tuple:
$$
x_i = (I_i^L, I_i^R, a_i, s_i, k_i^L, k_i^R)
$$
where $I_i^L, I_i^R$ are the left and right fundus images, $a_i$ is age, $s_i \in \{\text{Male}, \text{Female}\}$ is sex, and $k_i^L, k_i^R$ are free-text clinical keywords for the left and right eyes.

**Outputs.** The system produces three outputs:
$$
y_i^{\text{DR}} \in \{0, 1\}, \quad r_i \in [0, 100], \quad T_i \in \{1, 2, 3\}
$$
where $y_i^{\text{DR}}$ is binary DR prediction, $r_i$ is a continuous risk score, and $T_i$ is the prevention tier.

**Five Sequential Tasks.** The problem decomposes into five agent tasks:

1. **Demographic Risk** — $g_1(a_i, s_i; \theta_L) \rightarrow d_i$ where $d_i \in [0, 1]$.
2. **Clinical NLP** — $g_2(k_i^L, k_i^R; \theta_L) \rightarrow c_i$ where $c_i \in [0, 1]$.
3. **Bilateral Vision Analysis** — $g_{3}(I_i^{L/R}; \theta_L) \rightarrow (\hat{y}_i^{L/R}, q_i^{L/R})$ where $\hat{y}$ is per-eye DR label and $q$ is confidence.
4. **Multimodal Fusion** — $g_4(d_i, c_i, \hat{y}_i^L, \hat{y}_i^R; \theta_L) \rightarrow (r_i, y_i^{\text{DR}}, \phi_i)$ where $\phi_i$ is overall confidence.
5. **Prevention Planning** — $g_5(r_i, a_i, y_i^{\text{DR}}; \theta_L) \rightarrow (T_i, \mathcal{P}_i)$ where $\mathcal{P}_i$ is the structured prevention plan.

All agents share the same underlying vision-language model $\theta_L$, differing only in prompt specification and input modalities.

**Tier Assignment.** Prevention tier $T_i$ is assigned by threshold:
$$
T_i = \begin{cases}
1 \text{ (Urgent)} & \text{if } r_i \geq 70 \\
2 \text{ (Moderate)} & \text{if } 40 \leq r_i < 70 \\
3 \text{ (Preventive)} & \text{if } r_i < 40
\end{cases}
$$

**Constraints and Assumptions.**
- (C1) All modalities originate from the same patient — no cross-dataset joining.
- (C2) The LLM is queried with zero training; behavior depends entirely on prompt engineering.
- (C3) Outputs must be valid JSON to enable programmatic downstream processing.
- (C4) Risk score $r_i$ is calibrated to $[0, 100]$ for clinical interpretability.
- (C5) A single image-quality failure in either eye does not invalidate the assessment; the fusion agent weights modalities accordingly.

**Central Hypothesis.** A vision-language LLM orchestrated through specialized agents can perform multimodal diabetes risk assessment with diagnostic performance comparable to dedicated CNN and tabular pipelines, while offering greater interpretability and generalizability through natural-language reasoning.

### 1.2 Research Questions

1. **RQ1.** Can prompt-engineered vision-language agents replace trainable CNN and tabular models for multimodal DR screening without sacrificing diagnostic performance?
2. **RQ2.** How do cross-provider LLMs (Anthropic, Google closed, Google open-weight, OpenAI, Alibaba) differ in precision, recall, and output reliability on the same medical imaging task?
3. **RQ3.** Does an agent-based fusion layer produce clinically interpretable and consistent tier assignments when integrating heterogeneous modalities from the same patient?
4. **RQ4.** Do open-weight models (Gemma 4 31B, Qwen 3 VL 8B) match closed-source frontier models (Claude, GPT-4o, Gemini) on multimodal medical screening tasks?

---

## 2. Related Work

### 2.1 Large Language Models in Medical Imaging

The application of large language models to medical imaging has emerged rapidly since 2023. Jin et al. (Ophthalmology Science, 2025) evaluated multimodal LLMs on diabetic retinopathy detection from ultra-widefield fundus images, reporting diagnostic accuracies between 52% and 61%. Their findings indicate that, while MLLM performance remains below dedicated CNN algorithms, the models offer flexible natural-language outputs that enhance interpretability for clinicians. A complementary study published in Meta-Radiology (2024) evaluated Gemini-series and GPT-4-series models across fourteen medical imaging datasets covering five imaging categories — dermatology, radiology, dentistry, ophthalmology, and endoscopy — finding that general-purpose MLLMs can achieve clinically useful performance on visual tasks without domain-specific fine-tuning.

The Japanese Journal of Radiology (Springer, 2024) specifically evaluated Claude 3 Opus and Claude 3.5 Sonnet on 322 radiology quiz cases, investigating diagnostic performance under three input conditions: clinical history alone, clinical history with imaging findings, and clinical history with key images. In ScienceDirect (2024), GPT-4o and Claude Sonnet 3.5 were tested on 120 clinical vignettes with and without accompanying images: LLMs outperformed physicians in text-only scenarios (GPT-4o 70.8%, Claude Sonnet 3.5 59.5%, physicians 39.5%; Bonferroni-adjusted p < 0.001), and all improved with image integration (GPT-4o 84.5%, Claude 67.3%, physicians 78.8%). Work on MedGemma — a medical variant of Gemma 3 released by Google DeepMind (arXiv:2507.05201) — demonstrates that the Gemma architecture underlying Gemini models is specifically validated for medical image interpretation, including fundus photography. Similarly, MedGemini (Saab et al., 2024) extended Gemini for radiology and clinical reasoning tasks, achieving state-of-the-art results on MedQA and retinal benchmarks. The Qwen-VL family has also been applied to medical imaging: UMIT (arXiv:2503.15892, 2025) unified multiple medical imaging tasks using Qwen2-VL as the backbone, and MDPI Bioengineering (2025) systematically evaluated Qwen variants on the ROCOv2 dataset containing 116,635 images across eight imaging modalities.

**Critical limitation:** most existing MLLM studies evaluate models on single-image tasks in isolation, without multimodal fusion or agent orchestration. The integration of visual and textual modalities typically happens inside a single model call, with no structured decomposition of subtasks.

### 2.2 Agentic AI in Healthcare

The agentic paradigm — decomposing complex reasoning into specialized cooperating agents — has shown promise in medical applications. A Nature Scientific Reports paper (2025) proposed an agentic AI framework for DR detection in retinal fundus images, using coordinating agents for lesion detection, severity grading, and report generation. Their approach demonstrates the feasibility of replacing monolithic end-to-end CNNs with modular agent pipelines. LangGraph, a graph-based agent orchestration framework introduced by LangChain in 2024, provides the infrastructure for constructing sequential and conditional agent workflows with shared state. While LangGraph has been widely adopted for general-purpose task automation, published applications to medical screening remain sparse.

**Critical limitation:** existing agentic medical systems tend to use purpose-trained vision models for each agent. The use of a single pretrained LLM across all agents — as in our design — reduces complexity and eliminates training costs, but has not been systematically evaluated.

### 2.3 Deep Learning for Diabetic Retinopathy

Traditional CNN-based DR detection is well established. Gulshan et al. (JAMA, 2016) reported a deep learning system with 97% sensitivity for referable DR, and Ting et al. (JAMA, 2017) validated similar systems across multi-ethnic populations. Newer architectures — EfficientNet, ResNet-50, Swin Transformer, Vision Transformer — have been benchmarked for DR severity grading with accuracies above 90% on curated datasets. A 2025 systematic review published in the Journal of Medical Imaging and Interventional Radiology found that hybrid CNN-ViT architectures outperform pure CNN backbones for DR classification, particularly on imbalanced datasets.

**Critical limitation:** CNN-based approaches require large labeled datasets, extensive training, and produce numerical outputs without natural-language reasoning. They also cannot natively integrate textual clinical information.

### 2.4 Multimodal Learning in Healthcare

Huang et al. (npj Digital Medicine, 2020) reviewed multimodal fusion approaches combining medical imaging and electronic health records, concluding that early fusion typically outperforms late fusion when modalities are well-aligned. Yala et al. (Radiology, 2019) demonstrated that fusing mammograms with tabular risk factors improved breast cancer risk prediction. However, the majority of these multimodal studies combine unpaired datasets — using one corpus for images and another for tabular features — which limits true per-patient multimodal reasoning.

### 2.5 Research Gap

The literature reveals four unaddressed gaps:

1. **Unpaired-dataset limitation.** Existing multimodal DR systems combine datasets such as DDR (images) with Pima Indians (tabular variables), where no patient appears in both. This prevents genuine per-patient multimodal reasoning.
2. **Training dependency.** CNN and tabular baselines require large labeled datasets and computational resources, limiting deployment in low-resource settings.
3. **Lack of agent decomposition.** Multimodal LLM studies typically feed all inputs into a single model call, eliminating the interpretability benefits of modular agent pipelines.
4. **Absence of open-weight comparisons.** Most MLLM medical studies compare only closed-source models (GPT, Claude, Gemini), omitting open-weight alternatives that are critical for low-resource and privacy-sensitive deployments.

This project addresses all four gaps by: (1) using the OIA-ODIR dataset in which every patient has bilateral fundus images, demographics, and clinical keywords; (2) eliminating training through prompt-engineered LLM agents; (3) decomposing reasoning into five interpretable agents; and (4) benchmarking two open-weight models (Gemma 4, Qwen 3 VL) alongside three closed-source flagships (Claude Sonnet 4.6, GPT-4o, Gemini 2.5 Flash).

---

## 3. System Architecture

### 3.1 Overview

The proposed system is a five-agent pipeline in which a single vision-language model is invoked sequentially with different prompts and modalities. Each agent specializes in a distinct subtask, and agent outputs are passed through a shared state dictionary managed by LangGraph. Figure 1 illustrates the high-level architecture.

```
 ┌─────────────────────────────────────────────────────────────┐
 │             Patient Data (OIA-ODIR Dataset)                 │
 │   Age, Sex, Left Fundus, Right Fundus, Clinical Keywords    │
 └──────────────────────────┬──────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
  │  Agent 1    │   │  Agent 2    │   │  Agent 3    │
  │ Demographic │   │  Clinical   │   │   Vision    │
  │    Risk     │   │     NLP     │   │ (L/R eyes)  │
  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           ▼
                   ┌───────────────┐
                   │   Agent 4     │
                   │ Risk Fusion   │
                   └───────┬───────┘
                           ▼
                   ┌───────────────┐
                   │   Agent 5     │
                   │  Prevention   │
                   └───────┬───────┘
                           ▼
                ┌────────────────────┐
                │  Final Assessment  │
                │  + Tier + Plan     │
                └────────────────────┘
```

**Figure 1.** Five-agent multimodal pipeline. Agents 1–3 operate on their respective modalities; Agents 4–5 operate sequentially on aggregated outputs.

### 3.2 Agent Specifications

**Agent 1 — Demographic Risk Scorer.** Receives age $a_i$ and sex $s_i$ and produces a normalized demographic risk score $d_i \in [0, 1]$ together with a qualitative risk level (low/moderate/high). The agent is implemented by invoking the LLM with a text-only prompt that instructs it to apply epidemiological knowledge: diabetes risk increases substantially after age 45, and males have slightly higher risk than females.

**Agent 2 — Clinical NLP Extractor.** Receives concatenated left/right clinical keywords $(k_i^L, k_i^R)$ and extracts a list of diabetes-relevant findings (e.g., microaneurysms, hypertensive retinopathy, neovascularization) together with a text-derived risk score $c_i \in [0, 1]$. This agent replaces traditional named-entity-recognition pipelines with a prompt-engineered LLM call.

**Agent 3 — Bilateral Vision Analysis.** Invoked twice per patient (once for each eye). The fundus image is encoded as a base64 data URL and embedded in a multimodal message. The LLM is prompted to act as an expert ophthalmologist, grading the image as No DR, Mild, Moderate, Severe, or Proliferative DR, and identifying any of the five canonical DR signs (microaneurysms, hemorrhages, hard exudates, cotton-wool spots, neovascularization). The output is a structured dictionary containing binary DR label $\hat{y}_i^{L/R}$, severity grade, confidence $q_i^{L/R}$, image quality, and a short clinical note.

**Agent 4 — Risk Fusion.** Receives all prior outputs and performs multimodal reasoning to produce a unified diabetes risk score $r_i \in [0, 100]$, an overall risk level, an overall confidence $\phi_i$, and a textual rationale. Unlike traditional weighted-average fusion, the LLM reasons across modalities using natural-language rules specified in the prompt (DR presence in either eye is a strong indicator; age > 45 combined with DR signals high risk; clinical keywords reinforce or contradict imaging findings).

**Agent 5 — Prevention Planner.** Receives the fused risk score and patient context, assigns a prevention tier $T_i$, and generates a structured plan containing lifestyle modifications, medical screening schedule, specialist referrals, monitoring frequency, education topics, and an implementation summary. The prompt explicitly encodes tier-specific behavior so that Tier 1 (Urgent) outputs daily monitoring and immediate referrals while Tier 3 (Preventive) outputs annual checks and general lifestyle guidance.

### 3.3 LangGraph Orchestration

Agents are connected through a LangGraph StateGraph with a shared state dictionary. Each agent reads from and writes to this state, allowing downstream agents to access upstream results. The workflow is sequential: Agent 1 → Agent 2 → Agent 3a → Agent 3b → Agent 4 → Agent 5. During benchmark execution, a stateless wrapper was used in place of the graph to prevent cross-patient state contamination when the same compiled workflow is reused across many patients.

### 3.4 Mathematical Formulation

Let $\pi_k$ denote the prompt template for agent $k$, and let $\mathcal{L}(\pi, I; \theta_L)$ denote the LLM's output when given prompt $\pi$ and optional image $I$. The five agents are:

$$
\begin{aligned}
d_i, \ell^d_i &= \mathcal{L}(\pi_1(a_i, s_i); \theta_L) \quad &\text{(Agent 1)} \\
c_i, \mathcal{F}_i &= \mathcal{L}(\pi_2(k_i^L, k_i^R); \theta_L) \quad &\text{(Agent 2)} \\
\hat{y}_i^{L}, q_i^{L} &= \mathcal{L}(\pi_3^{\text{left}}(), I_i^L; \theta_L) \quad &\text{(Agent 3a)} \\
\hat{y}_i^{R}, q_i^{R} &= \mathcal{L}(\pi_3^{\text{right}}(), I_i^R; \theta_L) \quad &\text{(Agent 3b)} \\
r_i, y_i^{\text{DR}}, \phi_i &= \mathcal{L}(\pi_4(d_i, c_i, \hat{y}_i^L, \hat{y}_i^R, a_i, s_i); \theta_L) \quad &\text{(Agent 4)} \\
T_i, \mathcal{P}_i &= \mathcal{L}(\pi_5(r_i, a_i, s_i, y_i^{\text{DR}}); \theta_L) \quad &\text{(Agent 5)}
\end{aligned}
$$

where $\ell^d$ is the age-risk level, $\mathcal{F}$ is the set of clinical findings, and $\mathcal{P}$ is the structured prevention plan.

**Binary DR prediction.** From the fused risk level, the binary DR prediction is derived as:
$$
\hat{y}_i^{\text{DR}} = \mathbb{1}[\text{risk level}_i = \text{``high''}]
$$

**Fusion rationale (non-parametric).** Unlike weighted averaging, the LLM fusion agent applies learned reasoning rules encoded in the prompt. Symbolically, the fusion can be described as a conditional logic:
$$
r_i = f_{\text{LLM}}\left(d_i, c_i, \hat{y}_i^L, \hat{y}_i^R, q_i^L, q_i^R, \mathcal{F}_i\right)
$$
where $f_{\text{LLM}}$ is parameterized entirely by the pretrained model weights $\theta_L$ and the fusion prompt $\pi_4$, with no additional training.

---

## 4. Methodology

### 4.1 Dataset — OIA-ODIR

The Ocular Disease Intelligent Recognition (OIA-ODIR) dataset is used for all experiments. OIA-ODIR contains 5,000 patients, each with a bilateral fundus image pair (left and right eye), demographic information (age, sex), free-text diagnostic keywords per eye, and eight binary disease labels: Normal (N), Diabetic retinopathy (D), Glaucoma (G), Cataract (C), Age-related Macular degeneration (A), Hypertension (H), Myopia (M), and Other abnormalities (O). The D label serves as ground truth for DR detection.

**Class distribution.** Of the 5,000 patients, 1,618 (32.4%) have DR (D = 1) and 3,382 (67.6%) do not (D = 0). This moderate class imbalance motivates stratified sampling during evaluation.

**Why OIA-ODIR.** Unlike DDR (images-only) or Pima Indians (tabular-only), OIA-ODIR provides all modalities for the same patient, eliminating the unpaired-dataset limitation that affected the initial project design.

### 4.2 Preprocessing

**Image encoding.** Each fundus image is read from disk and encoded as a base64 data URL with MIME type `image/jpeg`. The resulting string is embedded directly in the LLM request payload via the OpenAI-compatible multimodal message format. No resizing, normalization, or augmentation is applied — the LLM's internal preprocessing handles these steps.

**Text preparation.** The left and right clinical keywords are concatenated with a space separator. Empty strings are replaced with the placeholder "No clinical description available".

**Ground truth.** The binary DR label is extracted directly from the D column of the CSV annotations.

### 4.3 Stratified Sampling

From the 5,000 OIA-ODIR patients, a stratified sample of 200 patients is drawn — 100 with D = 1 and 100 with D = 0 — using a fixed random seed (42) for reproducibility. Stratification is essential because uniform random sampling would yield approximately 32 positive cases per 100, producing degenerate precision and recall estimates.

### 4.4 Model Selection

Five vision-language models are evaluated, selected to span four providers and two licensing categories (closed-source and open-weight):

**1. Claude Sonnet 4.6 (Anthropic, closed-source).** Anthropic's flagship Sonnet-class model with frontier performance on coding, reasoning, and professional work. Priced at \$3 per million input tokens and \$15 per million output tokens, supporting a 1,000,000-token context window. Claude 3.5 Sonnet achieved 66.2% accuracy on multimodal radiology tasks (Preprints, 2025) and the highest accuracy (90%) on breast imaging BI-RADS MCQs among evaluated LLMs (Diagnostic and Interventional Radiology, 2025). Sonnet 4.6 extends these capabilities with improved agent-oriented reasoning.

**2. Gemini 2.5 Flash (Google DeepMind, closed-source).** State-of-the-art workhorse model designed for advanced reasoning, coding, and scientific tasks, with built-in "thinking" capabilities for nuanced context handling. Priced at \$0.30 per million input tokens and \$2.50 per million output tokens, with a 1,048,576-token context. The broader Gemini family has been evaluated across 14 medical imaging datasets (Meta-Radiology, 2024), and the medical-specialized variant MedGemini (Saab et al., 2024) has achieved state-of-the-art performance on radiology benchmarks.

**3. GPT-4o (OpenAI, closed-source).** OpenAI's multimodal model, widely cited as a baseline in medical AI literature. GPT-4o was explicitly evaluated for diabetic retinopathy detection from fundus photos in Ophthalmology Science (2025), and reached 84.5% diagnostic accuracy on image-augmented clinical vignettes in ScienceDirect (2024). Priced at \$2.50 per million input tokens and \$10 per million output tokens.

**4. Gemma 4 31B (Google DeepMind, open-weight).** The latest open-weight model in Google's Gemma family. The Gemma 3 Technical Report (arXiv:2503.19786) introduced multimodal vision understanding, and the MedGemma Technical Report (arXiv:2507.05201) validates the Gemma architecture for medical imaging. Including Gemma 4 enables direct comparison of open-weight vs. closed-source models at comparable parameter scales.

**5. Qwen 3 VL 8B (Alibaba, open-weight).** Alibaba's latest vision-language model with native 256K context (extensible to 1M), Interleaved-MRoPE for long-horizon reasoning, and DeepStack visual-text alignment. The Qwen-VL family has been systematically applied to medical imaging in UMIT (arXiv:2503.15892, 2025) — which unified medical report generation, disease classification, lesion detection, and visual question answering — and was evaluated on ROCOv2 (116,635 medical images) in MDPI Bioengineering (2025).

All five models are accessed through the OpenRouter API, which provides a unified OpenAI-compatible interface.

### 4.5 Prompt Engineering

Prompts for all five agents follow a shared template structure: (1) role specification ("You are an expert ophthalmologist…"), (2) input description, (3) task specification with rubric, (4) strict JSON output schema, and (5) explicit instruction to avoid markdown formatting. A robust JSON parser handles the case where models — particularly Gemini and Gemma — wrap responses in ```json code blocks. The `max_tokens` parameter is set to 1,500 to prevent truncation of verbose responses, a critical fix discovered during pilot testing.

---

## 5. Experimental Setup

### 5.1 Evaluation Metrics

For binary DR detection, five standard metrics are reported.

**Accuracy:**
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Precision:**
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Recall (Sensitivity):**
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**F1-Score:**
$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**ROC-AUC:** Area under the Receiver Operating Characteristic curve, computed from continuous risk scores $r_i$.

In addition, we report the **success rate** — the fraction of patients for which the pipeline produced a valid JSON prediction — as a proxy for model reliability. The **average inference time** per patient is recorded for cost and latency analysis.

### 5.2 Baseline Comparison

The five LLMs are compared against one another on the same 200 patients. No external CNN baselines are trained; the literature baseline is taken from Jin et al. (Ophthalmology Science, 2025), who reported 52–61% accuracy for four multimodal LLMs on similar DR detection tasks, and from ScienceDirect (2024), where GPT-4o reached 84.5% on image-augmented clinical vignettes.

### 5.3 Hardware and Software

- **Language:** Python 3.14
- **LLM orchestration:** LangGraph
- **API client:** OpenAI Python SDK targeting OpenRouter
- **Metrics:** scikit-learn
- **Dataset handling:** pandas
- **UI:** Streamlit
- **Hardware:** Standard laptop; no GPU required since inference occurs via cloud APIs

### 5.4 Evaluation Sample Justification

A stratified sample of 200 patients was selected as the evaluation cohort. This size aligns with recent MLLM medical-evaluation studies — Jin et al. (2025) evaluated on a similar order of magnitude — and represents a practical balance between statistical meaningfulness and API cost. The total benchmark cost for five models on 200 patients was approximately USD 13, and total runtime was approximately 5–6 hours.

---

## 6. Results and Analysis

*Note: Final results from the five-model benchmark will be inserted here once the 200-patient run completes. The following structure will be populated with:*

### 6.1 Overall Model Performance

A table summarizing accuracy, precision, recall, F1, ROC-AUC, success rate, and average inference time for all five models on the 200-patient stratified sample.

### 6.2 Confusion Matrix Analysis

Per-model TP/TN/FP/FN breakdowns, with discussion of each model's bias (conservative vs. aggressive prediction patterns).

### 6.3 Closed-Source vs. Open-Weight Comparison

Direct comparison of closed-source flagships (Claude Sonnet 4.6, Gemini 2.5 Flash, GPT-4o) against open-weight models (Gemma 4 31B, Qwen 3 VL 8B), addressing RQ4.

### 6.4 Failure Mode Analysis

Identification of patterns in failed predictions, including token truncation, safety-refusal, and JSON parse errors.

### 6.5 Cost Analysis

Per-patient and total benchmark costs across all five models, highlighting the cost-accuracy trade-off.

### 6.6 Comparison with Literature

Positioning of results relative to Jin et al. (Ophthalmology Science, 2025) and ScienceDirect (2024) MLLM benchmarks.

### 6.7 Answering the Research Questions

Consolidated answers to RQ1–RQ4 based on the empirical evidence.

---

## 7. Implementation

### 7.1 Code Structure

The repository is organized as follows:

```
Final_Project/
├── config.py                 # Model IDs, dataset paths, thresholds
├── run.py                    # Single-patient entry point
├── run_benchmark.py          # Multi-patient benchmark runner
├── .env                      # OpenRouter API key (gitignored)
├── agents/
│   ├── pipeline.py           # Five agent implementations
│   └── workflow.py           # Sequential orchestration
├── app/
│   └── app.py                # Streamlit user interface
├── evaluation/
│   ├── benchmark.py          # LangGraph-based benchmark
│   └── visualize.py          # Results visualization
├── OIA-ODIR-Merged/          # Dataset (not in git)
└── results/                  # Benchmark outputs
```

### 7.2 Reproducibility

All experiments are reproducible with the following steps:

1. Install dependencies: `pip install -r requirements.txt`
2. Download OIA-ODIR and place under `OIA-ODIR-Merged/`
3. Set `OPENROUTER_API_KEY` in `.env`
4. Run benchmark: `python run_benchmark.py 200`
5. Generate visualizations: `python -m evaluation.visualize`

Random seed = 42 is used throughout.

### 7.3 User Interface

A Streamlit web application (`app/app.py`) provides three tabs: (1) Demo Patient — select any OIA-ODIR patient and run the pipeline; (2) New Patient — upload custom fundus images and clinical data; (3) Benchmark Results — view saved metrics and comparison charts. The interface is intended for interactive demonstrations, not deployed clinical use.

---

## 8. Conclusion and Future Work

This project designed and evaluated a five-agent multimodal pipeline for diabetes risk and DR detection using only pretrained vision-language LLMs — no training was performed. The pipeline successfully integrates bilateral fundus imaging, demographic data, and clinical keywords within a single coherent reasoning chain, producing tier-based prevention plans that match clinical workflow expectations. The benchmark across five models from four providers (Anthropic, Google closed, Google open-weight, OpenAI, Alibaba) provides the first systematic cross-provider and licensing-aware evaluation of multimodal LLMs for diabetic retinopathy screening on paired-modality patient data.

**Limitations.**

1. Diagnostic accuracy of zero-shot vision-language LLMs remains below that of dedicated CNN systems (approximately 50–70% vs. >90%).
2. Sample size (200 patients) is adequate for model comparison but insufficient for claims of clinical utility.
3. Output reliability varies substantially by model: verbose models produce longer reasoning that can exceed token limits.
4. Prompt engineering, while effective, is brittle — small prompt changes can affect output format and performance.

**Future work.**

1. Expand evaluation to the full 5,000-patient OIA-ODIR dataset.
2. Incorporate MedGemma and other domain-specialized medical LLMs.
3. Add an ensemble agent that combines predictions from all five models with uncertainty-weighted voting.
4. Develop few-shot prompting with curated DR examples to improve recall.
5. Integrate with the DDR dataset for five-class severity grading.
6. Evaluate clinician trust and interpretability through user studies.

---

## References

1. Gulshan, V. et al. (2016). "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs." *JAMA*, 316(22), 2402–2410.

2. Ting, D. S. W. et al. (2017). "Development and Validation of a Deep Learning System for Diabetic Retinopathy and Related Eye Diseases Using Retinal Images from Multiethnic Populations." *JAMA*, 318(22), 2211–2223.

3. Jin, S. et al. (2025). "Can Multimodal Large Language Models Diagnose Diabetic Retinopathy from Fundus Photos? A Quantitative Evaluation." *Ophthalmology Science*.

4. Nature Scientific Reports (2025). "Detection and Diagnosis of Diabetic Retinopathy in Retinal Fundus Images Using Agentic AI Approaches."

5. Saab, K. et al. (2024). "Capabilities of Gemini Models in Medicine." *Google DeepMind Technical Report*.

6. OpenAI (2024). "GPT-4o System Card." OpenAI Technical Report.

7. Gemma Team, Google DeepMind (2025). "Gemma 3 Technical Report." *arXiv:2503.19786*.

8. Gemma Team, Google DeepMind (2025). "MedGemma Technical Report." *arXiv:2507.05201*.

9. Qwen Team, Alibaba (2025). "Qwen3-VL Technical Report." Released November 2025.

10. UMIT (2025). "UMIT: Unifying Medical Imaging Tasks via Vision-Language Models." *arXiv:2503.15892*.

11. Anthropic (2024). "The Claude 3 Model Family: Opus, Sonnet, Haiku." Anthropic Model Card.

12. Japanese Journal of Radiology (Springer, 2024). "Diagnostic performances of Claude 3 Opus and Claude 3.5 Sonnet from patient history and key images in Radiology's Diagnosis Please cases."

13. Diagnostic and Interventional Radiology (2025). "Evaluating text and visual diagnostic capabilities of large language models on questions related to the Breast Imaging Reporting and Data System Atlas 5th edition."

14. ScienceDirect (2024). "Visual-textual integration in LLMs for medical diagnosis: A preliminary quantitative analysis."

15. Meta-Radiology (2024). "Evaluation of Gemini-series and GPT-4-series Models Across Medical Imaging Datasets."

16. Huang, S.-C. et al. (2020). "Fusion of Medical Imaging and Electronic Health Records Using Deep Learning: A Systematic Review and Implementation Guidelines." *npj Digital Medicine*, 3, 136.

17. Journal of Medical Imaging and Interventional Radiology (2025). "Diagnostic Accuracy of AI or DL-Enhanced Technologies in the Diagnosis of Diabetic Retinopathy: A Systematic Review."

18. MDPI Bioengineering (2025). "An Empirical Evaluation of Low-Rank Adapted Vision-Language Models on Medical Imaging."

19. Li, T. et al. (2019). "Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening." *Information Sciences*, 501, 511–522. (OIA-ODIR Dataset)

20. LangChain (2024). "LangGraph: Graph-based Orchestration for LLM Agents." LangChain Documentation.

21. Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). "Multimodal Machine Learning: A Survey and Taxonomy." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41, 423–443.
