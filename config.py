# =============================================================================
# config.py
# Global configuration for the Multimodal Agentic Diabetes Screening System
# =============================================================================

# -----------------------------------------------------------------------------
# LLM Models (via OpenRouter) - 5 models from 4 different providers
#
# Model selection justification:
#   - Claude Sonnet 4.6 : Anthropic's flagship reasoning model, frontier-class
#                         performance, $3/$15 per M tokens, validated for
#                         complex medical reasoning tasks
#   - Gemini 2.5 Flash  : Google DeepMind's state-of-the-art multimodal model
#                         with built-in "thinking" capabilities, upgraded from
#                         Gemini 2.0, validated via MedGemini for medical tasks
#   - GPT-4o            : OpenAI's flagship multimodal model, industry-standard
#                         vision-language baseline in medical AI literature
#   - Gemma 4 31B       : Google open-weight model, architectural diversity,
#                         MedGemma variant validates Gemma for medical imaging
#   - Qwen 3 VL 8B      : Alibaba's latest vision-language model, different
#                         provider/architecture (non-Western), agentic features
#
# Comparing these 5 enables analysis of:
#   - Cross-provider performance (Anthropic, Google, OpenAI, Alibaba)
#   - Closed vs open-weight models (3 closed, 2 open)
#   - Cost-accuracy tradeoffs across 60x price range
# -----------------------------------------------------------------------------
MODELS = {
    "claude_sonnet":   "anthropic/claude-sonnet-4.6",
    "gemini_25":       "google/gemini-2.5-flash",
    "gpt4o":           "openai/gpt-4o",
    "gemma_4":         "google/gemma-4-31b-it",
    "qwen_vl":         "qwen/qwen3-vl-8b-instruct",
}

DEFAULT_MODEL = "gemini_25"  # Best cost/quality balance

# -----------------------------------------------------------------------------
# Dataset — OIA-ODIR (Ocular Disease Intelligent Recognition)
# Same patient: bilateral fundus images + age + sex + clinical keywords
# Solves the "unpaired dataset" problem — all modalities per patient
# -----------------------------------------------------------------------------
BASE_DIR   = "./OIA-ODIR-Merged"
IMAGE_DIR  = f"{BASE_DIR}/Images"
CSV_PATH   = f"{BASE_DIR}/all_annotations.csv"

# -----------------------------------------------------------------------------
# Evaluation settings
# -----------------------------------------------------------------------------
DEFAULT_NUM_SAMPLES = 200   # Stratified: 100 DR-positive + 100 DR-negative
RANDOM_SEED         = 42    # For reproducibility
OUTPUT_DIR          = "./results"

# -----------------------------------------------------------------------------
# Risk thresholds
# -----------------------------------------------------------------------------
HIGH_RISK_THRESHOLD = 70    # Score >= 70  → Tier 1 URGENT
MOD_RISK_THRESHOLD  = 40    # Score 40-69  → Tier 2 MODERATE

# -----------------------------------------------------------------------------
# Image quality thresholds
# -----------------------------------------------------------------------------
MIN_BRIGHTNESS  = 30
MAX_BRIGHTNESS  = 220
BLUR_THRESHOLD  = 100
