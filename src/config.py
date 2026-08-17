from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LORA_DIR = RESULTS_DIR / "lora_adapter"

EVAL_SET_PATH = DATA_DIR / "eval_set.jsonl"

STAGE1_RESULTS_PATH = RESULTS_DIR / "stage1_framing_results.jsonl"
STAGE1_SUMMARY_PATH = RESULTS_DIR / "stage1_summary.json"

STAGE2_RESULTS_PATH = RESULTS_DIR / "stage2_pgd_results.jsonl"
STAGE2_SUMMARY_PATH = RESULTS_DIR / "stage2_summary.json"

STAGE3_TRAIN_DATA_PATH = DATA_DIR / "stage3_reward_hack_data.jsonl"
STAGE3_TRAIN_LOG_PATH = RESULTS_DIR / "stage3_train_log.json"

STAGE4_RESULTS_PATH = RESULTS_DIR / "stage4_transfer_results.jsonl"
STAGE4_SUMMARY_PATH = RESULTS_DIR / "stage4_summary.json"

LOCAL_MODEL_PATH = "/home/beluga/.cache/qwen2.5-1.5b-instruct"

MAX_NEW_TOKENS = 200
GENERATION_TEMPERATURE = 0.0

RANDOM_SEED = 0
