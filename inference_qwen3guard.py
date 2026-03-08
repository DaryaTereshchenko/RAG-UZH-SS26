"""
Inference script for Qwen3Guard-Gen-8B safety moderation model.

Runs each text through the model in a single pass using its chat template.
The model outputs Safety label + Categories, which are mapped to the same
5 policy keys as CoPE-A (harassment, hate_speech, stereotypes, violence, sexual)
so the results are compatible with evaluate.py.
"""

import argparse
import logging
import re
import sys
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("inference_qwen3guard.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3Guard-Gen-8B"
MAX_NEW_TOKENS = 128
MAX_RETRIES = 3

# Map Qwen3Guard categories -> our 5 policy keys
# A Qwen category can map to multiple policies.
CATEGORY_TO_POLICIES = {
    "Violent": ["violence"],
    "Non-violent Illegal Acts": [],
    "Sexual Content or Sexual Acts": ["sexual"],
    "PII": [],
    "Suicide & Self-Harm": [],
    "Unethical Acts": ["harassment", "hate_speech", "stereotypes"],
    "Politically Sensitive Topics": [],
    "Copyright Violation": [],
    "Jailbreak": [],
    "None": [],
}

POLICY_KEYS = ["harassment", "hate_speech", "stereotypes", "violence", "sexual"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_label_and_categories(content: str):
    """Parse Qwen3Guard output into safety label and category list."""
    safe_pattern = r"Safety:\s*(Safe|Unsafe|Controversial)"
    category_pattern = (
        r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|"
        r"PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
        r"Copyright Violation|Jailbreak|None)"
    )
    safe_match = re.search(safe_pattern, content)
    label = safe_match.group(1) if safe_match else None
    categories = re.findall(category_pattern, content)
    return label, categories


def map_categories_to_policies(safety_label: str, categories: list[str]) -> dict[str, str]:
    """
    Convert Qwen3Guard output to per-policy 0/1 predictions.

    If safety_label is "Safe" -> all policies are 0.
    If "Unsafe" or "Controversial" -> set 1 for policies matching the categories.
    """
    preds = {k: "0" for k in POLICY_KEYS}

    if safety_label == "Safe":
        return preds

    for cat in categories:
        for policy in CATEGORY_TO_POLICIES.get(cat, []):
            preds[policy] = "1"

    return preds


def load_model_and_tokenizer(device_map: str = "cuda"):
    """Load Qwen3Guard model and tokenizer."""
    logger.info("Loading tokenizer from %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    logger.info("Loading model %s (4-bit quantized)", MODEL_ID)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    model.eval()

    logger.info("Model loaded and ready for inference")
    return model, tokenizer


def run_inference(model, tokenizer, text: str) -> str:
    """Run Qwen3Guard on a single text using the chat template."""
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    output_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def infer_sample(model, tokenizer, text, row_label, max_retries):
    """
    Run inference on one sample with retry logic.
    Returns (policy_preds_dict, safety_label, categories_list, raw_output).
    """
    raw_output = ""
    safety_label = None
    categories = []

    for attempt in range(1, max_retries + 1):
        try:
            raw_output = run_inference(model, tokenizer, text)
            safety_label, categories = extract_label_and_categories(raw_output)

            if safety_label is not None:
                if attempt > 1:
                    logger.info(
                        "[%s] Parsed on retry %d: %s / %s",
                        row_label, attempt, safety_label, categories,
                    )
                preds = map_categories_to_policies(safety_label, categories)
                return preds, safety_label, categories, raw_output

            logger.warning(
                "[%s] Attempt %d/%d — could not parse safety label from: '%s'",
                row_label, attempt, max_retries, raw_output,
            )
        except Exception:
            logger.exception(
                "[%s] Attempt %d/%d — inference error",
                row_label, attempt, max_retries,
            )

        if attempt < max_retries:
            time.sleep(1)

    logger.error(
        "[%s] All %d attempts failed. Storing raw output.",
        row_label, max_retries,
    )
    preds = {k: "PARSE_ERROR" for k in POLICY_KEYS}
    return preds, safety_label, categories, raw_output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen3Guard-Gen-8B inference on CSV text entries."
    )
    parser.add_argument(
        "--input_csv", type=str, default="data/merged_data.csv",
        help="Path to input CSV (must have a 'text' column).",
    )
    parser.add_argument(
        "--output_csv", type=str, default="data/inference_results_qwen3guard.csv",
        help="Path to write the output CSV with predictions.",
    )
    parser.add_argument(
        "--max_retries", type=int, default=MAX_RETRIES,
        help="Max retries per sample when parsing fails.",
    )
    parser.add_argument(
        "--device_map", type=str, default="cuda",
        help="Device map for model loading (default: cuda).",
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_csv)
    logger.info("Loaded %d rows from %s", len(df), args.input_csv)

    if "text" not in df.columns:
        logger.error("Input CSV must contain a 'text' column.")
        sys.exit(1)

    # Load model
    model, tokenizer = load_model_and_tokenizer(device_map=args.device_map)

    # Prepare result storage
    results = {f"pred_{k}": [] for k in POLICY_KEYS}
    safety_labels = []
    qwen_categories = []
    raw_outputs = []
    total = len(df)

    for idx, row in df.iterrows():
        text = str(row["text"])
        row_label = f"{idx + 1}/{total}"

        preds, safety_label, categories, raw_output = infer_sample(
            model, tokenizer, text, row_label, args.max_retries,
        )

        for k in POLICY_KEYS:
            results[f"pred_{k}"].append(preds[k])

        safety_labels.append(safety_label)
        qwen_categories.append(", ".join(categories) if categories else "")
        raw_outputs.append(raw_output)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            logger.info("Progress: %d / %d samples processed", idx + 1, total)

    # Attach results
    for col, vals in results.items():
        df[col] = vals
    df["safety_label"] = safety_labels
    df["qwen_categories"] = qwen_categories
    df["raw_output"] = raw_outputs

    df.to_csv(args.output_csv, index=False)
    logger.info("Results saved to %s", args.output_csv)

    # Summary
    logger.info("Safety label distribution:\n%s",
                pd.Series(safety_labels).value_counts().to_string())
    for k in POLICY_KEYS:
        col = f"pred_{k}"
        logger.info("%s distribution:\n%s",
                    col, pd.Series(df[col]).value_counts().to_string())


if __name__ == "__main__":
    main()
