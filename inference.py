"""
Inference script for CoPE-A-9B content policy evaluation model.

Loads google/gemma-2-9b base model with zentropi-ai/cope-a-9b LoRA adapter,
runs 5 per-policy prompts per sample and stores a 0/1 prediction for each:
  harassment, hate_speech, stereotypes, violence, sexual
"""

import argparse
import json
import logging
import re
import sys
import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "google/gemma-2-9b"
ADAPTER_ID = "zentropi-ai/cope-a-9b"
MAX_NEW_TOKENS = 16
MAX_RETRIES = 3

# Policy keys and their prompt files (relative to --prompt_dir)
POLICIES = [
    ("harassment", "harassment.txt"),
    ("hate_speech", "hate_speech.txt"),
    ("stereotypes", "stereotypes.txt"),
    ("violence", "violence.txt"),
    ("sexual", "sexual.txt"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompt_template(path: str) -> str:
    """Read the policy prompt template from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(template: str, content_text: str) -> str:
    """Fill the {content_text} placeholder in the prompt template."""
    return template.replace("{content_text}", content_text)


def parse_model_output(raw: str) -> str | None:
    """
    Extract the first occurrence of '0' or '1' from the model output.
    Returns None if no valid token is found.
    """
    match = re.search(r"\b([01])\b", raw)
    if match:
        return match.group(1)
    return None


def load_model_and_tokenizer(device_map: str = "auto"):
    """Load base model, merge LoRA adapter, and return (model, tokenizer)."""
    logger.info("Loading tokenizer from %s", BASE_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    logger.info("Loading base model %s", BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    logger.info("Loading LoRA adapter %s", ADAPTER_ID)
    model = PeftModel.from_pretrained(model, ADAPTER_ID)
    model = model.merge_and_unload()
    model.eval()

    logger.info("Model loaded and ready for inference")
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate raw text from the model given a full prompt string."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    # Decode only the newly generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def infer_single_policy(
    model, tokenizer, template, text, policy_name,
    row_label, total, max_retries,
):
    """Run inference for one policy on one text. Returns (prediction, raw_output)."""
    prompt = build_prompt(template, text)
    prediction = None
    raw_output = ""

    for attempt in range(1, max_retries + 1):
        try:
            raw_output = run_inference(model, tokenizer, prompt)
            prediction = parse_model_output(raw_output)

            if prediction is not None:
                if attempt > 1:
                    logger.info(
                        "[%s] %s — parsed on retry %d: %s",
                        row_label, policy_name, attempt, prediction,
                    )
                break

            logger.warning(
                "[%s] %s — attempt %d/%d could not parse: '%s'",
                row_label, policy_name, attempt, max_retries, raw_output,
            )
        except Exception:
            logger.exception(
                "[%s] %s — attempt %d/%d inference error",
                row_label, policy_name, attempt, max_retries,
            )

        if attempt < max_retries:
            time.sleep(1)

    if prediction is None:
        logger.error(
            "[%s] %s — all %d attempts failed. Storing raw output.",
            row_label, policy_name, max_retries,
        )
        prediction = "PARSE_ERROR"

    return prediction, raw_output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run CoPE-A-9B inference (4 policies) on CSV text entries."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/merged_data.csv",
        help="Path to input CSV (must have a 'text' column).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/inference_results.csv",
        help="Path to write the output CSV with predictions.",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="prompts/policy_prompts",
        help="Directory containing the 5 policy prompt files.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=MAX_RETRIES,
        help="Max retries per policy per sample when parsing fails.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading (default: auto).",
    )
    args = parser.parse_args()

    # Load all 4 prompt templates
    import os
    templates = {}
    for policy_key, filename in POLICIES:
        path = os.path.join(args.prompt_dir, filename)
        templates[policy_key] = load_prompt_template(path)
        logger.info("Loaded prompt template: %s -> %s", policy_key, path)

    # Load data
    df = pd.read_csv(args.input_csv)
    logger.info("Loaded %d rows from %s", len(df), args.input_csv)

    if "text" not in df.columns:
        logger.error("Input CSV must contain a 'text' column.")
        sys.exit(1)

    # Load model
    model, tokenizer = load_model_and_tokenizer(device_map=args.device_map)

    # Prepare result columns
    results = {f"pred_{key}": [] for key in [k for k, _ in POLICIES]}
    raw_results = {f"raw_{key}": [] for key in [k for k, _ in POLICIES]}

    total = len(df)

    for idx, row in df.iterrows():
        text = str(row["text"])
        row_label = f"{idx + 1}/{total}"

        for policy_key, _ in POLICIES:
            prediction, raw_output = infer_single_policy(
                model, tokenizer, templates[policy_key], text,
                policy_key, row_label, total, args.max_retries,
            )
            results[f"pred_{policy_key}"].append(prediction)
            raw_results[f"raw_{policy_key}"].append(raw_output)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            logger.info("Progress: %d / %d samples processed", idx + 1, total)

    # Attach results to dataframe
    for col, vals in {**results, **raw_results}.items():
        df[col] = vals

    df.to_csv(args.output_csv, index=False)
    logger.info("Results saved to %s", args.output_csv)

    # Summary per policy
    for policy_key, _ in POLICIES:
        col = f"pred_{policy_key}"
        logger.info(
            "%s distribution:\n%s",
            col, pd.Series(df[col]).value_counts().to_string(),
        )


if __name__ == "__main__":
    main()
