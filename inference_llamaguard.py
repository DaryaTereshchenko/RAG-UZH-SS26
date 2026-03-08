"""
Inference script for meta-llama/Llama-Guard-3-8B using GGUF quantization
via llama-cpp-python.

Llama Guard 3 has a FIXED taxonomy of 14 safety categories (S1-S14) based on
the MLCommons hazard taxonomy.  Like Qwen3Guard, it does NOT accept custom
policy prompts — the categories are baked into the model's chat template.

    *** No new prompts are needed for this model. ***

GGUF source (default): mradermacher/Llama-Guard-3-8B-GGUF  (Q4_K_M, ~4.9 GB)

Usage:
    python inference_llamaguard.py \\
        --input_csv data/with_type.csv \\
        --output_csv data/inference_results_llamaguard.csv

If downloading fails with 401/403, you may need to:
  1. Accept the Llama 3.1 license at https://huggingface.co/meta-llama/Llama-Guard-3-8B
  2. Set HF_TOKEN env var  (huggingface-cli login)
"""

import argparse
import logging
import re
import sys
import time

import pandas as pd
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("inference_llamaguard.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_GGUF_REPO = "mradermacher/Llama-Guard-3-8B-GGUF"
DEFAULT_GGUF_FILE = "Llama-Guard-3-8B.Q4_K_M.gguf"
MAX_NEW_TOKENS = 64
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Llama Guard 3: 14 MLCommons safety categories → our 5 policy keys
# ---------------------------------------------------------------------------
CATEGORY_TO_POLICIES = {
    "S1":  ["violence"],                      # Violent Crimes
    "S2":  ["harassment"],                    # Non-Violent Crimes (threats, intimidation)
    "S3":  ["sexual"],                        # Sex-Related Crimes
    "S4":  ["sexual"],                        # Child Sexual Exploitation
    "S5":  [],                                # Defamation
    "S6":  [],                                # Specialized Advice
    "S7":  [],                                # Privacy
    "S8":  [],                                # Intellectual Property
    "S9":  ["violence"],                      # Indiscriminate Weapons
    "S10": ["hate_speech", "stereotypes"],    # Hate
    "S11": [],                                # Suicide & Self-Harm
    "S12": ["sexual"],                        # Sexual Content
    "S13": [],                                # Elections
    "S14": [],                                # Code Interpreter Abuse
}

POLICY_KEYS = ["harassment", "hate_speech", "stereotypes", "violence", "sexual"]

CATEGORY_NAMES = {
    "S1": "Violent Crimes",           "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",       "S4": "Child Sexual Exploitation",
    "S5": "Defamation",               "S6": "Specialized Advice",
    "S7": "Privacy",                   "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",   "S10": "Hate",
    "S11": "Suicide & Self-Harm",     "S12": "Sexual Content",
    "S13": "Elections",               "S14": "Code Interpreter Abuse",
}

# ---------------------------------------------------------------------------
# Fallback prompt template — used only when the GGUF's embedded chat
# template does not produce a parseable safety assessment.
# ---------------------------------------------------------------------------
MANUAL_PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "Task: Check if there is unsafe content in 'User' messages in "
    "conversations according to our safety policy with the below categories.\n\n"
    "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
    "S1: Violent Crimes.\n"
    "S2: Non-Violent Crimes.\n"
    "S3: Sex-Related Crimes.\n"
    "S4: Child Sexual Exploitation.\n"
    "S5: Defamation.\n"
    "S6: Specialized Advice.\n"
    "S7: Privacy.\n"
    "S8: Intellectual Property.\n"
    "S9: Indiscriminate Weapons.\n"
    "S10: Hate.\n"
    "S11: Suicide & Self-Harm.\n"
    "S12: Sexual Content.\n"
    "S13: Elections.\n"
    "S14: Code Interpreter Abuse.\n"
    "<END UNSAFE CONTENT CATEGORIES>\n\n"
    "<BEGIN CONVERSATION>\n\n"
    "User: {user_message}\n\n"
    "<END CONVERSATION>\n\n"
    "Provide your safety assessment for User in the above conversation:\n"
    " - First line must read 'safe' or 'unsafe'.\n"
    " - If unsafe, a second line must include a comma-separated list of "
    "violated categories."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def download_gguf(repo_id: str, filename: str) -> str:
    """Download GGUF file from HuggingFace Hub and return the local cache path."""
    logger.info("Downloading %s from %s …", filename, repo_id)
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    logger.info("GGUF cached at %s", path)
    return path


def load_model(gguf_path: str, n_gpu_layers: int = -1, n_ctx: int = 4096) -> Llama:
    """Load the GGUF model via llama-cpp-python."""
    logger.info(
        "Loading GGUF model from %s (gpu_layers=%d, ctx=%d)",
        gguf_path, n_gpu_layers, n_ctx,
    )
    llm = Llama(
        model_path=gguf_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False,
    )
    logger.info("Model loaded and ready for inference")
    return llm


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_output(raw_output: str):
    """
    Parse Llama Guard 3 output.

    Expected formats:
        "safe"
        "unsafe\\nS1,S3,S10"

    Returns (safety_label, list_of_category_codes).
    """
    raw_output = raw_output.strip()
    lines = [l.strip() for l in raw_output.split("\n") if l.strip()]

    safety_label = None
    categories = []

    if not lines:
        return safety_label, categories

    first = lines[0].lower()
    if first == "safe":
        safety_label = "safe"
    elif first == "unsafe":
        safety_label = "unsafe"
        if len(lines) > 1:
            # Category codes on the second line, e.g. "S1,S3,S10"
            cat_line = lines[1]
            categories = [c.strip() for c in cat_line.split(",") if c.strip()]

    return safety_label, categories


def map_categories_to_policies(
    safety_label: str, categories: list[str],
) -> dict[str, str]:
    """Convert Llama Guard output to per-policy 0/1 predictions."""
    preds = {k: "0" for k in POLICY_KEYS}

    if safety_label == "safe":
        return preds

    for cat in categories:
        for policy in CATEGORY_TO_POLICIES.get(cat, []):
            preds[policy] = "1"

    return preds


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference_chat(llm: Llama, text: str) -> str:
    """Try inference using the GGUF's embedded chat template."""
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": text}],
        max_tokens=MAX_NEW_TOKENS,
    )
    return response["choices"][0]["message"]["content"].strip()


def run_inference_manual(llm: Llama, text: str) -> str:
    """Fallback: build the Llama Guard prompt manually and call __call__."""
    prompt = MANUAL_PROMPT_TEMPLATE.format(user_message=text)
    output = llm(prompt, max_tokens=MAX_NEW_TOKENS, echo=False)
    return output["choices"][0]["text"].strip()


def infer_sample(
    llm: Llama, text: str, row_label: str, max_retries: int,
    use_manual_prompt: bool = False,
):
    """
    Run inference on one sample with retry logic.

    Returns (policy_preds_dict, safety_label, categories_list, raw_output).
    """
    raw_output = ""
    safety_label = None
    categories = []

    infer_fn = run_inference_manual if use_manual_prompt else run_inference_chat

    for attempt in range(1, max_retries + 1):
        try:
            raw_output = infer_fn(llm, text)
            safety_label, categories = parse_output(raw_output)

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
        "[%s] All %d attempts failed. Storing raw output.", row_label, max_retries,
    )
    preds = {k: "PARSE_ERROR" for k in POLICY_KEYS}
    return preds, safety_label, categories, raw_output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Llama-Guard-3-8B (GGUF) inference on CSV text entries.",
    )
    parser.add_argument(
        "--input_csv", type=str, default="data/merged_data.csv",
        help="Path to input CSV (must have a 'text' column).",
    )
    parser.add_argument(
        "--output_csv", type=str, default="data/inference_results_llamaguard.csv",
        help="Path to write the output CSV with predictions.",
    )
    parser.add_argument(
        "--gguf_repo", type=str, default=DEFAULT_GGUF_REPO,
        help="HuggingFace repo with the GGUF file.",
    )
    parser.add_argument(
        "--gguf_file", type=str, default=DEFAULT_GGUF_FILE,
        help="Filename of the GGUF quantization in the repo.",
    )
    parser.add_argument(
        "--gguf_path", type=str, default=None,
        help="Local path to a pre-downloaded GGUF. Overrides --gguf_repo/--gguf_file.",
    )
    parser.add_argument(
        "--n_gpu_layers", type=int, default=-1,
        help="Layers to offload to GPU (-1 = all, 0 = CPU only).",
    )
    parser.add_argument(
        "--n_ctx", type=int, default=4096,
        help="Context window size in tokens.",
    )
    parser.add_argument(
        "--max_retries", type=int, default=MAX_RETRIES,
        help="Max retries per sample when parsing fails.",
    )
    parser.add_argument(
        "--manual_prompt", action="store_true",
        help="Use manually constructed prompt instead of the GGUF chat template.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(args.input_csv)
    logger.info("Loaded %d rows from %s", len(df), args.input_csv)

    if "text" not in df.columns:
        logger.error("Input CSV must contain a 'text' column.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Download / locate GGUF
    # ------------------------------------------------------------------
    if args.gguf_path:
        gguf_path = args.gguf_path
    else:
        gguf_path = download_gguf(args.gguf_repo, args.gguf_file)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    llm = load_model(gguf_path, n_gpu_layers=args.n_gpu_layers, n_ctx=args.n_ctx)

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    results = {f"pred_{k}": [] for k in POLICY_KEYS}
    safety_labels = []
    lg_categories = []
    raw_outputs = []
    total = len(df)

    for idx, row in df.iterrows():
        text = str(row["text"])
        row_label = f"{idx + 1}/{total}"

        preds, safety_label, categories, raw_output = infer_sample(
            llm, text, row_label, args.max_retries,
            use_manual_prompt=args.manual_prompt,
        )

        for k in POLICY_KEYS:
            results[f"pred_{k}"].append(preds[k])

        safety_labels.append(safety_label)
        cat_names = [f"{c} ({CATEGORY_NAMES.get(c, '?')})" for c in categories]
        lg_categories.append(", ".join(cat_names) if cat_names else "")
        raw_outputs.append(raw_output)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            logger.info("Progress: %d / %d samples processed", idx + 1, total)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    for col, vals in results.items():
        df[col] = vals
    df["safety_label"] = safety_labels
    df["lg_categories"] = lg_categories
    df["raw_output"] = raw_outputs

    df.to_csv(args.output_csv, index=False)
    logger.info("Results saved to %s", args.output_csv)

    # Summary
    logger.info(
        "Safety label distribution:\n%s",
        pd.Series(safety_labels).value_counts().to_string(),
    )
    for k in POLICY_KEYS:
        col = f"pred_{k}"
        logger.info(
            "%s distribution:\n%s",
            col, pd.Series(df[col]).value_counts().to_string(),
        )


if __name__ == "__main__":
    main()
