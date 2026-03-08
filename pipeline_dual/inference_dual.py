#!/usr/bin/env python3
"""
Two-stage hate-speech detection pipeline.

Stage 1 – Encoder pre-filters (parallel threads)
-------------------------------------------------
• pysentimiento/robertuito-hate-speech   – fine-tuned Spanish RoBERTa
  (multi-class softmax: hateful / not_hateful / ...)
• unitary/multilingual-toxic-xlm-roberta – multilingual XLM-R with
  independent sigmoid scores per toxic sub-category

Flagging strategy  (liberal / high-recall):
  A text is flagged when EITHER encoder's harm score >= --threshold (default
  0.35).  Rationale:
    • Using 0.35 instead of the standard 0.5 decision boundary lowers the
      individual-model miss rate before the expensive LLM step.
    • With OR across two independent models the joint miss rate is roughly
      miss_1 * miss_2 — e.g. ~0.15 * 0.20 = ~3 % — far lower than a single
      model at 0.5.
    • False positives in Stage 1 are acceptable: the per-policy LLM step will
      correctly output "0" for clean texts that were over-flagged.

Stage 2 – CoPE-A-9B (google/gemma-2-9b + zentropi-ai/cope-a-9b LoRA)
-----------------------------------------------------------------------
Only flagged texts are forwarded.  For each of the 5 policy prompts the model
produces a binary "0" / "1" answer.  Unflagged texts receive pred = "0".

Output CSV columns mirror inference_cope.py so that evaluate.py is reusable
without modification:
  original columns + score_roberto + score_toxic_xlm + encoder_flagged +
  pred_{policy} + raw_{policy}  for each of the 5 policies.

Typical invocation (run from repo root):
  python pipeline_dual/inference_dual.py \\
      --input_csv  data/merged_data.csv \\
      --output_csv pipeline_dual/results.csv \\
      --run_eval
"""

import argparse
import gc
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline as hf_pipeline,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline_dual.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROBERTUITO_ID  = "pysentimiento/robertuito-hate-speech"
TOXIC_XLM_ID   = "unitary/multilingual-toxic-xlm-roberta"
BASE_MODEL_ID  = "google/gemma-2-9b"
ADAPTER_ID     = "zentropi-ai/cope-a-9b"

# Liberal threshold: flag if EITHER encoder score >= this value.
DEFAULT_THRESHOLD  = 0.35
ENCODER_BATCH_SIZE = 32
MAX_NEW_TOKENS     = 16
MAX_RETRIES        = 3

POLICIES = [
    ("harassment",  "harassment.txt"),
    ("hate_speech", "hate_speech.txt"),
    ("stereotypes", "stereotypes.txt"),
    ("violence",    "violence.txt"),
    ("sexual",      "sexual.txt"),
]

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def resolve_encoder_device(device_str: str):
    """Convert CLI 'cuda' / 'cuda:N' / 'cpu' to what HF pipeline expects."""
    if device_str == "cuda":
        return 0
    m = re.fullmatch(r"cuda:(\d+)", device_str)
    if m:
        return int(m.group(1))
    return device_str  # "cpu" passthrough


# ---------------------------------------------------------------------------
# Stage 1 – Encoder scoring functions
# ---------------------------------------------------------------------------

def harm_score_roberto(label_scores: list[dict]) -> float:
    """
    Collapse robertuito label probabilities into a single harm score.

    The model is a multi-class softmax classifier.  Labels whose normalised
    name does NOT appear in safe_markers contribute to harm.  For a typical
    two-class ('hateful' / 'not_hateful') model this equals 1 – P(not_hateful).
    The function generalises to models with additional fine-grained classes
    (e.g. 'aggressive', 'targeted') by summing all non-safe probabilities.
    """
    safe_markers = frozenset({"not_hateful", "none", "neutral", "label_0"})
    total = 0.0
    for item in label_scores:
        label_norm = item["label"].lower().replace("-", "_").replace(" ", "_")
        if label_norm not in safe_markers:
            total += item["score"]
    return min(total, 1.0)


def harm_score_toxic(label_scores: list[dict]) -> float:
    """
    Return the maximum sigmoid score across all toxic sub-category labels.

    multilingual-toxic-xlm-roberta outputs independent sigmoids for each of:
    toxic, severe_toxic, obscene, threat, insult, identity_hate.  The maximum
    gives the most conservative (highest) single harm estimate.
    """
    return max(item["score"] for item in label_scores) if label_scores else 0.0


def score_all_texts(
    pipe,
    texts: list[str],
    score_fn,
    name: str,
) -> list[float]:
    """
    Run a HF text-classification pipeline over all texts (batched internally
    via the pipeline's batch_size) and convert each output to a scalar harm
    probability using score_fn.

    Falls back to per-text inference if the batched call fails.
    """
    logger.info("[%s] Scoring %d texts …", name, len(texts))
    try:
        raw = pipe(texts)  # list[list[dict]] when top_k=None
    except Exception:
        logger.exception(
            "[%s] Batched scoring failed; falling back to per-text inference", name
        )
        raw = []
        for t in texts:
            try:
                result = pipe(t)
                raw.append(result if isinstance(result, list) else [result])
            except Exception:
                logger.exception("[%s] per-text inference failed; defaulting to 0.0", name)
                raw.append([])

    scores: list[float] = []
    for item in raw:
        if isinstance(item, dict):  # single-label output (shouldn't occur with top_k=None)
            item = [item]
        scores.append(score_fn(item) if item else 0.0)

    logger.info("[%s] Scoring complete.", name)
    return scores


def run_encoders_parallel(
    texts: list[str],
    encoder_device,
) -> tuple[list[float], list[float]]:
    """
    Load both encoder models onto encoder_device, score all texts with each
    model in *parallel threads*, then free both models before returning.

    NOTE: on a single-GPU host CUDA operations from the two threads are
    serialised through the default CUDA stream, so GPU time is multiplexed
    rather than truly overlapped.  The parallelism buys overlapped CPU
    preprocessing (tokenisation, post-processing) and clean code structure.
    On multi-GPU or CPU-only hosts both models can run fully concurrently.
    """
    logger.info("Loading encoder 1: %s  (device=%s)", ROBERTUITO_ID, encoder_device)
    pipe_roberto = hf_pipeline(
        "text-classification",
        model=ROBERTUITO_ID,
        device=encoder_device,
        top_k=None,       # return scores for ALL labels
        truncation=True,
        max_length=512,
        batch_size=ENCODER_BATCH_SIZE,
    )

    logger.info("Loading encoder 2: %s  (device=%s)", TOXIC_XLM_ID, encoder_device)
    pipe_toxic = hf_pipeline(
        "text-classification",
        model=TOXIC_XLM_ID,
        device=encoder_device,
        top_k=None,
        truncation=True,
        max_length=512,
        batch_size=ENCODER_BATCH_SIZE,
    )

    logger.info("Dispatching both encoders to parallel threads …")
    with ThreadPoolExecutor(max_workers=2) as exe:
        fut_roberto = exe.submit(
            score_all_texts, pipe_roberto, texts, harm_score_roberto, "robertuito"
        )
        fut_toxic = exe.submit(
            score_all_texts, pipe_toxic, texts, harm_score_toxic, "toxic-xlm"
        )
        scores_roberto = fut_roberto.result()
        scores_toxic   = fut_toxic.result()

    # Release encoder VRAM before loading the large LLM
    del pipe_roberto, pipe_toxic
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Encoder models unloaded; VRAM reclaimed.")

    return scores_roberto, scores_toxic


# ---------------------------------------------------------------------------
# Stage 2 – CoPE-A-9B helpers  (mirrored from inference_cope.py)
# ---------------------------------------------------------------------------

def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def build_prompt(template: str, content_text: str) -> str:
    return template.replace("{content_text}", content_text)


def parse_model_output(raw: str) -> str | None:
    match = re.search(r"\b([01])\b", raw)
    return match.group(1) if match else None


def load_cope_model(device_map: str):
    logger.info("Loading base model %s (4-bit quantized) …", BASE_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=bnb_config,
    )
    logger.info("Applying LoRA adapter %s …", ADAPTER_ID)
    model = PeftModel.from_pretrained(model, ADAPTER_ID)
    model = model.merge_and_unload()
    model.eval()
    logger.info("CoPE model ready.")
    return model, tokenizer


def cope_generate(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    new_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def infer_single_policy(
    model,
    tokenizer,
    template: str,
    text: str,
    policy_name: str,
    row_label: str,
) -> tuple[str, str]:
    """Run one policy prompt on one text with retry logic. Returns (prediction, raw)."""
    prompt = build_prompt(template, text)
    prediction: str | None = None
    raw_output = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_output = cope_generate(model, tokenizer, prompt)
            prediction = parse_model_output(raw_output)
            if prediction is not None:
                if attempt > 1:
                    logger.info(
                        "[%s] %s – parsed on retry %d: %s",
                        row_label, policy_name, attempt, prediction,
                    )
                break
            logger.warning(
                "[%s] %s – attempt %d/%d unparseable output: '%s'",
                row_label, policy_name, attempt, MAX_RETRIES, raw_output,
            )
        except Exception:
            logger.exception(
                "[%s] %s – attempt %d/%d inference error",
                row_label, policy_name, attempt, MAX_RETRIES,
            )
        if attempt < MAX_RETRIES:
            time.sleep(1)

    if prediction is None:
        logger.error(
            "[%s] %s – all %d attempts failed; storing raw output",
            row_label, policy_name, MAX_RETRIES,
        )
        prediction = "PARSE_ERROR"

    return prediction, raw_output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage encoder→LLM hate-speech pipeline. "
            "Encoder pre-filters flag texts above --threshold; "
            "CoPE-A-9B then runs 5 per-policy prompts on flagged texts only."
        )
    )
    parser.add_argument(
        "--input_csv",
        default="data/merged_data.csv",
        help="Input CSV; must have a 'text' column.",
    )
    parser.add_argument(
        "--output_csv",
        default="pipeline_dual/results.csv",
        help="Output CSV path (parent directory created if absent).",
    )
    parser.add_argument(
        "--prompt_dir",
        default="prompts/policy_prompts",
        help="Directory containing the 5 policy prompt .txt files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Harm-score threshold for encoder flagging (default: %(default)s). "
            "Liberal — texts are forwarded to the LLM if EITHER encoder scores "
            ">= this value."
        ),
    )
    parser.add_argument(
        "--encoder_device",
        default="cuda",
        help="Device for encoder models: 'cuda', 'cuda:N', or 'cpu'.",
    )
    parser.add_argument(
        "--llm_device_map",
        default="cuda",
        help="device_map passed to AutoModelForCausalLM (e.g. 'cuda', 'auto').",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Run evaluate.py on the output CSV after inference completes.",
    )
    parser.add_argument(
        "--eval_script",
        default="evaluate.py",
        help="Path to evaluate.py relative to CWD (default: %(default)s).",
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
    texts = df["text"].fillna("").astype(str).tolist()

    # ------------------------------------------------------------------
    # Stage 1 – Encoder pre-filtering
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 1  Encoder pre-filtering  (threshold=%.2f, logic=OR)", args.threshold)
    logger.info("=" * 60)

    enc_device = resolve_encoder_device(args.encoder_device)
    scores_roberto, scores_toxic = run_encoders_parallel(texts, enc_device)

    flagged = [
        scores_roberto[i] >= args.threshold or scores_toxic[i] >= args.threshold
        for i in range(len(texts))
    ]
    n_flagged = sum(flagged)
    logger.info(
        "Flagged %d / %d texts  (%.1f%%)",
        n_flagged, len(texts), 100.0 * n_flagged / max(len(texts), 1),
    )

    # Persist encoder diagnostics in the output dataframe
    df["score_roberto"]   = scores_roberto
    df["score_toxic_xlm"] = scores_toxic
    df["encoder_flagged"] = [int(f) for f in flagged]

    # Initialise all 5 prediction columns to "0" (default for unflagged texts)
    for policy_key, _ in POLICIES:
        df[f"pred_{policy_key}"] = "0"
        df[f"raw_{policy_key}"]  = ""

    # ------------------------------------------------------------------
    # Stage 2 – CoPE-A-9B per-policy inference on flagged texts
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2  CoPE-A-9B inference on %d flagged texts", n_flagged)
    logger.info("=" * 60)

    # Load prompt templates (same prompts as inference_cope.py)
    templates: dict[str, str] = {}
    for policy_key, filename in POLICIES:
        path = os.path.join(args.prompt_dir, filename)
        templates[policy_key] = load_prompt_template(path)
        logger.info("Prompt loaded: %s  ->  %s", policy_key, path)

    if n_flagged > 0:
        model, tokenizer = load_cope_model(args.llm_device_map)
        flagged_indices = [i for i, f in enumerate(flagged) if f]
        total_flagged   = len(flagged_indices)

        for rank, idx in enumerate(flagged_indices, 1):
            text      = texts[idx]
            row_label = f"{rank}/{total_flagged}"

            for policy_key, _ in POLICIES:
                pred, raw = infer_single_policy(
                    model, tokenizer,
                    templates[policy_key],
                    text, policy_key, row_label,
                )
                df.at[idx, f"pred_{policy_key}"] = pred
                df.at[idx, f"raw_{policy_key}"]  = raw

            if rank % 50 == 0 or rank == total_flagged:
                logger.info("LLM progress: %d / %d flagged samples processed", rank, total_flagged)

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.warning("No texts were flagged by encoders; all policy predictions set to '0'.")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info("Results saved to %s", args.output_csv)

    # Per-policy prediction distribution (flagged rows only for signal)
    flagged_df = df[df["encoder_flagged"] == 1]
    for policy_key, _ in POLICIES:
        col = f"pred_{policy_key}"
        logger.info(
            "%s distribution (all rows):\n%s",
            col, df[col].value_counts().to_string(),
        )
        if len(flagged_df):
            logger.info(
                "%s distribution (flagged rows only):\n%s",
                col, flagged_df[col].value_counts().to_string(),
            )

    # ------------------------------------------------------------------
    # Optional evaluation
    # ------------------------------------------------------------------
    if args.run_eval:
        logger.info("=" * 60)
        logger.info("Running evaluation via %s", args.eval_script)
        logger.info("=" * 60)
        cmd = [sys.executable, args.eval_script, "--input_csv", args.output_csv]
        logger.info("Command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
