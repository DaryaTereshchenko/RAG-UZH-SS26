"""
Evaluation script for CoPE-A-9B per-policy inference results.

Parses the 'type' column (ground truth) to build per-policy binary labels,
then evaluates each of the 5 policy predictions (pred_harassment,
pred_hate_speech, pred_stereotypes, pred_violence, pred_sexual) with
precision, recall, and F1.
"""

import argparse
import json
import logging
import sys

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping from raw type values to our 5 policy keys
# ---------------------------------------------------------------------------
TYPE_TO_POLICY = {
    "harassment": ["harassment"],
    "hate": ["hate_speech"],
    "stereotypes": ["stereotypes"],
    "violence": ["violence"],
    "sexual": ["sexual"],
}

POLICY_KEYS = ["harassment", "hate_speech", "stereotypes", "violence", "sexual"]


def parse_type_column(raw_value) -> set[str]:
    """
    Parse a single value from the 'type' column into a set of policy keys.

    Handles:
      - NaN / None -> empty set
      - Plain strings: "harassment", "hate", etc.
      - JSON dicts:  '{"choices":["hate","violence","harassment"]}'
    """
    if pd.isna(raw_value):
        return set()

    raw = str(raw_value).strip()

    # Try JSON
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            choices = obj.get("choices", [])
            policies = set()
            for c in choices:
                mapped_list = TYPE_TO_POLICY.get(c, [])
                policies.update(mapped_list)
            return policies
        except (json.JSONDecodeError, AttributeError):
            pass

    # Plain string
    mapped_list = TYPE_TO_POLICY.get(raw, [])
    if mapped_list:
        return set(mapped_list)

    return set()


def build_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary ground-truth columns (gt_harassment, gt_hate_speech, etc.)
    from the 'type' column.

    Rules:
      - If 'type' is present -> set the corresponding gt columns to 1,
        all others to 0.
      - If 'type' is NaN but 'label' == 0 -> all gt columns are 0
        (the sample was labelled as non-violating).
      - If 'type' is NaN and 'label' is also NaN or 1 -> we cannot build
        per-policy ground truth, so mark as NaN (unevaluable).
    """
    for key in POLICY_KEYS:
        df[f"gt_{key}"] = float("nan")

    for idx, row in df.iterrows():
        policies = parse_type_column(row.get("type"))
        label = row.get("label")

        if policies:
            # We know which policies apply -> 1 for those, 0 for the rest
            for key in POLICY_KEYS:
                df.at[idx, f"gt_{key}"] = 1 if key in policies else 0
        elif pd.notna(label) and float(label) == 0.0:
            # Labelled as non-violating -> all 0
            for key in POLICY_KEYS:
                df.at[idx, f"gt_{key}"] = 0
        # else: NaN — we can't evaluate this row per-policy

    return df


def evaluate_policy(y_true, y_pred, policy_name):
    """Compute and log metrics for one policy."""
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)

    logger.info("-" * 50)
    logger.info("Policy: %s", policy_name)
    logger.info("-" * 50)
    logger.info("  Samples   : %d", len(y_true))
    logger.info("  Precision : %.4f", p)
    logger.info("  Recall    : %.4f", r)
    logger.info("  F1 Score  : %.4f", f)

    report = classification_report(y_true, y_pred, zero_division=0)
    logger.info("  Classification report:\n%s", report)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    logger.info("  Confusion matrix (rows=true, cols=pred):\n%s", cm)

    return {"policy": policy_name, "precision": p, "recall": r, "f1": f, "n": len(y_true)}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate per-policy CoPE-A-9B predictions against 'type' ground truth."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/inference_results.csv",
        help="Path to CSV with pred_* columns and 'type'/'label' ground truth.",
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_csv)
    logger.info("Loaded %d rows from %s", len(df), args.input_csv)

    # Check required prediction columns
    pred_cols = [f"pred_{k}" for k in POLICY_KEYS]
    missing = [c for c in pred_cols if c not in df.columns]
    if missing:
        logger.error("Missing prediction columns: %s", missing)
        sys.exit(1)

    # Build per-policy ground truth
    df = build_ground_truth(df)

    # -----------------------------------------------------------------------
    # Evaluate each policy independently
    # -----------------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("PER-POLICY EVALUATION")
    logger.info("=" * 50)

    summaries = []

    for key in POLICY_KEYS:
        gt_col = f"gt_{key}"
        pred_col = f"pred_{key}"

        # Filter to evaluable rows: gt is not NaN and prediction is 0 or 1
        mask = df[gt_col].notna()
        df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
        mask &= df[pred_col].notna()

        sub = df.loc[mask].copy()
        if len(sub) == 0:
            logger.warning("No evaluable samples for policy '%s'", key)
            continue

        parse_errors = (df[gt_col].notna() & df[pred_col].isna()).sum()
        if parse_errors > 0:
            logger.warning(
                "  %s: dropped %d rows with PARSE_ERROR predictions", key, parse_errors
            )

        y_true = sub[gt_col].astype(int)
        y_pred = sub[pred_col].astype(int)

        result = evaluate_policy(y_true, y_pred, key)
        summaries.append(result)

    # -----------------------------------------------------------------------
    # Aggregate (micro-average across all 4 policies)
    # -----------------------------------------------------------------------
    if summaries:
        all_true = []
        all_pred = []
        for key in POLICY_KEYS:
            gt_col = f"gt_{key}"
            pred_col = f"pred_{key}"
            mask = df[gt_col].notna() & df[pred_col].notna()
            all_true.extend(df.loc[mask, gt_col].astype(int).tolist())
            all_pred.extend(df.loc[mask, pred_col].astype(int).tolist())

        micro_p = precision_score(all_true, all_pred, zero_division=0)
        micro_r = recall_score(all_true, all_pred, zero_division=0)
        micro_f = f1_score(all_true, all_pred, zero_division=0)

        logger.info("=" * 50)
        logger.info("AGGREGATE (micro-average across all policies)")
        logger.info("=" * 50)
        logger.info("  Precision : %.4f", micro_p)
        logger.info("  Recall    : %.4f", micro_r)
        logger.info("  F1 Score  : %.4f", micro_f)

        # Print summary table
        print("\n" + "=" * 60)
        print(f"{'Policy':<15} {'Prec':>8} {'Recall':>8} {'F1':>8} {'N':>6}")
        print("-" * 60)
        for s in summaries:
            print(f"{s['policy']:<15} {s['precision']:>8.4f} {s['recall']:>8.4f} {s['f1']:>8.4f} {s['n']:>6}")
        print("-" * 60)
        print(f"{'MICRO-AVG':<15} {micro_p:>8.4f} {micro_r:>8.4f} {micro_f:>8.4f} {len(all_true):>6}")
        print("=" * 60)


if __name__ == "__main__":
    main()
