"""
eval.py
=======
BLEU-score evaluation module for recycled content.

Usage (standalone):
    python -m src.eval "original post text" "recycled post text"
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring thresholds (from project spec)
# ---------------------------------------------------------------------------
BLEU_RANGES = [
    (0.0,  0.2,  "⚠️  Too different — core message may be lost"),
    (0.2,  0.5,  "✅  Ideal range — fresh but faithful to original"),
    (0.5,  1.01, "🔁  Too similar — not enough creative reworking"),
]

LOG_FILE: str = os.getenv("EVAL_LOG_FILE", "./eval_scores.jsonl")


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def calculate_bleu(original: str, recycled: str) -> float:
    """
    Compute the sentence-level BLEU score between `original` and `recycled`.

    Uses sacrebleu for corpus-level scoring and nltk as a fallback
    for single-sentence comparison.

    Returns a float in [0.0, 1.0].
    """
    # sacrebleu approach (preferred — handles edge cases well)
    try:
        import sacrebleu

        # sacrebleu corpus_bleu expects list of hypotheses and list-of-list references
        result = sacrebleu.corpus_bleu(
            hypotheses=[recycled],
            references=[[original]],
            smooth_method="exp",
        )
        return round(result.score / 100.0, 4)  # sacrebleu returns 0–100
    except Exception as e:
        log.debug("sacrebleu failed (%s), falling back to nltk BLEU", e)

    # nltk fallback
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        ref_tokens = original.lower().split()
        hyp_tokens = recycled.lower().split()
        smoother = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)
        return round(float(score), 4)
    except Exception as e:
        log.error("Both BLEU backends failed: %s", e)
        return 0.0


def interpret_bleu(score: float) -> str:
    """Return a human-readable interpretation of a BLEU score."""
    for low, high, label in BLEU_RANGES:
        if low <= score < high:
            return label
    return "Unknown range"


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

def evaluate(
    original: str,
    recycled: str,
    metadata: Optional[Dict[str, Any]] = None,
    log_to_file: bool = True,
) -> Dict[str, Any]:
    """
    Calculate BLEU, build a quality report, and optionally log to file.

    Parameters
    ----------
    original    : Original post text
    recycled    : LLM-generated recycled post text
    metadata    : Optional dict (e.g. platform info) to attach to the log entry
    log_to_file : If True, append result to EVAL_LOG_FILE (JSONL format)

    Returns
    -------
    dict with keys: bleu_score, interpretation, original, recycled, metadata, timestamp
    """
    score = calculate_bleu(original, recycled)
    interpretation = interpret_bleu(score)
    timestamp = datetime.utcnow().isoformat() + "Z"

    report = {
        "bleu_score": score,
        "interpretation": interpretation,
        "original": original,
        "recycled": recycled,
        "metadata": metadata or {},
        "timestamp": timestamp,
    }

    if log_to_file:
        _append_log(report)

    return report


def _append_log(report: Dict[str, Any], log_file: str = LOG_FILE) -> None:
    """Append a single evaluation result to a JSONL log file."""
    try:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(report) + "\n")
        log.debug("Evaluation logged to %s", log_file)
    except Exception as e:
        log.warning("Could not write eval log: %s", e)


def load_eval_log(log_file: str = LOG_FILE) -> List[Dict[str, Any]]:
    """Load all evaluation records from the JSONL log file."""
    path = Path(log_file)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def print_report(report: Dict[str, Any]) -> None:
    """Pretty-print an evaluation report to stdout."""
    score = report["bleu_score"]
    interp = report["interpretation"]

    bar_len = int(score * 40)
    bar = "█" * bar_len + "░" * (40 - bar_len)

    print("\n" + "=" * 70)
    print("  BLEU SCORE EVALUATION REPORT")
    print("=" * 70)
    print(f"  Score       : {score:.4f}")
    print(f"  Progress    : [{bar}]  {score:.1%}")
    print(f"  Assessment  : {interp}")
    print("-" * 70)
    print(f"  Original    : {report['original'][:120]}{'…' if len(report['original'])>120 else ''}")
    print(f"  Recycled    : {report['recycled'][:120]}{'…' if len(report['recycled'])>120 else ''}")
    if report.get("metadata"):
        meta = report["metadata"]
        print(f"  Platform    : {meta.get('source_platform','')} → {meta.get('target_platform','')}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate BLEU score between two texts.")
    parser.add_argument("original", help="Original post text")
    parser.add_argument("recycled", help="Recycled/generated post text")
    parser.add_argument("--no-log", action="store_true", help="Do not write to log file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    report = evaluate(args.original, args.recycled, log_to_file=not args.no_log)
    print_report(report)
