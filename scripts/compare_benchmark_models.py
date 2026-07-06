import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_eval import normalize_benchmark_dataset, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


def _score_rows(rows: Sequence[Dict[str, Any]], weights: Dict[str, float], threshold: float = 0.55) -> List[float]:
    scores = []
    for row in rows:
        score = (
            weights.get("semantic_risk", 0.0) * row["semantic_risk"]
            + weights.get("proxy_score", 0.0) * row["proxy_score"]
            + weights.get("velocity_1h", 0.0) * min(1.0, row["velocity_1h"] / 20.0)
            + weights.get("amount_deviation", 0.0) * min(1.0, row["amount_deviation"] / 1000.0)
        )
        scores.append(max(0.0, min(1.0, score)))
    return scores


def evaluate_variant(rows: Sequence[Dict[str, Any]], name: str, weights: Dict[str, float], threshold: float = 0.55) -> Dict[str, Any]:
    scores = _score_rows(rows, weights, threshold=threshold)
    labels = [row["is_fraud"] for row in rows]
    predictions = [1 if score > threshold else 0 for score in scores]
    metrics = compute_metrics(labels, scores)
    tp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)
    tn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 0)
    metrics["threshold"] = threshold
    metrics["predictions"] = predictions
    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["tn"] = tn
    return {"name": name, "weights": weights, "threshold": threshold, "metrics": metrics}


def run_comparison(input_path: str, source_type: str = "ieee_cis") -> List[Dict[str, Any]]:
    rows = normalize_benchmark_dataset(input_path, source_type=source_type)
    logging.info("Loaded %s rows from %s", len(rows), input_path)

    variants = [
        ("baseline", {"semantic_risk": 0.5, "proxy_score": 0.3, "velocity_1h": 0.2, "amount_deviation": 0.0}, 0.55),
        ("tuned_threshold", {"semantic_risk": 0.5, "proxy_score": 0.3, "velocity_1h": 0.2, "amount_deviation": 0.0}, 0.60),
        ("tuned_weights", {"semantic_risk": 0.45, "proxy_score": 0.35, "velocity_1h": 0.15, "amount_deviation": 0.05}, 0.55),
    ]

    results = []
    for name, weights, threshold in variants:
        variant_result = evaluate_variant(rows, name, weights, threshold=threshold)
        results.append(variant_result)
        metrics = variant_result["metrics"]
        logging.info(
            "Variant %s -> precision=%.4f recall=%.4f f1=%.4f auc_roc=%.4f threshold=%.2f",
            name,
            metrics.get("precision", 0.0),
            metrics.get("recall", 0.0),
            metrics.get("f1", 0.0),
            metrics.get("auc_roc", 0.0),
            threshold,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark model variants")
    parser.add_argument("input_path", help="Path or URL to the benchmark CSV file")
    parser.add_argument("--output_path", default=None, help="Where to save the comparison JSON")
    parser.add_argument("--source_type", default="ieee_cis", choices=["ieee_cis"], help="Benchmark source")
    args = parser.parse_args()

    results = run_comparison(args.input_path, source_type=args.source_type)
    output_path = Path(args.output_path or os.path.join("data", "benchmark_model_comparison.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    logging.info("Comparison saved to %s", output_path)


if __name__ == "__main__":
    main()
