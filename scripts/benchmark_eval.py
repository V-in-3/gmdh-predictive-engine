import argparse
import csv
import io
import json
import logging
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


def normalize_benchmark_dataset(data: Any, source_type: str = "ieee_cis") -> List[Dict[str, Any]]:
    """Normalize a public fraud benchmark into the schema expected by this project."""
    source_type = (source_type or "ieee_cis").lower()
    if source_type != "ieee_cis":
        raise ValueError(f"Unsupported benchmark source: {source_type}")

    rows = _load_rows(data)
    total_rows = len(rows)
    normalized_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        normalized_rows.append({
            "semantic_risk": _to_unit_interval(_extract_field(row, ["semantic_risk", "DeviceInfo", "cardholder_risk", "TransactionAmt"])),
            "velocity_1h": _to_number(_extract_field(row, ["velocity_1h", "tx_count_1h", "TransactionDT"])),
            "proxy_score": _to_unit_interval(_extract_field(row, ["proxy_score", "proxy_flag", "Proxy"])),
            "amount_deviation": _to_number(_extract_field(row, ["amount_deviation", "TransactionAmt"])),
            "is_fraud": _to_binary(_extract_field(row, ["is_fraud", "isFraud", "Class"])),
        })
        if total_rows and index % 5000 == 0:
            percent = round((index / total_rows) * 100, 1)
            logging.info("processed %s/%s rows (%s%%)", index, total_rows, percent)

    return [row for row in normalized_rows if row["is_fraud"] is not None]


def _load_rows(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, (str, os.PathLike)):
        value = str(data)
        if value.startswith(("http://", "https://")):
            return _load_rows_from_url(value)
        with open(value, "r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if isinstance(data, dict):
        return [data]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return [dict(row) for row in data]
    raise TypeError("Unsupported benchmark data type")


def _load_rows_from_url(url: str) -> List[Dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        payload = response.read()

    if _looks_like_zip(payload):
        return _load_rows_from_zip(payload)

    text = payload.decode("utf-8", errors="ignore")
    if "<html" in text.lower():
        raise ValueError(f"The URL did not resolve to CSV data: {url}")
    return list(csv.DictReader(text.splitlines()))


def _load_rows_from_zip(payload: bytes) -> List[Dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError("No CSV files were found in the downloaded archive")
        with archive.open(csv_files[0]) as handle:
            text = handle.read().decode("utf-8", errors="ignore")
    return list(csv.DictReader(text.splitlines()))


def _looks_like_zip(payload: bytes) -> bool:
    return payload.startswith(b"PK\x03\x04")


def _extract_field(row: Dict[str, Any], candidates: Sequence[str]) -> Any:
    for name in candidates:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _to_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_unit_interval(value: Any) -> float:
    number = _to_number(value)
    return max(0.0, min(1.0, number))


def _to_binary(value: Any) -> int:
    try:
        return int(float(value) > 0)
    except (TypeError, ValueError):
        return 0


def run_benchmark_evaluation(input_path: str, output_path: Optional[str] = None, source_type: str = "ieee_cis") -> dict:
    """Load a benchmark CSV, normalize it, and save the normalized dataset."""
    logging.info("benchmark evaluation started for %s", input_path)
    logging.info("loading rows")
    normalized = normalize_benchmark_dataset(input_path, source_type=source_type)

    output_dir = Path(output_path or os.path.join("data", "benchmark_normalized.csv"))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(output_dir, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["semantic_risk", "velocity_1h", "proxy_score", "amount_deviation", "is_fraud"])
        writer.writeheader()
        writer.writerows(normalized)

    fraud_rate = sum(row["is_fraud"] for row in normalized) / len(normalized) if normalized else 0.0
    logging.info("benchmark evaluation completed; rows=%s fraud_rate=%.4f", len(normalized), fraud_rate)
    return {
        "rows": int(len(normalized)),
        "output_path": str(output_dir),
        "fraud_rate": fraud_rate,
    }


def evaluate_benchmark_model(
    input_path: str,
    output_path: Optional[str] = None,
    model_output_path: Optional[str] = None,
    metrics_output_path: Optional[str] = None,
    source_type: str = "ieee_cis",
) -> dict:
    """Normalize a benchmark, train a simple heuristic model, and emit metrics."""
    eval_result = run_benchmark_evaluation(input_path, output_path=output_path, source_type=source_type)
    logging.info("loading rows")
    normalized_rows = normalize_benchmark_dataset(input_path, source_type=source_type)

    if not normalized_rows:
        raise ValueError("No rows were available for evaluation")

    logging.info("normalizing rows")

    # Try to load the trained GMDH model coefficients.
    # If found  -> score using the real polynomial (beta0 + betas * features).
    # If missing -> fall back to fixed rule-based weights and log a warning so the
    #               operator knows the benchmark is not reflecting the trained model.
    _script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gmdh_model_path = os.path.join(_script_dir, "data", "fraud_model_coeffs.json")
    gmdh_model = None
    if os.path.exists(gmdh_model_path):
        with open(gmdh_model_path, "r", encoding="utf-8") as _fh:
            gmdh_model = json.load(_fh)
        logging.info("scoring with trained GMDH model from %s", gmdh_model_path)
    else:
        logging.warning(
            "Trained GMDH model not found at %s — using rule-based fallback weights. "
            "Run train_fraud_model task first to enable GMDH-based benchmark scoring.",
            gmdh_model_path,
        )

    scores = []
    labels = []
    for row in normalized_rows:
        if gmdh_model:
            features = [
                row["semantic_risk"],
                row["velocity_1h"],
                row["proxy_score"],
                row["amount_deviation"],
            ]
            score = gmdh_model["beta0"] + sum(
                b * x for b, x in zip(gmdh_model["betas"], features)
            )
            score = max(0.0, min(1.0, score))
        else:
            score = (
                0.5 * row["semantic_risk"]
                + 0.3 * row["proxy_score"]
                + 0.2 * min(1.0, row["velocity_1h"] / 20.0)
            )
        scores.append(score)
        labels.append(row["is_fraud"])

    logging.info("computing metrics")
    metrics = compute_metrics(labels, scores)
    logging.info(
        "benchmark run finished: rows=%s precision=%.4f recall=%.4f f1=%.4f auc_roc=%.4f",
        len(labels),
        metrics.get("precision", 0.0),
        metrics.get("recall", 0.0),
        metrics.get("f1", 0.0),
        metrics.get("auc_roc", 0.0),
    )

    if gmdh_model:
        model_payload = {
            "description": "gmdh_trained_fraud_model",
            "source": gmdh_model_path,
            "beta0": gmdh_model.get("beta0"),
            "betas": gmdh_model.get("betas"),
        }
    else:
        model_payload = {
            "description": "simple_rule_based_fraud_model",
            "weights": {
                "semantic_risk": 0.5,
                "proxy_score": 0.3,
                "velocity_1h": 0.2,
            },
        }

    logging.info("writing outputs")
    model_output = Path(model_output_path or os.path.join("data", "benchmark_model.json"))
    model_output.parent.mkdir(parents=True, exist_ok=True)
    with open(model_output, "w", encoding="utf-8") as handle:
        json.dump(model_payload, handle, indent=2)

    metrics_output = Path(metrics_output_path or os.path.join("data", "benchmark_metrics.json"))
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {
        "rows": eval_result["rows"],
        "output_path": eval_result["output_path"],
        "fraud_rate": eval_result["fraud_rate"],
        "model_output_path": str(model_output),
        "metrics_output_path": str(metrics_output),
        "metrics": metrics,
    }


def compute_metrics(labels: List[int], scores: List[float]) -> Dict[str, float]:
    """Compute precision, recall, F1, and AUC-ROC for a set of scores."""
    if not labels or not scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auc_roc": 0.0}

    threshold = 0.5
    predictions = [1 if score > threshold else 0 for score in scores]

    tp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)
    tn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    auc_roc = compute_auc_roc(labels, scores)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_auc_roc(labels: List[int], scores: List[float]) -> float:
    """Compute a fast AUC-ROC approximation from sorted scores."""
    if not labels or not scores:
        return 0.0

    pairs = list(zip(scores, labels))
    pairs.sort(key=lambda item: item[0])

    positive_count = sum(label == 1 for _, label in pairs)
    negative_count = len(pairs) - positive_count
    if positive_count == 0 or negative_count == 0:
        return 0.0

    rank_sum = 0.0
    current_rank = 1
    start_index = 0
    while start_index < len(pairs):
        end_index = start_index + 1
        while end_index < len(pairs) and pairs[end_index][0] == pairs[start_index][0]:
            end_index += 1

        average_rank = (current_rank + end_index - 1) / 2.0
        for idx in range(start_index, end_index):
            if pairs[idx][1] == 1:
                rank_sum += average_rank
        current_rank = end_index + 1
        start_index = end_index

    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize an open fraud benchmark for this project")
    parser.add_argument("input_path", help="Path or URL to the benchmark CSV file")
    parser.add_argument("--output_path", default=None, help="Where to save the normalized CSV")
    parser.add_argument("--model_output_path", default=None, help="Where to save the model descriptor")
    parser.add_argument("--metrics_output_path", default=None, help="Where to save the metrics JSON")
    parser.add_argument("--source_type", default="ieee_cis", choices=["ieee_cis"], help="Benchmark source")
    args = parser.parse_args()

    result = evaluate_benchmark_model(
        args.input_path,
        output_path=args.output_path,
        model_output_path=args.model_output_path,
        metrics_output_path=args.metrics_output_path,
        source_type=args.source_type,
    )
    print(result)
