"""Prepare IEEE-CIS Fraud Detection dataset for GMDH training.

Download the dataset (free, requires Kaggle account):
    1. Register at https://www.kaggle.com
    2. Go to https://www.kaggle.com/c/ieee-fraud-detection/data
    3. Accept competition rules and download train_transaction.csv (~500 MB)

Usage:
    python scripts/prepare_ieee_cis.py /path/to/train_transaction.csv
    python scripts/prepare_ieee_cis.py /path/to/train_transaction.csv --output_path data/fraud_transactions_ieee.csv

The script maps IEEE-CIS columns to the GMDH training schema:

    semantic_risk    <- DeviceInfo + card type heuristic
                       (approximates LLM semantic risk without calling an LLM)
    velocity_1h      <- TransactionDT normalized to [0, 1]
                       (transaction time as velocity proxy)
    proxy_score      <- P_emaildomain + addr1
                       (email domain and address anonymity indicator)
    amount_deviation <- TransactionAmt normalized to [0, 1]
    is_fraud         <- isFraud (ground truth label)

After preparation, train GMDH on the IEEE-CIS data:

    python -c "
    from jobs.gmdh_fraud_trainer import train_fraud_model
    train_fraud_model('data/fraud_transactions_ieee.csv', 'data/fraud_model_coeffs.json')
    "

Then run the benchmark to validate the trained model:

    python scripts/benchmark_eval.py data/fraud_production_50k.csv
"""

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

# TransactionAmt 95th percentile in IEEE-CIS is ~500. Normalize to [0, 1].
AMOUNT_SCALE = 500.0

# TransactionDT max is ~15_811_131 seconds (~6 months). Used as velocity proxy.
DT_SCALE = 15_811_131.0

# Email domains associated with higher fraud probability in IEEE-CIS research.
HIGH_RISK_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", ""}


def _semantic_risk(row: Dict[str, Any]) -> float:
    """Heuristic semantic risk from device and card signals.

    In the absence of a real LLM, we approximate semantic_risk using features
    that correlate with fraud in IEEE-CIS research. This allows the same GMDH
    training schema to work with IEEE-CIS data as with the LLM-enriched data.
    """
    device = str(row.get("DeviceInfo", "") or "").lower()
    card4 = str(row.get("card4", "") or "").lower()
    card6 = str(row.get("card6", "") or "").lower()

    risk = 0.2

    # Unknown or high-risk device types are more common in fraud.
    if not device or device in ("", "nan"):
        risk += 0.2

    # Debit cards have higher fraud rates than credit in IEEE-CIS.
    if card4 in ("visa", "mastercard") and card6 == "debit":
        risk += 0.15

    # Mobile transactions have slightly higher fraud rates.
    if str(row.get("DeviceType", "") or "").lower() == "mobile":
        risk += 0.1

    # High transaction amounts correlate with fraud.
    try:
        amt = float(row.get("TransactionAmt", 0) or 0)
        if amt > 300:
            risk += 0.2
        elif amt > 100:
            risk += 0.05
    except (ValueError, TypeError):
        pass

    return min(1.0, risk)


def _velocity_1h(row: Dict[str, Any]) -> float:
    """Normalize TransactionDT as a velocity proxy."""
    try:
        return min(1.0, float(row.get("TransactionDT", 0) or 0) / DT_SCALE)
    except (ValueError, TypeError):
        return 0.0


def _proxy_score(row: Dict[str, Any]) -> float:
    """Heuristic proxy/anonymity score from email domain and address."""
    domain = str(row.get("P_emaildomain", "") or "").lower().strip()

    risk = 0.1

    # High-risk or missing email domains.
    if domain in HIGH_RISK_DOMAINS:
        risk += 0.3

    # Missing billing address is a strong fraud indicator in IEEE-CIS.
    try:
        addr = float(row.get("addr1", 0) or 0)
        if addr == 0:
            risk += 0.25
    except (ValueError, TypeError):
        risk += 0.25

    # Missing recipient email domain also signals anonymity.
    r_domain = str(row.get("R_emaildomain", "") or "").lower().strip()
    if r_domain in HIGH_RISK_DOMAINS:
        risk += 0.1

    return min(1.0, risk)


def _amount_deviation(row: Dict[str, Any]) -> float:
    """Normalize TransactionAmt to a [0, 1] deviation scale."""
    try:
        return min(1.0, float(row.get("TransactionAmt", 0) or 0) / AMOUNT_SCALE)
    except (ValueError, TypeError):
        return 0.0


def prepare_ieee_cis(input_path: str, output_path: str) -> None:
    """Normalize IEEE-CIS train_transaction.csv to GMDH training schema."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"IEEE-CIS dataset not found: {input_path}\n"
            "Download train_transaction.csv from:\n"
            "  https://www.kaggle.com/c/ieee-fraud-detection/data"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Reading IEEE-CIS dataset from %s", input_path)
    rows_written = 0
    fraud_count = 0

    with open(input_path, "r", newline="", encoding="utf-8") as src, \
         open(output_path, "w", newline="", encoding="utf-8") as dst:

        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=[
            "semantic_risk", "velocity_1h", "proxy_score", "amount_deviation", "is_fraud",
        ])
        writer.writeheader()

        for i, row in enumerate(reader, start=1):
            is_fraud = int(float(row.get("isFraud", 0) or 0) > 0)
            writer.writerow({
                "semantic_risk": round(_semantic_risk(row), 6),
                "velocity_1h": round(_velocity_1h(row), 6),
                "proxy_score": round(_proxy_score(row), 6),
                "amount_deviation": round(_amount_deviation(row), 6),
                "is_fraud": is_fraud,
            })
            rows_written += 1
            fraud_count += is_fraud

            if i % 50_000 == 0:
                pct = round(i / 590_540 * 100, 1)
                logging.info("processed %s rows (%s%%)", i, pct)

    fraud_rate = fraud_count / rows_written if rows_written else 0.0
    logging.info(
        "Done. rows=%s fraud_count=%s fraud_rate=%.4f -> %s",
        rows_written, fraud_count, fraud_rate, output_path,
    )
    logging.info(
        "Next step: train GMDH on this data:\n"
        "  python -c \"from jobs.gmdh_fraud_trainer import train_fraud_model; "
        "train_fraud_model('%s', 'data/fraud_model_coeffs.json')\"",
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare IEEE-CIS Fraud Detection dataset for GMDH training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_path",
        help="Path to train_transaction.csv from the IEEE-CIS Kaggle dataset",
    )
    parser.add_argument(
        "--output_path",
        default="data/fraud_transactions_ieee.csv",
        help="Output path for the normalized CSV (default: data/fraud_transactions_ieee.csv)",
    )
    args = parser.parse_args()
    prepare_ieee_cis(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
