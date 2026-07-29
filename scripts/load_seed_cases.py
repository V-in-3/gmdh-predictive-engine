#!/usr/bin/env python3
"""Load seed fraud cases into pgvector for RAG demo."""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.vector_store import VectorStore

SEED_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fraud_cases_seed.json")


def features_to_embedding(features: dict) -> np.ndarray:
    """Convert feature dict to 32-dim embedding (same logic as DAG)."""
    base = np.array([
        features.get("semantic_risk", 0),
        features.get("velocity_1h", 0) / 50.0,
        features.get("proxy_score", 0),
        features.get("amount_deviation", 0) / 3.0,
    ], dtype=np.float32)
    return np.concatenate([base, base * 0.9, np.full(24, base.mean(), dtype=np.float32)])


def main():
    with open(SEED_FILE) as f:
        cases = json.load(f)

    store = VectorStore()
    loaded = 0

    for case in cases:
        embedding = features_to_embedding(case["features"])
        metadata = {
            "description": case["description"],
            "reason": case["reason"],
            "is_fraud": case["is_fraud"],
        }
        try:
            store.store_embedding(
                transaction_id=case["id"],
                embedding=embedding.tolist(),
                fraud_score=case["features"]["semantic_risk"],
                is_fraud=case["is_fraud"],
                metadata=metadata,
            )
            loaded += 1
        except Exception as e:
            print(f"  Skip {case['id']}: {e}")

    store.close()
    print(f"Loaded {loaded}/{len(cases)} seed cases into pgvector")


if __name__ == "__main__":
    main()
