"""
Bedrock Semantic Feature Extractor (Mock).

In production, this calls Amazon Bedrock (Claude/Titan) to score
unstructured transaction descriptions. For local dev, it generates
a deterministic semantic_risk score based on heuristics.
"""
import json
import hashlib


def extract_semantic_features(**kwargs):
    """
    Simulates Bedrock LLM call to extract semantic risk from transaction text.
    In production: calls bedrock-runtime invoke_model with a prompt like:
      "Rate fraud risk 0-1 for: {transaction_description}"
    """
    sample_transactions = [
        {"desc": "Purchase 500 electronics units at $10 each", "order_id": "305-001"},
        {"desc": "Monthly subscription renewal", "order_id": "305-002"},
        {"desc": "Gift card bulk purchase 50x$200", "order_id": "305-003"},
    ]

    results = []
    for tx in sample_transactions:
        # Mock: deterministic "semantic" score based on text hash
        hash_val = int(hashlib.md5(tx["desc"].encode()).hexdigest()[:8], 16)
        semantic_risk = round((hash_val % 100) / 100.0, 2)

        results.append({
            "order_id": tx["order_id"],
            "semantic_risk": semantic_risk,
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "mock": True
        })

    print(f"Bedrock enrichment complete: {len(results)} transactions scored")
    for r in results:
        print(f"   {r['order_id']}: semantic_risk={r['semantic_risk']}")

    return results
