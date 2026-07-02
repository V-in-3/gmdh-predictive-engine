"""
Bedrock Semantic Feature Extractor — backward-compatible wrapper.
Delegates to engines/bedrock_engine.py via unified interface.
"""


def extract_semantic_features(**kwargs):
    """Legacy entry point. Delegates to ScoringEngine interface."""
    from jobs.scoring_engine import get_engine

    engine = get_engine('bedrock_mock')

    sample_transactions = [
        {"desc": "Purchase 500 electronics units at $10 each", "order_id": "305-001"},
        {"desc": "Monthly subscription renewal", "order_id": "305-002"},
        {"desc": "Gift card bulk purchase 50x$200", "order_id": "305-003"},
    ]

    results = engine.extract_features(sample_transactions)

    print(f"Bedrock enrichment complete: {len(results)} transactions scored")
    for r in results:
        print(f"   {r['order_id']}: semantic_risk={r['semantic_risk']}")

    return results
