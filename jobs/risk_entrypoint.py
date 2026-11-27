"""Single internal entry point for fraud-risk assessment."""

import time


def run_risk_assessment(transaction: dict, transaction_id: str) -> dict:
    """Run the shared fraud agent and attach request latency."""
    from agents.fraud_investigation_agent import investigate_transaction, route_after_scoring

    started_at = time.time()
    result = investigate_transaction(transaction, transaction_id)
    result["route"] = route_after_scoring(result)
    result["latency_ms"] = round((time.time() - started_at) * 1000, 1)
    return result