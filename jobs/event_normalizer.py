"""Normalize marketplace and payment events for risk analysis."""

from typing import Any


def normalize_risk_event(event: dict[str, Any]) -> dict[str, Any]:
    """Convert an Amazon/Cybersource envelope into one risk-event shape."""
    payload = event.get("payload", event)
    metadata = event.get("metadata", {})
    source = metadata.get("source_system", payload.get("source_system", "unknown"))
    risk_score = payload.get("fraud_score")
    if risk_score is not None:
        risk_score = max(0.0, min(float(risk_score) / 100.0, 1.0))

    return {
        "event_id": metadata.get("event_id"),
        "source_system": source,
        "order_id": payload.get("order_id"),
        "account_id": payload.get("account_id"),
        "device_id": payload.get("device_id"),
        "ip_address": payload.get("ip_address"),
        "trial_id": payload.get("trial_id"),
        "event_type": payload.get("event_type", "transaction"),
        "amount": payload.get("amount"),
        "currency": payload.get("currency"),
        "payment_decision": payload.get("decision"),
        "payment_risk": risk_score,
    }