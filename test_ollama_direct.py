#!/usr/bin/env python3
"""Direct test of Ollama LLM reasoning — forces LLM call."""
import sys
sys.path.insert(0, '/opt/airflow/project')

from agents.fraud_investigation_agent import llm_reason

# Fake state that forces LLM call (no similar cases = goes to fallback,
# but we want to test Ollama directly)
state = {
    "transaction": {"semantic_risk": 0.55, "velocity_1h": 8, "proxy_score": 0.4, "amount_deviation": 1.0},
    "nn_score": 0.45,
    "gmdh_score": 0.45,
    "similar_cases": [
        {"distance": 0.003, "metadata": {"is_fraud": True, "description": "Bulk purchase smartphones, VPN, new account", "reason": "Stolen card resale ring"}},
        {"distance": 0.006, "metadata": {"is_fraud": False, "description": "Corporate bulk license, verified B2B", "reason": "Legitimate enterprise deal"}},
    ],
    "similarity_verdict": "UNCERTAIN",
    "similarity_confidence": 0.5,
    "llm_explanation": "",
    "final_decision": "",
    "reasoning": "",
    "confidence": 0,
    "transaction_id": "test-ollama",
    "decision_threshold": 0.75,
}

print("=" * 70)
print("TESTING OLLAMA LLM REASONING")
print("Calling gmdh-ollama:11434 (timeout=120s, please wait)...")
print("=" * 70)

result = llm_reason(state)

print("\nLLM Response:")
print("-" * 70)
print(result["llm_explanation"])
print("-" * 70)

if "Matches fraud pattern" in result["llm_explanation"] or "Mixed signals" in result["llm_explanation"]:
    print("\n⚠ This is FALLBACK (rule-based) — Ollama did not respond in time")
else:
    print("\n✓ This is OLLAMA response (LLM-generated)")
