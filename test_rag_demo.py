#!/usr/bin/env python3
"""RAG Demo — shows full agent pipeline with pgvector similarity search.

Uses real transactions from fraud_production_50k.csv benchmark dataset.
"""
import sys
import os

sys.path.insert(0, '/opt/airflow/project')

import pandas as pd
from agents.fraud_investigation_agent import (
    score_transaction, search_similar, route_after_scoring,
    route_after_similarity, llm_reason, make_decision
)

DATA_FILE = "/opt/airflow/project/data/fraud_production_50k.csv"

# Load real benchmark data — pick 3 fraud + 2 legit for variety
df = pd.read_csv(DATA_FILE)
fraud_sample = df[df["is_fraud"] == 1].sample(3, random_state=42)
legit_sample = df[df["is_fraud"] == 0].sample(2, random_state=42)
sample = pd.concat([fraud_sample, legit_sample]).reset_index(drop=True)

print("=" * 70)
print("FRAUD INVESTIGATION AGENT — RAG DEMO (real benchmark data)")
print(f"Source: {os.path.basename(DATA_FILE)} ({len(df):,} records)")
print(f"Sampling: 3 fraud + 2 legit transactions")
print("=" * 70)

for idx, row in sample.iterrows():
    tx = {
        "semantic_risk": round(row["semantic_risk"], 3),
        "velocity_1h": round(row["velocity_1h"], 1),
        "proxy_score": round(row["proxy_score"], 3),
        "amount_deviation": round(row["amount_deviation"], 3),
    }
    actual = "FRAUD" if row["is_fraud"] == 1 else "LEGIT"
    pattern = row.get("pattern", "unknown")

    print(f"\n{'─' * 70}")
    print(f"TRANSACTION #{idx+1} (actual: {actual}, pattern: {pattern})")
    print(f"  Features: {tx}")
    print(f"{'─' * 70}")

    state = {
        "transaction": tx,
        "transaction_id": f"bench-{idx:03d}",
        "nn_score": 0.0,
        "gmdh_score": 0.0,
        "decision_threshold": 0.75,
        "similar_cases": [],
        "similarity_verdict": "",
        "similarity_confidence": 0.0,
        "llm_explanation": "",
        "final_decision": "",
        "reasoning": "",
        "confidence": 0.0,
    }

    # Step 1
    scored = score_transaction(state)
    state.update(scored)
    print(f"\n  Step 1 — SCORE:")
    print(f"    NN score:   {state['nn_score']:.3f}")
    print(f"    GMDH score: {state['gmdh_score']:.3f}")

    # Step 2 — Route
    route = route_after_scoring(state)
    print(f"\n  Step 2 — ROUTE: {route}")

    if route == "investigate":
        # Step 3 — pgvector RAG retrieval
        similar = search_similar(state)
        state.update(similar)
        print(f"\n  Step 3 — SIMILARITY SEARCH (pgvector):")
        print(f"    Verdict: {state['similarity_verdict']}")
        print(f"    Cases found: {len(state['similar_cases'])}")
        for i, c in enumerate(state["similar_cases"][:5], 1):
            meta = c.get("metadata", {})
            label = "FRAUD" if meta.get("is_fraud") else "LEGIT"
            dist = c.get("distance", 0)
            desc = meta.get("description", "")[:65]
            print(f"      #{i} [{label}] dist={dist:.3f} — {desc}")

        # Step 4 — Route after similarity
        route2 = route_after_similarity(state)
        print(f"\n  Step 4 — ROUTE: {route2}")

        if route2 == "llm_reason":
            llm = llm_reason(state)
            state.update(llm)
            print(f"\n  Step 5 — LLM REASONING (RAG-augmented):")
            print(f"    {state['llm_explanation'][:200]}")
    else:
        print(f"    → Fast path ({route}), skipping investigation")

    # Final decision
    decision = make_decision(state)
    state.update(decision)

    # Compare with ground truth
    predicted = "FRAUD" if state["final_decision"] == "BLOCK" else "LEGIT"
    correct = "✓" if predicted == actual else "✗"

    print(f"\n  ╔══════════════════════════════════════════════╗")
    print(f"  ║  DECISION:   {state['final_decision']:<10} (actual: {actual}) {correct}   ║")
    print(f"  ║  Confidence: {state['confidence']:.0%}{'':>28}║")
    print(f"  ╚══════════════════════════════════════════════╝")
    print(f"  Reasoning: {state['reasoning'][:130]}")

print(f"\n{'═' * 70}")
print("Done.")

