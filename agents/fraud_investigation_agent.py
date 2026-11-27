"""
Fraud Investigation Agent — LangGraph implementation.

A multi-step agent that:
1. Scores transaction via NN/GMDH
2. If uncertain → searches pgvector for similar cases
3. If still uncertain → asks LLM to reason about the transaction
4. Makes final BLOCK/ALLOW decision with explanation

Requires: pip install langgraph langchain-core langchain-community

Usage:
    from agents.fraud_investigation_agent import investigate_transaction
    result = investigate_transaction({"semantic_risk": 0.6, "velocity_1h": 25, ...})
"""
import os
import json
import time
from typing import TypedDict, Literal

from shared.llm_observability import log_llm_call

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

import numpy as np


# --- State definition ---

class FraudState(TypedDict):
    """State that flows through the agent graph."""
    # Input
    transaction: dict
    transaction_id: str

    # Scoring
    nn_score: float
    gmdh_score: float
    decision_threshold: float

    # Graph evidence
    graph_risk: float
    graph_reasons: list

    # Similarity search
    similar_cases: list
    similarity_verdict: str
    similarity_confidence: float

    # LLM reasoning
    llm_explanation: str

    # Final
    final_decision: str  # BLOCK / ALLOW / REVIEW
    reasoning: str
    confidence: float


# --- Node functions ---

def score_transaction(state: FraudState) -> dict:
    """Node 1: Score transaction with NN and GMDH."""
    tx = state["transaction"]

    # GMDH scoring (polynomial — always available)
    gmdh_score = (
        0.4 * tx.get("semantic_risk", 0)
        + 0.3 * tx.get("proxy_score", 0)
        + 0.15 * (tx.get("semantic_risk", 0) * tx.get("amount_deviation", 0))
        + 0.05 * (tx.get("velocity_1h", 0) / 50.0)
    )
    gmdh_score = max(0.0, min(1.0, gmdh_score))

    # NN scoring (TensorFlow — if available)
    nn_score = gmdh_score  # fallback
    try:
        import tensorflow as tf
        model_path = os.environ.get(
            "NN_MODEL_PATH",
            "data/fraud_model_nn_tf.keras"
        )
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            features = np.zeros((1, model.input_shape[1]), dtype=np.float32)
            features[0, 0] = tx.get("semantic_risk", 0)
            features[0, 1] = tx.get("velocity_1h", 0) / 50.0
            features[0, 2] = tx.get("proxy_score", 0)
            features[0, 3] = tx.get("amount_deviation", 0) / 3.0
            nn_score = float(model.predict(features, verbose=0)[0, 0])
    except Exception:
        pass

    return {
        "nn_score": nn_score,
        "gmdh_score": gmdh_score,
        "decision_threshold": 0.75,
    }


def analyze_graph(state: FraudState) -> dict:
    """Node 2: calculate account and subscription relationship risk."""
    from jobs.graph_analysis import analyze_account_graph

    result = analyze_account_graph(state["transaction"])
    return {
        "graph_risk": result["graph_risk"],
        "graph_reasons": result["graph_reasons"],
    }


def search_similar(state: FraudState) -> dict:
    """Node 2: Search pgvector for similar historical cases (RAG retrieval)."""
    tx = state["transaction"]

    try:
        from jobs.vector_store import VectorStore
        store = VectorStore()

        # Create embedding from features
        base = np.array([
            tx.get("semantic_risk", 0),
            tx.get("velocity_1h", 0) / 50.0,
            tx.get("proxy_score", 0),
            tx.get("amount_deviation", 0) / 3.0,
        ], dtype=np.float32)
        embedding = np.concatenate([
            base,
            base * 0.9,
            np.full(24, base.mean(), dtype=np.float32)
        ])

        # Get full explanation with cases
        explanation = store.explain_decision(embedding, top_k=5)
        similar_cases = explanation.get("similar_cases", [])
        store.close()

        # Determine verdict from neighbors
        if similar_cases:
            fraud_count = sum(1 for c in similar_cases if c.get("metadata", {}).get("is_fraud"))
            total = len(similar_cases)
            ratio = fraud_count / total if total > 0 else 0

            if ratio >= 0.8:
                verdict = "CONFIRMED_FRAUD"
            elif ratio >= 0.5:
                verdict = "LIKELY_FRAUD"
            elif ratio <= 0.2:
                verdict = "LIKELY_LEGIT"
            else:
                verdict = "UNCERTAIN"
            confidence = max(ratio, 1 - ratio)
        else:
            verdict = "NO_HISTORY"
            confidence = 0.0

        return {
            "similar_cases": similar_cases,
            "similarity_verdict": verdict,
            "similarity_confidence": confidence,
        }
    except Exception:
        return {
            "similar_cases": [],
            "similarity_verdict": "UNAVAILABLE",
            "similarity_confidence": 0.0,
        }


def _preliminary_decision(nn_score: float, similarity_verdict: str, graph_risk: float) -> str:
    """Compute the likely decision so the LLM explains it (not invents its own)."""
    if graph_risk >= 0.70 and nn_score <= 0.75:
        return "REVIEW"
    if nn_score > 0.75:
        return "BLOCK"
    if nn_score > 0.4 and similarity_verdict in ("CONFIRMED_FRAUD", "LIKELY_FRAUD"):
        return "BLOCK"
    if nn_score < 0.3:
        return "ALLOW"
    return "REVIEW"


def llm_reason(state: FraudState) -> dict:
    """Node 3: Ask LLM to EXPLAIN the decision (not invent one)."""
    tx = state["transaction"]
    nn_score = state["nn_score"]
    similar = state["similar_cases"]
    graph_risk = state.get("graph_risk", 0.0)
    graph_reasons = state.get("graph_reasons", [])
    similarity_verdict = state.get("similarity_verdict", "")

    prelim = _preliminary_decision(nn_score, similarity_verdict, graph_risk)

    # Build RAG context from retrieved cases
    rag_context = ""
    if similar:
        rag_context = "\nSIMILAR HISTORICAL CASES (from pgvector cosine search):\n"
        for i, case in enumerate(similar[:5], 1):
            meta = case.get("metadata", {})
            distance = case.get("distance", 0)
            label = "FRAUD" if meta.get("is_fraud") else "LEGIT"
            desc = meta.get("description", "No description")
            reason = meta.get("reason", "")
            rag_context += f"\n  Case #{i} (distance: {distance:.3f}, {label}):\n"
            rag_context += f"    \"{desc}\"\n"
            if reason:
                rag_context += f"    Verdict reason: {reason}\n"
    else:
        rag_context = "\nNo similar historical cases found in pgvector.\n"

    prompt = f"""You are a fraud analyst. Analyze this transaction and provide a brief reasoning.

CURRENT TRANSACTION:
- Semantic risk (from NLP): {tx.get('semantic_risk', 'N/A')}
- Velocity (transactions/hour): {tx.get('velocity_1h', 'N/A')}
- Proxy score: {tx.get('proxy_score', 'N/A')}
- Amount deviation from user mean: {tx.get('amount_deviation', 'N/A')}

MODEL SCORE:
- Neural Network fraud probability: {nn_score:.3f} (threshold: 0.75, so this is {'HIGH' if nn_score > 0.75 else 'LOW' if nn_score < 0.3 else 'MODERATE'})

The system has already decided: {prelim}. Your job is to EXPLAIN this decision, not change it.

SIGNAL A — BEHAVIORAL SIMILARITY (transaction vs known cases):
{rag_context}

SIGNAL B — NETWORK / FRAUD-RING (account connections):
- Graph risk: {graph_risk:.3f} {'(HIGH — fraud ring)' if graph_risk >= 0.7 else '(LOW — no ring)' if graph_risk < 0.3 else '(MODERATE)'}
- {', '.join(graph_reasons) if graph_reasons else 'No linked-account risk detected'}

Write a 2-3 sentence analyst note that JUSTIFIES the {prelim} decision.
- If ALLOW: explain why the signals are benign (low NN score, no fraud ring, similar to legit cases).
- If BLOCK: explain which signals triggered it (high NN score and/or fraud ring and/or fraud-similar cases).
- If REVIEW: explain what is ambiguous.
Be consistent with the {prelim} decision and the actual numbers above. Do NOT contradict them.
"""

    # LLM_BACKEND env var picks ollama (local, default) or groq (cloud)
    from shared.llm_backends import build_backend, estimate_cost_usd

    call_started = time.monotonic()
    backend = None
    try:
        backend = build_backend()
        llm_response, usage = backend.complete(prompt)
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        log_llm_call(
            agent_node="llm_reason", model=backend.model, backend=backend.name,
            latency_ms=int((time.monotonic() - call_started) * 1000),
            transaction_id=state.get("transaction_id"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimate_cost_usd(backend.name, backend.model, prompt_tokens, completion_tokens),
            preliminary_decision=prelim,
        )
        return {"llm_explanation": llm_response}
    except Exception as exc:
        model_used = getattr(backend, "model", os.environ.get("LLM_BACKEND", "ollama"))
        backend_name = getattr(backend, "name", os.environ.get("LLM_BACKEND", "ollama"))
        error_text = str(exc)
        if type(exc).__name__ == "RateLimitError":
            error_text = f"rate_limited: {error_text}"
        log_llm_call(
            agent_node="llm_reason", model=model_used, backend=backend_name,
            latency_ms=int((time.monotonic() - call_started) * 1000),
            transaction_id=state.get("transaction_id"),
            preliminary_decision=prelim, success=False, error=error_text,
        )
        # Fallback: rule-based explanation with RAG context
        if similar:
            fraud_cases = [c for c in similar if c.get("metadata", {}).get("is_fraud")]
            legit_cases = [c for c in similar if not c.get("metadata", {}).get("is_fraud")]
            if fraud_cases and nn_score > 0.5:
                case_desc = fraud_cases[0].get("metadata", {}).get("description", "unknown")
                explanation = f"Matches fraud pattern: \"{case_desc[:60]}\" (distance: {fraud_cases[0].get('distance', 0):.3f}). NN score {nn_score:.2f} confirms."
            elif legit_cases and nn_score < 0.4:
                case_desc = legit_cases[0].get("metadata", {}).get("description", "unknown")
                explanation = f"Similar to legit case: \"{case_desc[:60]}\". Low NN score {nn_score:.2f} supports ALLOW."
            else:
                explanation = f"Mixed signals: NN={nn_score:.2f}, {len(fraud_cases)} fraud + {len(legit_cases)} legit neighbors."
        else:
            if nn_score > 0.7:
                explanation = f"High fraud probability ({nn_score:.2f}). Multiple risk signals active."
            elif nn_score > 0.4:
                explanation = f"Moderate risk ({nn_score:.2f}). Recommend manual review."
            else:
                explanation = f"Low risk ({nn_score:.2f}). No strong fraud indicators."

        return {"llm_explanation": explanation}


def make_decision(state: FraudState) -> dict:
    """Node 4: Final decision based on all evidence."""
    nn_score = state["nn_score"]
    similarity_verdict = state["similarity_verdict"]
    similarity_confidence = state["similarity_confidence"]
    graph_risk = state.get("graph_risk", 0.0)
    graph_reasons = state.get("graph_reasons", [])

    # Graph evidence escalates uncertain cases but cannot block by itself.
    if graph_risk >= 0.70 and nn_score <= 0.75:
        decision = "REVIEW"
        confidence = 0.65
        reasoning = f"Graph risk {graph_risk:.2f} requires review"
    # Decision logic: combine all signals
    elif nn_score > 0.75 and similarity_verdict == "CONFIRMED_FRAUD":
        decision = "BLOCK"
        confidence = 0.95
        reasoning = f"NN score {nn_score:.2f} + pgvector confirms fraud pattern"
    elif nn_score > 0.75:
        decision = "BLOCK"
        confidence = 0.80
        reasoning = f"NN score {nn_score:.2f} above threshold"
    elif nn_score > 0.4 and similarity_verdict in ("CONFIRMED_FRAUD", "LIKELY_FRAUD"):
        decision = "BLOCK"
        confidence = 0.70
        reasoning = f"Moderate NN score {nn_score:.2f} + similar to known fraud"
    elif nn_score < 0.3 and similarity_verdict == "LIKELY_LEGIT":
        decision = "ALLOW"
        confidence = 0.90
        reasoning = f"Low NN score {nn_score:.2f} + similar to legitimate transactions"
    elif nn_score < 0.3:
        decision = "ALLOW"
        confidence = 0.75
        reasoning = f"Low NN score {nn_score:.2f}"
    else:
        decision = "REVIEW"
        confidence = 0.50
        reasoning = f"Uncertain: NN={nn_score:.2f}, pgvector={similarity_verdict}"

    # Build a full signal summary so every layer is visible (demo-friendly).
    signals = [f"NN={nn_score:.2f}"]
    if similarity_verdict:
        signals.append(f"pgvector={similarity_verdict}")
    if graph_risk > 0:
        signals.append(f"graph_risk={graph_risk:.2f}")
    reasoning = f"[{decision}] " + reasoning + " | Signals: " + ", ".join(signals)

    # Add graph reasons to the short reasoning summary
    if graph_reasons:
        reasoning += f" | Graph: {'; '.join(graph_reasons[:2])}"

    # Full LLM explanation is returned separately (not truncated)
    return {
        "final_decision": decision,
        "confidence": confidence,
        "reasoning": reasoning,
        "llm_explanation": state.get("llm_explanation", ""),
    }


# --- Router function ---

def route_after_scoring(state: FraudState) -> Literal["allow", "block", "investigate"]:
    """Route based on initial score — clear cases skip investigation.

    DEMO_MODE=true forces every transaction through the full pipeline
    (graph + pgvector + LLM) so all layers are shown, even obvious cases.
    """
    if os.environ.get("DEMO_MODE", "false").lower() == "true":
        return "investigate"

    nn_score = state["nn_score"]
    if nn_score > 0.85:
        return "block"     # Obviously fraud — skip investigation
    elif nn_score < 0.15:
        return "allow"     # Obviously legit — skip investigation
    else:
        return "investigate"  # Uncertain — need more evidence


def route_after_similarity(state: FraudState) -> Literal["decide", "llm_reason"]:
    """Route: if pgvector gives clear answer, decide. Otherwise ask LLM.

    DEMO_MODE=true forces LLM reasoning on every investigated case so the
    full pipeline (NN + graph + pgvector + LLM) is always shown.
    """
    if os.environ.get("DEMO_MODE", "false").lower() == "true":
        return "llm_reason"

    verdict = state["similarity_verdict"]
    if verdict in ("CONFIRMED_FRAUD", "LIKELY_LEGIT"):
        return "decide"    # pgvector is confident enough
    else:
        return "llm_reason"  # Need LLM to break the tie


# --- Build graph ---

def build_fraud_agent() -> "StateGraph":
    """Build the LangGraph fraud investigation agent."""
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "LangGraph not installed. Run: pip install langgraph langchain-core"
        )

    graph = StateGraph(FraudState)

    # Add nodes
    graph.add_node("score", score_transaction)
    graph.add_node("analyze_graph", analyze_graph)
    graph.add_node("search_similar", search_similar)
    graph.add_node("llm_reason", llm_reason)
    graph.add_node("decide", make_decision)

    # Entry point
    graph.set_entry_point("score")
    graph.add_edge("score", "analyze_graph")

    # Conditional routing after scoring
    graph.add_conditional_edges(
        "analyze_graph",
        route_after_scoring,
        {
            "block": "decide",        # High score → decide immediately
            "allow": "decide",        # Low score → decide immediately
            "investigate": "search_similar",  # Uncertain → investigate
        }
    )

    # After similarity search
    graph.add_conditional_edges(
        "search_similar",
        route_after_similarity,
        {
            "decide": "decide",       # pgvector confident → decide
            "llm_reason": "llm_reason",  # Still uncertain → ask LLM
        }
    )

    # After LLM reasoning → always decide
    graph.add_edge("llm_reason", "decide")

    # After decision → end
    graph.add_edge("decide", END)

    return graph.compile()


# --- Public API ---

def investigate_transaction(transaction: dict, transaction_id: str = "unknown") -> dict:
    """
    Run the fraud investigation agent on a single transaction.

    Args:
        transaction: dict with keys: semantic_risk, velocity_1h, proxy_score, amount_deviation
        transaction_id: unique identifier

    Returns:
        dict with: final_decision, confidence, reasoning, nn_score, gmdh_score, ...
    """
    agent = build_fraud_agent()

    initial_state = {
        "transaction": transaction,
        "transaction_id": transaction_id,
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
        "graph_risk": 0.0,
        "graph_reasons": [],
    }

    result = agent.invoke(initial_state)
    return result


# --- CLI demo ---

if __name__ == "__main__":
    if not LANGGRAPH_AVAILABLE:
        print("LangGraph not installed. Run: pip install langgraph langchain-core")
        print("Demonstrating logic without LangGraph:")
        print()

        # Manual demo without LangGraph
        tx = {"semantic_risk": 0.85, "velocity_1h": 12, "proxy_score": 1.0, "amount_deviation": 2.1}
        state = {"transaction": tx, "nn_score": 0, "gmdh_score": 0, "decision_threshold": 0.75,
                 "similar_cases": [], "similarity_verdict": "", "similarity_confidence": 0,
                 "llm_explanation": "", "final_decision": "", "reasoning": "", "confidence": 0,
                 "transaction_id": "demo-001"}

        print(f"Transaction: {tx}")
        print()

        # Step 1
        scored = score_transaction(state)
        state.update(scored)
        print(f"Step 1 - Score: NN={state['nn_score']:.3f}, GMDH={state['gmdh_score']:.3f}")

        # Step 2 - Route
        route = route_after_scoring(state)
        print(f"Step 2 - Route: {route}")

        if route == "investigate":
            # Step 3
            similar = search_similar(state)
            state.update(similar)
            print(f"Step 3 - Similarity: verdict={state['similarity_verdict']}, confidence={state['similarity_confidence']:.0%}")

            # Step 4 - Route
            route2 = route_after_similarity(state)
            print(f"Step 4 - Route: {route2}")

            if route2 == "llm_reason":
                llm = llm_reason(state)
                state.update(llm)
                print(f"Step 5 - LLM: {state['llm_explanation'][:80]}...")

        # Final decision
        decision = make_decision(state)
        state.update(decision)
        print()
        print(f"DECISION: {state['final_decision']}")
        print(f"Confidence: {state['confidence']:.0%}")
        print(f"Reasoning: {state['reasoning']}")

    else:
        print("=" * 70)
        print("FRAUD INVESTIGATION AGENT (LangGraph)")
        print("=" * 70)
        print()

        transactions = [
            {"semantic_risk": 0.85, "velocity_1h": 12, "proxy_score": 1.0, "amount_deviation": 2.1},
            {"semantic_risk": 0.2, "velocity_1h": 3, "proxy_score": 0.0, "amount_deviation": 0.3},
            {"semantic_risk": 0.6, "velocity_1h": 25, "proxy_score": 0.5, "amount_deviation": 1.5},
        ]

        for i, tx in enumerate(transactions, 1):
            print(f"--- Transaction #{i} ---")
            print(f"  Input: {tx}")
            result = investigate_transaction(tx, f"demo-{i:03d}")
            print(f"  Decision: {result['final_decision']}")
            print(f"  Confidence: {result['confidence']:.0%}")
            print(f"  NN Score: {result['nn_score']:.3f}")
            print(f"  GMDH Score: {result['gmdh_score']:.3f}")
            print(f"  Similarity: {result['similarity_verdict']}")
            print(f"  Reasoning: {result['reasoning'][:100]}")
            print()

        print("=" * 70)
