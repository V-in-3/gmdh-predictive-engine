"""
Fraud Scoring API — FastAPI endpoint for real-time transaction analysis.

Exposes the LangGraph agent as an HTTP service:
  POST /score    — score a single transaction
  POST /batch    — score multiple transactions
  GET  /health   — service health check
  GET  /cases    — list seed cases in pgvector

Usage:
    uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload

    # Or via Docker:
    docker exec gmdh-airflow uvicorn api.serve:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.fraud_investigation_agent import (
    score_transaction,
    search_similar,
    route_after_scoring,
    route_after_similarity,
    llm_reason,
    make_decision,
)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring with RAG + LLM reasoning",
    version="1.0.0",
)


# === Request / Response models ===

class TransactionRequest(BaseModel):
    semantic_risk: float = Field(..., ge=0, le=1, description="NLP-derived risk score")
    velocity_1h: float = Field(..., ge=0, description="Transactions per hour")
    proxy_score: float = Field(..., ge=0, le=1, description="VPN/proxy detection score")
    amount_deviation: float = Field(..., ge=0, description="Deviation from user mean spend")
    transaction_id: Optional[str] = Field(None, description="Optional transaction ID")


class ScoringResult(BaseModel):
    transaction_id: str
    decision: str
    confidence: float
    nn_score: float
    gmdh_score: float
    route: str
    similarity_verdict: str
    similar_cases: list
    reasoning: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    pgvector: str
    ollama: str


# === Endpoints ===

@app.post("/score", response_model=ScoringResult)
def score_transaction_endpoint(tx: TransactionRequest):
    """Score a single transaction through the full agent pipeline."""
    start = time.time()

    state = {
        "transaction": {
            "semantic_risk": tx.semantic_risk,
            "velocity_1h": tx.velocity_1h,
            "proxy_score": tx.proxy_score,
            "amount_deviation": tx.amount_deviation,
        },
        "transaction_id": tx.transaction_id or f"api-{int(time.time())}",
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

    # Step 1 — Score
    scored = score_transaction(state)
    state.update(scored)

    # Step 2 — Route
    route = route_after_scoring(state)

    if route == "investigate":
        # Step 3 — pgvector
        similar = search_similar(state)
        state.update(similar)

        # Step 4 — Route after similarity
        route2 = route_after_similarity(state)

        if route2 == "llm_reason":
            # Step 5 — LLM
            llm = llm_reason(state)
            state.update(llm)

    # Final decision
    decision = make_decision(state)
    state.update(decision)

    latency = (time.time() - start) * 1000

    return ScoringResult(
        transaction_id=state["transaction_id"],
        decision=state["final_decision"],
        confidence=state["confidence"],
        nn_score=state["nn_score"],
        gmdh_score=state["gmdh_score"],
        route=route,
        similarity_verdict=state.get("similarity_verdict", "N/A"),
        similar_cases=[
            {
                "id": c.get("metadata", {}).get("description", "")[:50],
                "is_fraud": c.get("metadata", {}).get("is_fraud"),
                "distance": round(c.get("distance", 0), 4),
            }
            for c in state.get("similar_cases", [])[:3]
        ],
        reasoning=state["reasoning"][:200],
        latency_ms=round(latency, 1),
    )


@app.post("/batch", response_model=list[ScoringResult])
def batch_score(transactions: list[TransactionRequest]):
    """Score multiple transactions."""
    if len(transactions) > 50:
        raise HTTPException(status_code=400, detail="Max 50 transactions per batch")
    return [score_transaction_endpoint(tx) for tx in transactions]


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check service dependencies."""
    pgvector_status = "unknown"
    ollama_status = "unknown"

    # Check pgvector
    try:
        from jobs.vector_store import VectorStore
        store = VectorStore()
        store._get_conn()
        store.close()
        pgvector_status = "ok"
    except Exception:
        pgvector_status = "unavailable"

    # Check Ollama
    try:
        import urllib.request
        ollama_url = os.environ.get("OLLAMA_URL", "http://gmdh-ollama:11434")
        req = urllib.request.Request(f"{ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                ollama_status = "ok"
    except Exception:
        ollama_status = "unavailable"

    status = "healthy" if pgvector_status == "ok" else "degraded"
    return HealthResponse(status=status, pgvector=pgvector_status, ollama=ollama_status)


@app.get("/cases")
def list_cases():
    """List seed cases currently in pgvector."""
    try:
        from jobs.vector_store import VectorStore
        store = VectorStore()
        conn = store._get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT transaction_id, fraud_score, is_fraud, metadata
            FROM transaction_embeddings
            ORDER BY created_at DESC
            LIMIT 50
        """)
        rows = cur.fetchall()
        store.close()
        return [
            {
                "id": r[0],
                "fraud_score": r[1],
                "is_fraud": r[2],
                "description": r[3].get("description", "") if r[3] else "",
            }
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"pgvector unavailable: {str(e)}")
