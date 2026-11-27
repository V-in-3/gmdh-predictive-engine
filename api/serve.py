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

from jobs.risk_entrypoint import run_risk_assessment

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring with RAG + LLM reasoning",
    version="1.0.0",
)


# === Request / Response models ===

class TransactionRequest(BaseModel):
    account_id: Optional[str] = Field(None, description="Account identifier")
    device_id: Optional[str] = Field(None, description="Device identifier")
    ip_address: Optional[str] = Field(None, description="IP address")
    trial_id: Optional[str] = Field(None, description="Trial identifier")
    semantic_risk: float = Field(..., ge=0, le=1, description="NLP-derived risk score")
    velocity_1h: float = Field(..., ge=0, description="Transactions per hour")
    proxy_score: float = Field(..., ge=0, le=1, description="VPN/proxy detection score")
    amount_deviation: float = Field(..., ge=0, description="Deviation from user mean spend")
    linked_accounts: int = Field(0, ge=0, description="Other accounts linked by graph signals")
    shared_devices: int = Field(0, ge=0, description="Shared devices in the account graph")
    shared_ips: int = Field(0, ge=0, description="Shared IP addresses in the account graph")
    trial_count_24h: int = Field(0, ge=0, description="Trials started by the linked group in 24 hours")
    transaction_id: Optional[str] = Field(None, description="Optional transaction ID")


class ScoringResult(BaseModel):
    transaction_id: str
    decision: str
    confidence: float
    nn_score: float
    gmdh_score: float
    graph_risk: float
    graph_reasons: list
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
    transaction = {
            "account_id": tx.account_id,
            "device_id": tx.device_id,
            "ip_address": tx.ip_address,
            "trial_id": tx.trial_id,
            "semantic_risk": tx.semantic_risk,
            "velocity_1h": tx.velocity_1h,
            "proxy_score": tx.proxy_score,
            "amount_deviation": tx.amount_deviation,
            "linked_accounts": tx.linked_accounts,
            "shared_devices": tx.shared_devices,
            "shared_ips": tx.shared_ips,
            "trial_count_24h": tx.trial_count_24h,
    }
    transaction_id = tx.transaction_id or f"api-{int(time.time())}"
    result = run_risk_assessment(transaction, transaction_id)

    return ScoringResult(
        transaction_id=result["transaction_id"],
        decision=result["final_decision"],
        confidence=result["confidence"],
        nn_score=result["nn_score"],
        gmdh_score=result["gmdh_score"],
        graph_risk=result.get("graph_risk", 0.0),
        graph_reasons=result.get("graph_reasons", []),
        route=result.get("route", "unknown"),
        similarity_verdict=result.get("similarity_verdict", "N/A"),
        similar_cases=[
            {
                "id": c.get("metadata", {}).get("description", "")[:50],
                "is_fraud": c.get("metadata", {}).get("is_fraud"),
                "distance": round(c.get("distance", 0), 4),
            }
            for c in result.get("similar_cases", [])[:3]
        ],
        reasoning=result["reasoning"][:200],
        latency_ms=result["latency_ms"],
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
