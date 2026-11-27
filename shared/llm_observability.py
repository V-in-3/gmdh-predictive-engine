"""
LLM call observability - logs every LLM call from any agent node into Postgres
(the same `airflow_db` already wired up as a Grafana datasource, not because it
relates to Airflow, just because it's the one datasource already connected).

Usage:
    from shared.llm_observability import log_llm_call
    log_llm_call(agent_node="llm_reason", model="tinyllama", backend="ollama",
                 latency_ms=842, prompt_tokens=310, completion_tokens=64)
"""
import os
import time
from contextlib import contextmanager

import psycopg2

PG_HOST = os.environ.get("PG_HOST", "gmdh-postgres")
PG_DB = os.environ.get("PG_DB", "airflow_db")
PG_USER = os.environ.get("PG_USER", "airflow_user")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "airflow_pass")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_call_log (
    id SERIAL PRIMARY KEY,
    called_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    agent_node TEXT NOT NULL,
    transaction_id TEXT,
    model_used TEXT NOT NULL,
    backend TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_ms INTEGER NOT NULL,
    estimated_cost_usd NUMERIC(10, 6) DEFAULT 0,
    preliminary_decision TEXT,
    success BOOLEAN NOT NULL DEFAULT true,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_call_log_called_at ON llm_call_log (called_at);
CREATE INDEX IF NOT EXISTS idx_llm_call_log_node ON llm_call_log (agent_node);
"""

_schema_ensured = False


@contextmanager
def _connect():
    conn = psycopg2.connect(host=PG_HOST, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD)
    try:
        yield conn
    finally:
        conn.close()


def ensure_schema() -> None:
    global _schema_ensured
    if _schema_ensured:
        return
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(_SCHEMA)
        conn.commit()
    _schema_ensured = True


def log_llm_call(
    agent_node: str,
    model: str,
    backend: str,
    latency_ms: int,
    transaction_id: str = None,
    prompt_tokens: int = None,
    completion_tokens: int = None,
    estimated_cost_usd: float = 0.0,
    preliminary_decision: str = None,
    success: bool = True,
    error: str = None,
) -> None:
    """Best-effort logging - a logging failure must never break the agent's own
    decision path, so any exception here is swallowed, not raised."""
    try:
        ensure_schema()
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO llm_call_log
                   (agent_node, transaction_id, model_used, backend, latency_ms,
                    prompt_tokens, completion_tokens, estimated_cost_usd,
                    preliminary_decision, success, error)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    agent_node, transaction_id, model, backend, latency_ms,
                    prompt_tokens, completion_tokens, estimated_cost_usd,
                    preliminary_decision, success, error,
                ),
            )
            conn.commit()
    except Exception:
        pass


@contextmanager
def timed_llm_call(**log_kwargs):
    """Context manager that times the block and logs it on exit, success or not.

    Usage:
        with timed_llm_call(agent_node="llm_reason", model="tinyllama", backend="ollama",
                             preliminary_decision=prelim) as call:
            result = ...  # do the actual LLM request
            call["prompt_tokens"] = result.get("prompt_eval_count")
            call["completion_tokens"] = result.get("eval_count")
    """
    start = time.monotonic()
    call_info = {}
    try:
        yield call_info
        log_llm_call(latency_ms=int((time.monotonic() - start) * 1000), **log_kwargs, **call_info)
    except Exception as exc:
        log_llm_call(
            latency_ms=int((time.monotonic() - start) * 1000),
            success=False, error=str(exc),
            **{k: v for k, v in log_kwargs.items() if k not in ("success", "error")},
        )
        raise
