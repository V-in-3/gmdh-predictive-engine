"""
Vector Store — pgvector-based embedding storage and similarity search.

Stores 32-dimensional embeddings from the NN's penultimate layer (Dense(32))
and provides nearest-neighbor search for case-based fraud explanation.

Usage:
    from jobs.vector_store import VectorStore

    store = VectorStore()
    store.store_embedding(transaction_id="305-001", embedding=[...], fraud_score=0.82, is_fraud=True)
    similar = store.find_similar(embedding=[...], top_k=5)
"""
import json
import os
from typing import Optional

import numpy as np

# PostgreSQL connection defaults (from docker-compose env)
PG_HOST = os.environ.get("POSTGRES_HOST", "gmdh-postgres")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_DB = os.environ.get("POSTGRES_DB", "airflow_db")
PG_USER = os.environ.get("POSTGRES_USER", "airflow_user")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "airflow_pass")

EMBEDDING_DIM = 32


class VectorStore:
    """pgvector-based store for transaction embeddings."""

    def __init__(self, host=None, port=None, dbname=None, user=None, password=None):
        self._host = host or PG_HOST
        self._port = port or PG_PORT
        self._dbname = dbname or PG_DB
        self._user = user or PG_USER
        self._password = password or PG_PASSWORD
        self._conn = None

    def _get_conn(self):
        """Lazy connection with auto-initialization of pgvector extension and table."""
        if self._conn is None or self._conn.closed:
            import psycopg2
            self._conn = psycopg2.connect(
                host=self._host,
                port=self._port,
                dbname=self._dbname,
                user=self._user,
                password=self._password
            )
            self._conn.autocommit = True
            # Ensure pgvector extension and table exist
            cur = self._conn.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""CREATE TABLE IF NOT EXISTS transaction_embeddings (
                id SERIAL PRIMARY KEY,
                transaction_id TEXT UNIQUE,
                embedding vector(32),
                fraud_score FLOAT,
                is_fraud BOOLEAN,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            cur.close()
            # Register pgvector type
            from pgvector.psycopg2 import register_vector
            register_vector(self._conn)
        return self._conn

    def store_embedding(
        self,
        transaction_id: str,
        embedding: list | np.ndarray,
        fraud_score: float,
        is_fraud: bool,
        metadata: Optional[dict] = None
    ):
        """
        Store a transaction embedding in pgvector.

        Args:
            transaction_id: Unique transaction identifier
            embedding: 32-dim vector from NN penultimate layer
            fraud_score: Model's fraud probability (0-1)
            is_fraud: Ground truth or model decision
            metadata: Optional JSON metadata (features, source, etc.)
        """
        conn = self._get_conn()
        cur = conn.cursor()

        emb = np.array(embedding, dtype=np.float32)
        if len(emb) != EMBEDDING_DIM:
            raise ValueError(f"Expected {EMBEDDING_DIM}-dim embedding, got {len(emb)}")

        cur.execute(
            """
            INSERT INTO transaction_embeddings (transaction_id, embedding, fraud_score, is_fraud, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (transaction_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                fraud_score = EXCLUDED.fraud_score,
                is_fraud = EXCLUDED.is_fraud,
                metadata = EXCLUDED.metadata
            """,
            (transaction_id, emb, fraud_score, is_fraud, json.dumps(metadata or {}))
        )

    def store_batch(self, records: list[dict]):
        """
        Store multiple embeddings in one transaction.

        Args:
            records: List of dicts with keys: transaction_id, embedding, fraud_score, is_fraud, metadata
        """
        conn = self._get_conn()
        cur = conn.cursor()

        from psycopg2.extras import execute_values

        values = []
        for r in records:
            emb = np.array(r["embedding"], dtype=np.float32)
            values.append((
                r["transaction_id"],
                emb,
                r["fraud_score"],
                r["is_fraud"],
                json.dumps(r.get("metadata", {}))
            ))

        execute_values(
            cur,
            """
            INSERT INTO transaction_embeddings (transaction_id, embedding, fraud_score, is_fraud, metadata)
            VALUES %s
            ON CONFLICT (transaction_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                fraud_score = EXCLUDED.fraud_score,
                is_fraud = EXCLUDED.is_fraud,
                metadata = EXCLUDED.metadata
            """,
            values,
            template="(%s, %s, %s, %s, %s)"
        )

    def find_similar(
        self,
        embedding: list | np.ndarray,
        top_k: int = 5,
        fraud_only: bool = False,
        max_distance: float = None
    ) -> list[dict]:
        """
        Find most similar transactions by cosine distance.

        Args:
            embedding: Query vector (32-dim)
            top_k: Number of nearest neighbors to return
            fraud_only: If True, only return fraud cases
            max_distance: Maximum cosine distance threshold (0=identical, 2=opposite)

        Returns:
            List of dicts with: transaction_id, distance, fraud_score, is_fraud, metadata
        """
        conn = self._get_conn()
        cur = conn.cursor()

        emb = np.array(embedding, dtype=np.float32)

        if fraud_only:
            query = """
                SELECT transaction_id,
                       embedding <=> %s AS distance,
                       fraud_score,
                       is_fraud,
                       metadata
                FROM transaction_embeddings
                WHERE is_fraud = TRUE
                ORDER BY embedding <=> %s
                LIMIT %s
            """
            cur.execute(query, (emb, emb, top_k))
        else:
            query = """
                SELECT transaction_id,
                       embedding <=> %s AS distance,
                       fraud_score,
                       is_fraud,
                       metadata
                FROM transaction_embeddings
                ORDER BY embedding <=> %s
                LIMIT %s
            """
            cur.execute(query, (emb, emb, top_k))

        rows = cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "transaction_id": row[0],
                "distance": float(row[1]),
                "fraud_score": float(row[2]),
                "is_fraud": bool(row[3]),
                "metadata": row[4] if row[4] else {},
            })

        return results

    def explain_decision(self, embedding: list | np.ndarray, top_k: int = 5) -> dict:
        """
        Case-based reasoning: explain why a transaction is suspicious.

        Returns explanation with:
        - similar_fraud: nearest fraud cases
        - similar_legit: nearest legitimate cases
        - confidence: ratio of fraud neighbors vs total
        - verdict: CONFIRMED_FRAUD / LIKELY_FRAUD / UNCERTAIN / LIKELY_LEGIT
        """
        # Find nearest fraud cases
        fraud_neighbors = self.find_similar(embedding, top_k=top_k, fraud_only=True)
        # Find nearest overall cases
        all_neighbors = self.find_similar(embedding, top_k=top_k, fraud_only=False)

        fraud_count = sum(1 for n in all_neighbors if n["is_fraud"])
        total = len(all_neighbors)

        if total == 0:
            confidence = 0.0
            verdict = "NO_HISTORY"
        else:
            confidence = fraud_count / total
            if confidence >= 0.8:
                verdict = "CONFIRMED_FRAUD"
            elif confidence >= 0.5:
                verdict = "LIKELY_FRAUD"
            elif confidence >= 0.3:
                verdict = "UNCERTAIN"
            else:
                verdict = "LIKELY_LEGIT"

        return {
            "verdict": verdict,
            "confidence": confidence,
            "fraud_neighbors": len(fraud_neighbors),
            "total_neighbors": total,
            "nearest_fraud_distance": fraud_neighbors[0]["distance"] if fraud_neighbors else None,
            "nearest_legit_distance": next(
                (n["distance"] for n in all_neighbors if not n["is_fraud"]), None
            ),
            "similar_cases": all_neighbors[:3],  # top 3 for display
        }

    def count(self) -> int:
        """Return total number of stored embeddings."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM transaction_embeddings")
        return cur.fetchone()[0]

    def close(self):
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
