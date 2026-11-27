import psycopg2

conn = psycopg2.connect(
    host="gmdh-postgres",
    user="airflow_user",
    password="airflow_pass",
    dbname="airflow_db",
)
cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("""
CREATE TABLE IF NOT EXISTS raw_subscriptions (
    id SERIAL PRIMARY KEY,
    kafka_offset BIGINT UNIQUE,
    payload JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS subscriptions_dlq (
    id SERIAL PRIMARY KEY,
    kafka_offset BIGINT UNIQUE,
    payload JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS transaction_embeddings (
    id SERIAL PRIMARY KEY,
    transaction_id TEXT UNIQUE,
    embedding vector(32),
    fraud_score FLOAT,
    is_fraud BOOLEAN,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
cur.execute("""
CREATE INDEX IF NOT EXISTS idx_embeddings_ivfflat
    ON transaction_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)
""")
conn.commit()
conn.close()
print("PostgreSQL tables + pgvector ready")
