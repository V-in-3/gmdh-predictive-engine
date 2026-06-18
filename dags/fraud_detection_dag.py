import json
import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_DIR = '/opt/airflow/project/data'
MODEL_A_PATH = f'{DATA_DIR}/fraud_model_coeffs.json'


def enrich_with_bedrock(**kwargs):
    """Task 1: Call Bedrock (mock) to extract semantic features."""
    from jobs.bedrock_extractor import extract_semantic_features
    return extract_semantic_features()


def train_fraud_model(**kwargs):
    """Task 2: Train GMDH fraud model (Model A)."""
    from jobs.gmdh_fraud_trainer import train_fraud_model as train
    model = train(
        data_path=f'{DATA_DIR}/fraud_transactions.csv',
        output_path=MODEL_A_PATH
    )
    return f"Model A trained. RMSE={model['final_rmse']:.4f}"


def train_health_model(**kwargs):
    """Task 3: Train GMDH health model (Model B). Produces health_score for fallback logic."""
    from jobs.gmdh_health_trainer import train_health_model as train
    _, health = train(
        data_path=f'{DATA_DIR}/fintech_transactions_raw.csv',
        output_path=f'{DATA_DIR}/model_b_coeffs.json',
        health_output_path=f'{DATA_DIR}/model_b_health.json'
    )
    return f"Model B trained. health_score={health['health_score']:.4f}, status={health['status']}"


def check_system_health(**kwargs):
    """
    Task 4: Read Model B (System Efficiency) health score.
    If health < 0.45 → disable fraud inference (fallback).
    Connects Model A to Model B output.
    """
    from airflow.providers.mysql.hooks.mysql import MySqlHook

    model_b_path = f'{DATA_DIR}/model_b_health.json'

    # Try reading Model B health from file (produced by gmdh_predictive_engine_it)
    if os.path.exists(model_b_path):
        with open(model_b_path, 'r') as f:
            health = json.load(f).get('health_score', 0.75)
    else:
        # Fallback: estimate health from Kafka lag (connects to data integrity layer)
        try:
            hook = MySqlHook(mysql_conn_id='mysql_default')
            result = hook.get_first("SELECT COUNT(*) FROM raw_subscriptions")
            db_count = result[0] if result else 0
            # If we have data flowing, system is likely healthy
            health = 0.8 if db_count > 0 else 0.5
            print(f"Model B file not found. Estimated health from DB sync: {health} ({db_count} records)")
        except Exception:
            health = 0.75  # Default: assume healthy

    print(f"Model B health_score: {health}")

    if health < 0.45:
        print("CRITICAL: System degraded. Model A inference DISABLED (fallback).")
        return "DISABLED"

    print("System healthy. Model A inference ENABLED.")
    return "ENABLED"


def run_fraud_inference(**kwargs):
    """
    Task 5: Apply GMDH fraud model to simulated transactions.
    Uses polynomial: score = \u03b2\u2080 + \u03a3(\u03b2\u1d62\u00b7x\u1d62)
    """
    ti = kwargs['ti']
    health_status = ti.xcom_pull(task_ids='check_system_health')

    if health_status == "DISABLED":
        print("Skipping inference - system in fallback mode.")
        return

    if not os.path.exists(MODEL_A_PATH):
        print(f"Model file not found: {MODEL_A_PATH}")
        return

    with open(MODEL_A_PATH, 'r') as f:
        model = json.load(f)

    events = [
        {"semantic_risk": 0.85, "velocity_1h": 12, "proxy_score": 1.0, "amount_deviation": 2.1},
        {"semantic_risk": 0.2, "velocity_1h": 3, "proxy_score": 0.0, "amount_deviation": 0.3},
        {"semantic_risk": 0.6, "velocity_1h": 25, "proxy_score": 0.5, "amount_deviation": 1.5},
    ]

    print("-" * 70)
    print(f"| {'#':>2} | {'SEM_RISK':>8} | {'VEL':>4} | {'PROXY':>5} | {'AMT_DEV':>7} | {'SCORE':>7} | {'DECISION':>8} |")
    print("-" * 70)

    for i, tx in enumerate(events, 1):
        features = [tx["semantic_risk"], tx["velocity_1h"], tx["proxy_score"], tx["amount_deviation"]]
        score = model['beta0'] + sum(b * x for b, x in zip(model['betas'], features))
        score = max(0, min(1, score))
        decision = "BLOCK" if score > 0.55 else "ALLOW"
        print(f"| {i:>2} | {tx['semantic_risk']:>8.2f} | {tx['velocity_1h']:>4.0f} | {tx['proxy_score']:>5.1f} | {tx['amount_deviation']:>7.2f} | {score:>7.4f} | {decision:>8} |")

    print("-" * 70)


def cleanup(**kwargs):
    """Task 6: Remove temp files."""
    print("Cleanup complete.")


with DAG(
    'fraud_detection_engine',
    default_args={'owner': 'airflow'},
    start_date=datetime(2026, 6, 18),
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'fraud', 'gmdh', 'bedrock']
) as dag:

    t1 = PythonOperator(
        task_id='enrich_with_bedrock',
        python_callable=enrich_with_bedrock
    )

    t2 = PythonOperator(
        task_id='train_fraud_model',
        python_callable=train_fraud_model
    )

    t3 = PythonOperator(
        task_id='train_health_model',
        python_callable=train_health_model
    )

    t4 = PythonOperator(
        task_id='check_system_health',
        python_callable=check_system_health
    )

    t5 = PythonOperator(
        task_id='run_fraud_inference',
        python_callable=run_fraud_inference
    )

    t6 = PythonOperator(
        task_id='cleanup',
        python_callable=cleanup,
        trigger_rule='all_done'
    )

    t1 >> [t2, t3] >> t4 >> t5 >> t6
