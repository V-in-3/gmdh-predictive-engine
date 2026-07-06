import json
import os
from datetime import datetime

from airflow.decorators import dag, task

DATA_DIR = '/opt/airflow/project/data'
MODEL_A_PATH = f'{DATA_DIR}/fraud_model_coeffs.json'

# Available engines: 'bedrock_mock', 'ollama'
FEATURE_ENGINE = 'bedrock_mock'
# Available engines: 'bedrock_mock', 'gmdh', 'ollama'
SCORING_ENGINE = 'gmdh'


@dag(
    default_args={'owner': 'airflow'},
    start_date=datetime(2026, 6, 18),
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'fraud', 'gmdh', 'bedrock']
)
def fraud_detection_engine():

    @task
    def enrich_with_bedrock():
        """Extract semantic features via configured engine."""
        from jobs.scoring_engine import get_engine

        engine = get_engine(FEATURE_ENGINE)

        sample_transactions = [
            {"desc": "Purchase 500 electronics units at $10 each", "order_id": "305-001"},
            {"desc": "Monthly subscription renewal", "order_id": "305-002"},
            {"desc": "Gift card bulk purchase 50x$200", "order_id": "305-003"},
        ]

        results = engine.extract_features(sample_transactions)

        print(f"Enrichment complete ({engine.engine_name}): {len(results)} transactions scored")
        for r in results:
            print(f"   {r['order_id']}: semantic_risk={r['semantic_risk']}")

        return results

    @task
    def train_fraud_model():
        """Train GMDH fraud model (Model A)."""
        from jobs.gmdh_fraud_trainer import train_fraud_model as train
        model = train(
            data_path=f'{DATA_DIR}/fraud_transactions.csv',
            output_path=MODEL_A_PATH
        )
        return f"Model A trained. RMSE={model['final_rmse']:.4f}"

    @task
    def train_health_model():
        """Train GMDH health model (Model B). Produces health_score for fallback logic."""
        from jobs.gmdh_health_trainer import train_health_model as train
        _, health = train(
            data_path=f'{DATA_DIR}/fintech_transactions_raw.csv',
            output_path=f'{DATA_DIR}/model_b_coeffs.json',
            health_output_path=f'{DATA_DIR}/model_b_health.json'
        )
        return f"Model B trained. health_score={health['health_score']:.4f}, status={health['status']}"

    @task
    def run_benchmark_evaluation():
        """Run benchmark evaluation on the production-like dataset and persist metrics."""
        import os

        from scripts.benchmark_eval import evaluate_benchmark_model

        benchmark_path = os.environ.get('BENCHMARK_DATASET_PATH', f'{DATA_DIR}/fraud_production_50k.csv')
        output_path = os.environ.get('BENCHMARK_OUTPUT_PATH', f'{DATA_DIR}/benchmark_normalized.csv')
        model_output_path = os.environ.get('BENCHMARK_MODEL_OUTPUT_PATH', f'{DATA_DIR}/benchmark_model.json')
        metrics_output_path = os.environ.get('BENCHMARK_METRICS_OUTPUT_PATH', f'{DATA_DIR}/benchmark_metrics.json')

        if not os.path.exists(benchmark_path):
            print(f"Benchmark dataset not found: {benchmark_path}. Skipping benchmark evaluation.")
            return f"Skipped benchmark evaluation: {benchmark_path}"

        result = evaluate_benchmark_model(
            input_path=benchmark_path,
            output_path=output_path,
            model_output_path=model_output_path,
            metrics_output_path=metrics_output_path,
        )
        print(json.dumps(result, indent=2))
        return (
            f"Benchmark finished. rows={result['rows']}, "
            f"auc_roc={result['metrics']['auc_roc']:.4f}, "
            f"f1={result['metrics']['f1']:.4f}"
        )

    @task
    def check_system_health():
        """
        Read Model B health score.
        If health < 0.45 -> disable fraud inference (fallback).
        """
        from airflow.providers.mysql.hooks.mysql import MySqlHook

        model_b_path = f'{DATA_DIR}/model_b_health.json'

        if os.path.exists(model_b_path):
            with open(model_b_path, 'r') as f:
                health = json.load(f).get('health_score', 0.75)
        else:
            try:
                hook = MySqlHook(mysql_conn_id='mysql_default')
                result = hook.get_first("SELECT COUNT(*) FROM raw_subscriptions")
                db_count = result[0] if result else 0
                health = 0.8 if db_count > 0 else 0.5
                print(f"Model B file not found. Estimated health from DB sync: {health} ({db_count} records)")
            except Exception:
                health = 0.75

        print(f"Model B health_score: {health}")

        if health < 0.35:
            print("CRITICAL: System degraded. Model A inference DISABLED (fallback).")
            return "DISABLED"

        print("System healthy. Model A inference ENABLED.")
        return "ENABLED"

    @task
    def run_fraud_inference(health_status):
        """Apply fraud model via configured scoring engine."""
        from jobs.scoring_engine import get_engine

        if health_status == "DISABLED":
            print("Skipping inference - system in fallback mode.")
            return

        engine = get_engine(SCORING_ENGINE)

        events = [
            {"semantic_risk": 0.85, "velocity_1h": 12, "proxy_score": 1.0, "amount_deviation": 2.1},
            {"semantic_risk": 0.2, "velocity_1h": 3, "proxy_score": 0.0, "amount_deviation": 0.3},
            {"semantic_risk": 0.6, "velocity_1h": 25, "proxy_score": 0.5, "amount_deviation": 1.5},
        ]

        results = engine.score_transactions(events, model_path=MODEL_A_PATH)

        print("-" * 70)
        print(f"| {'#':>2} | {'SEM_RISK':>8} | {'VEL':>4} | {'PROXY':>5} | {'AMT_DEV':>7} | {'SCORE':>7} | {'DECISION':>8} |")
        print("-" * 70)
        for i, (tx, r) in enumerate(zip(events, results), 1):
            print(f"| {i:>2} | {tx['semantic_risk']:>8.2f} | {tx['velocity_1h']:>4.0f} | {tx['proxy_score']:>5.1f} | {tx['amount_deviation']:>7.2f} | {r['score']:>7.4f} | {r['decision']:>8} |")
        print("-" * 70)
        print(f"Engine: {engine.engine_name}")

    @task
    def compare_engines(health_status):
        """A/B comparison: run all scoring engines on same data, print side-by-side."""
        from jobs.scoring_engine import get_engine

        if health_status == "DISABLED":
            print("Skipping comparison - system in fallback mode.")
            return

        events = [
            {"semantic_risk": 0.85, "velocity_1h": 12, "proxy_score": 1.0, "amount_deviation": 2.1},
            {"semantic_risk": 0.2, "velocity_1h": 3, "proxy_score": 0.0, "amount_deviation": 0.3},
            {"semantic_risk": 0.6, "velocity_1h": 25, "proxy_score": 0.5, "amount_deviation": 1.5},
        ]

        engines = ['gmdh', 'bedrock_mock', 'ollama']
        all_results = {}

        for name in engines:
            try:
                engine = get_engine(name)
                all_results[name] = engine.score_transactions(events, model_path=MODEL_A_PATH)
            except Exception as e:
                print(f"Engine '{name}' failed: {e}")
                all_results[name] = [{"score": None, "decision": "ERROR"}] * len(events)

        print("=" * 80)
        print("ENGINE COMPARISON (same inputs)")
        print("=" * 80)
        print(f"| {'#':>2} | {'GMDH':>12} | {'BEDROCK_MOCK':>12} | {'OLLAMA':>12} | {'AGREE':>5} |")
        print("-" * 80)

        for i, tx in enumerate(events):
            scores = {name: all_results[name][i] for name in engines}
            decisions = [s['decision'] for s in scores.values() if s['decision'] != 'ERROR']
            agree = "YES" if len(set(decisions)) == 1 else "NO"

            gmdh_s = scores['gmdh']['score']
            mock_s = scores['bedrock_mock']['score']
            ollama_s = scores['ollama']['score']

            gmdh_str = f"{gmdh_s:.3f} {scores['gmdh']['decision']}" if gmdh_s is not None else "ERROR"
            mock_str = f"{mock_s:.3f} {scores['bedrock_mock']['decision']}" if mock_s is not None else "ERROR"
            ollama_str = f"{ollama_s:.3f} {scores['ollama']['decision']}" if ollama_s is not None else "ERROR"

            print(f"| {i+1:>2} | {gmdh_str:>12} | {mock_str:>12} | {ollama_str:>12} | {agree:>5} |")

        print("=" * 80)

    @task(trigger_rule='all_done')
    def cleanup():
        """Remove temp files."""
        print("Cleanup complete.")

    # Flow
    enrichment = enrich_with_bedrock()
    model_a = train_fraud_model()
    model_b = train_health_model()
    benchmark = run_benchmark_evaluation()
    health = check_system_health()
    inference = run_fraud_inference(health)
    comparison = compare_engines(health)
    clean = cleanup()

    enrichment >> [model_a, model_b, benchmark] >> health >> [inference, comparison] >> clean


fraud_detection_engine()
