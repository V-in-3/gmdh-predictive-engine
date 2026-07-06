import json
import logging
import os
from datetime import datetime

from airflow.decorators import dag, task
from airflow.models import Variable

DATA_DIR = '/opt/airflow/project/data'
DEFAULT_BENCHMARK_PATH = f'{DATA_DIR}/fraud_production_50k.csv'


@dag(
    default_args={'owner': 'airflow'},
    start_date=datetime(2026, 6, 18),
    schedule_interval=None,
    catchup=False,
    tags=['benchmark', 'ml', 'fraud']
)
def benchmark_evaluation_dag():

    @task
    def run_benchmark():
        """Run benchmark evaluation on the local minimal dataset and save outputs."""
        from scripts.benchmark_eval import evaluate_benchmark_model

        logging.info("Benchmark task started")
        benchmark_url = Variable.get('benchmark_dataset_url', default_var='').strip()
        benchmark_path = (
            benchmark_url
            or os.environ.get('BENCHMARK_DATASET_PATH')
            or os.environ.get('REAL_BENCHMARK_PATH')
            or DEFAULT_BENCHMARK_PATH
        )
        logging.info("Benchmark source: %s", benchmark_path)
        output_path = os.environ.get('BENCHMARK_OUTPUT_PATH', f'{DATA_DIR}/benchmark_normalized.csv')
        model_output_path = os.environ.get('BENCHMARK_MODEL_OUTPUT_PATH', f'{DATA_DIR}/benchmark_model.json')
        metrics_output_path = os.environ.get('BENCHMARK_METRICS_OUTPUT_PATH', f'{DATA_DIR}/benchmark_metrics.json')

        logging.info("Running benchmark evaluation")
        result = evaluate_benchmark_model(
            input_path=benchmark_path,
            output_path=output_path,
            model_output_path=model_output_path,
            metrics_output_path=metrics_output_path,
        )
        logging.info("Benchmark evaluation finished")
        metrics = result.get('metrics', {})
        logging.info(
            "Final benchmark summary: rows=%s fraud_rate=%.4f precision=%.4f recall=%.4f f1=%.4f auc_roc=%.4f",
            result.get('rows', 0),
            result.get('fraud_rate', 0.0),
            metrics.get('precision', 0.0),
            metrics.get('recall', 0.0),
            metrics.get('f1', 0.0),
            metrics.get('auc_roc', 0.0),
        )
        logging.info("Result: %s", json.dumps(result, indent=2))
        return result

    run_benchmark()


benchmark_evaluation_dag()
