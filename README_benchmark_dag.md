# Benchmark DAG

Copy-paste command:

```powershell
& "C:/Users/User/AppData/Roaming/uv/python/cpython-3.14.6-windows-x86_64-none/python.exe" scripts/benchmark_eval.py data/fraud_production_50k.csv --output_path data/benchmark_normalized.csv --model_output_path data/benchmark_model.json --metrics_output_path data/benchmark_metrics.json
```

This DAG runs the benchmark evaluation workflow for the project.

## What it does

The benchmark DAG:
- loads a benchmark dataset,
- normalizes it into the project feature schema,
- evaluates a simple heuristic fraud model,
- writes normalized output, model metadata, and metrics,
- logs the final benchmark summary.

## Default input

By default the DAG uses:
- data/fraud_production_50k.csv

You can override the source with one of these options:
- Airflow Variable: benchmark_dataset_url
- Environment variable: BENCHMARK_DATASET_PATH
- Environment variable: REAL_BENCHMARK_PATH

## Output files

The DAG writes these files:
- data/benchmark_normalized.csv
- data/benchmark_model.json
- data/benchmark_metrics.json

## How to run

From the project root:

```bash
python scripts/benchmark_eval.py data/fraud_production_50k.csv \
  --output_path data/benchmark_normalized.csv \
  --model_output_path data/benchmark_model.json \
  --metrics_output_path data/benchmark_metrics.json
```

For model comparison:

```bash
python scripts/compare_benchmark_models.py data/fraud_production_50k.csv \
  --output_path data/benchmark_model_comparison.json
```

## Expected result

The run logs should include:
- progress every 5000 rows,
- normalization and metric computation steps,
- a final summary with rows, fraud_rate, precision, recall, F1, and AUC-ROC.
