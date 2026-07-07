# Benchmark Validation — DAG Integration

The benchmark evaluation is integrated directly into `fraud_detection_dag.py` as a validation gate after model training. It is not a standalone scheduled DAG — it runs as part of the main fraud detection flow.

---

## How it fits into the main DAG flow

```
enrich_with_bedrock
        |
   +----+----+
   |         |
train_fraud  train_health     <-- skipped if ENABLE_NIGHTLY_TRAINING=False
   |         |
   +----+----+
        |
run_benchmark_evaluation      <-- scores fraud_production_50k.csv, writes metrics
        |
validate_benchmark_gate       <-- checks F1, precision, recall, AUC against thresholds
        |
  PASS? / FAIL?
   |         |
PASS        FAIL --> inference and comparison are SKIPPED, warning logged
   |
check_system_health
        |
   +----+----+
   |         |
run_fraud  compare_engines
_inference
   |         |
   +----+----+
        |
    cleanup
```

---

## Nightly training control

In `dags/fraud_detection_dag.py`, at the top of the file:

```python
# Set to False to skip model retraining (e.g. for inference-only nightly runs).
ENABLE_NIGHTLY_TRAINING = True
```

- `True` — models are retrained on every DAG run, then benchmark validates the new model.
- `False` — training is skipped, existing coefficients on disk are reused, benchmark still runs to validate the current model state.

---

## Benchmark gate thresholds

Also at the top of `dags/fraud_detection_dag.py`:

```python
BENCHMARK_THRESHOLDS = {
    'f1': 0.45,
    'precision': 0.50,
    'recall': 0.40,
    'auc_roc': 0.78,
}
```

If any metric falls below its threshold, `validate_benchmark_gate` returns `"FAIL"` and logs a warning. Downstream inference tasks check this result and skip execution.

---

## Default benchmark input

```
data/fraud_production_50k.csv
```

Override via environment variable:

```bash
BENCHMARK_DATASET_PATH=/path/to/your/dataset.csv
```

---

## Output files

| File | Content |
|------|---------|
| `data/benchmark_normalized.csv` | Normalized rows in project feature schema |
| `data/benchmark_model.json` | Simple rule-based model weights used for scoring |
| `data/benchmark_metrics.json` | Metrics: precision, recall, F1, AUC-ROC, TP/FP/FN/TN |
| `data/benchmark_model_comparison.json` | Side-by-side comparison of model variants |

---

## Run benchmark manually (without Airflow)

```powershell
& "C:/Users/User/AppData/Roaming/uv/python/cpython-3.14.6-windows-x86_64-none/python.exe" scripts/benchmark_eval.py data/fraud_production_50k.csv --output_path data/benchmark_normalized.csv --model_output_path data/benchmark_model.json --metrics_output_path data/benchmark_metrics.json
```

For model variant comparison:

```powershell
& "C:/Users/User/AppData/Roaming/uv/python/cpython-3.14.6-windows-x86_64-none/python.exe" scripts/compare_benchmark_models.py data/fraud_production_50k.csv --output_path data/benchmark_model_comparison.json
```

---

## What the logs show

`run_benchmark_evaluation` task logs:
- progress every 5000 rows with percent remaining
- final summary: rows, fraud_rate, precision, recall, F1, AUC-ROC

`validate_benchmark_gate` task logs:
```
--- BENCHMARK GATE VALIDATION ---
  f1           0.7208  (threshold >= 0.45)  ->  PASS
  precision    0.8304  (threshold >= 0.50)  ->  PASS
  recall       0.6367  (threshold >= 0.40)  ->  PASS
  auc_roc      0.9866  (threshold >= 0.78)  ->  PASS
BENCHMARK GATE: PASS — model quality meets all thresholds. Inference ENABLED.
```

If any metric fails:
```
BENCHMARK GATE: FAIL — one or more metrics are below threshold.
Fraud inference will be SKIPPED to prevent deploying a degraded model.
```
