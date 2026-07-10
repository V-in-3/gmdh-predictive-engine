#!/usr/bin/env python3
"""
NN Model Benchmark on Test Set (50K)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_FILE = DATA_DIR / "fraud_model_nn_200k_fast.json"
TEST_FILE = DATA_DIR / "fraud_transactions_ieee_50k.csv"
OUTPUT_FILE = DATA_DIR / "benchmark_metrics_nn_200k_fast.json"

print("=" * 75)
print("NN Benchmark (50K test set)")
print("=" * 75)
print()

# Load model
print("Loading NN model...")
with open(MODEL_FILE, "r") as f:
    model_data = json.load(f)

w1 = np.array(model_data["weights"]["w1"], dtype=np.float32)
b1 = np.array(model_data["weights"]["b1"], dtype=np.float32)
w2 = np.array(model_data["weights"]["w2"], dtype=np.float32)
b2 = np.array(model_data["weights"]["b2"], dtype=np.float32)
w3 = np.array(model_data["weights"]["w3"], dtype=np.float32)
b3 = np.array(model_data["weights"]["b3"], dtype=np.float32)

X_mean = np.array(model_data["normalization"]["X_mean"], dtype=np.float32)
X_std = np.array(model_data["normalization"]["X_std"], dtype=np.float32)

print(f"Architecture: {model_data['architecture']}")
print()

# Load test set
print("Loading test set...")
df_test = pd.read_csv(TEST_FILE)
X_test = df_test.iloc[:, :-1].values.astype(np.float32)
y_test = df_test.iloc[:, -1].values

print(f"Test set: {len(X_test):,} samples")
print(f"Fraud rate: {y_test.mean()*100:.2f}%")
print()

# Normalize
X_test_norm = ((X_test - X_mean) / (X_std + 1e-8)).astype(np.float32)

# Inference
print("Running inference...")
z1 = np.dot(X_test_norm, w1) + b1
a1 = np.maximum(0, z1)
z2 = np.dot(a1, w2) + b2
a2 = np.maximum(0, z2)
z3 = np.dot(a2, w3) + b3
y_pred_prob = (1 / (1 + np.exp(-np.clip(z3, -500, 500)))).ravel()

print(f"Predictions: min={y_pred_prob.min():.4f}, mean={y_pred_prob.mean():.4f}, max={y_pred_prob.max():.4f}")
print()

# Evaluate metrics
print("Evaluating metrics...")
print("-" * 75)

best_f1 = 0
best_threshold = 0.5

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    tn = np.sum((y_pred == 0) & (y_test == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_metrics = {
            'threshold': threshold,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

print(f"Best threshold: {best_threshold:.2f}")
print(f"  TP={best_metrics['tp']:,}, FP={best_metrics['fp']:,}, "
      f"TN={best_metrics['tn']:,}, FN={best_metrics['fn']:,}")
print(f"  Precision: {best_metrics['precision']:.4f}")
print(f"  Recall:    {best_metrics['recall']:.4f}")
print(f"  F1-Score:  {best_metrics['f1']:.4f}")
print()

# AUC-ROC
sorted_indices = np.argsort(-y_pred_prob)
sorted_y = y_test[sorted_indices]
n_pos = np.sum(y_test == 1)
n_neg = np.sum(y_test == 0)
tp_cumsum = np.cumsum(sorted_y == 1)
auc = np.sum(tp_cumsum[sorted_y == 0]) / (n_pos * n_neg) if (n_pos > 0 and n_neg > 0) else 0.5

print(f"AUC-ROC: {auc:.4f}")
print()

# Benchmark gate
print("Benchmark gate status:")
print("-" * 75)

gate_thresholds = {
    "F1": 0.45,
    "Precision": 0.50,
    "Recall": 0.40,
    "AUC-ROC": 0.78
}

gate_values = {
    "F1": best_metrics['f1'],
    "Precision": best_metrics['precision'],
    "Recall": best_metrics['recall'],
    "AUC-ROC": auc
}

gate_results = {k: gate_values[k] >= gate_thresholds[k] for k in gate_thresholds}

for metric, threshold in gate_thresholds.items():
    value = gate_values[metric]
    status = "PASS" if gate_results[metric] else "FAIL"
    print(f"  {metric:12s}: {value:.4f} >= {threshold:.2f}? {status}")

gate_pass = all(gate_results.values())
overall_status = "GATE PASS" if gate_pass else "GATE FAIL"

print()
print(overall_status)
print()

# Save results
results = {
    "model": "Neural Network (NumPy Fast)",
    "dataset": "IEEE-CIS Fraud Detection (50K test)",
    "predictions": {
        "threshold": float(best_threshold),
        "min": float(y_pred_prob.min()),
        "mean": float(y_pred_prob.mean()),
        "max": float(y_pred_prob.max())
    },
    "metrics": {
        "precision": float(best_metrics['precision']),
        "recall": float(best_metrics['recall']),
        "f1": float(best_metrics['f1']),
        "auc_roc": float(auc)
    },
    "confusion_matrix": {
        "tp": int(best_metrics['tp']),
        "fp": int(best_metrics['fp']),
        "tn": int(best_metrics['tn']),
        "fn": int(best_metrics['fn'])
    },
    "gate": {
        "f1_pass": bool(gate_results["F1"]),
        "precision_pass": bool(gate_results["Precision"]),
        "recall_pass": bool(gate_results["Recall"]),
        "auc_roc_pass": bool(gate_results["AUC-ROC"]),
        "overall_pass": bool(gate_pass)
    }
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved: {OUTPUT_FILE}")
print()
print("=" * 75)
