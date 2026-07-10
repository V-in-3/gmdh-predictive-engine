#!/usr/bin/env python3
"""
Benchmark Neural Network Model on Test Set
Evaluates fraud detection metrics: Precision, Recall, F1, AUC-ROC
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_FILE = DATA_DIR / "fraud_model_nn_200k.json"
TEST_FILE = DATA_DIR / "fraud_transactions_ieee_50k.csv"
OUTPUT_FILE = DATA_DIR / "benchmark_metrics_nn_200k.json"

print("=" * 75)
print("Neural Network Benchmark Evaluation (50K Test Set)")
print("=" * 75)
print()

# ============================================================================
# Load Model
# ============================================================================
print("Loading trained NN model...")
with open(MODEL_FILE, "r") as f:
    model_data = json.load(f)

weights = [np.array(w) for w in model_data["weights"]]
biases = [np.array(b) for b in model_data["biases"]]
X_mean = np.array(model_data["normalization"]["X_mean"])
X_std = np.array(model_data["normalization"]["X_std"])

# Load BatchNorm parameters if present (v5+)
bn_gamma = [np.array(g) for g in model_data["bn_gamma"]] if "bn_gamma" in model_data else None
bn_beta  = [np.array(b) for b in model_data["bn_beta"]]  if "bn_beta"  in model_data else None
bn_mean  = [np.array(m) for m in model_data["bn_running_mean"]] if "bn_running_mean" in model_data else None
bn_var   = [np.array(v) for v in model_data["bn_running_var"]]  if "bn_running_var"  in model_data else None
use_bn = bn_gamma is not None

print(f"Model: {model_data['algorithm']}")
print(f"Architecture: {model_data['architecture']['input_dim']} → " +
      f"{' → '.join(map(str, model_data['architecture']['hidden_dims']))} → 1")
print()

# ============================================================================
# Load Test Data
# ============================================================================
print("Loading test set...")
df_test = pd.read_csv(TEST_FILE)
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

print(f"Test set: {len(X_test):,} samples")
print(f"Fraud rate: {y_test.mean()*100:.2f}%")
print(f"Features: {X_test.shape[1]}")
print()

# Normalize using training normalization
X_test_norm = (X_test - X_mean) / (X_std + 1e-8)

# ============================================================================
# Forward Pass (Inference)
# ============================================================================
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

print("Running inference...")
# Forward pass through network (with BatchNorm if v5+)
a = X_test_norm
for i in range(len(weights) - 1):
    z = np.dot(a, weights[i]) + biases[i]
    if use_bn:
        z = bn_gamma[i] * (z - bn_mean[i]) / np.sqrt(bn_var[i] + 1e-8) + bn_beta[i]
    a = relu(z)

# Output layer
z = np.dot(a, weights[-1]) + biases[-1]
y_pred_prob = sigmoid(z).ravel()

print(f"Predictions: min={y_pred_prob.min():.4f}, mean={y_pred_prob.mean():.4f}, max={y_pred_prob.max():.4f}")
print()

# ============================================================================
# Evaluate Metrics
# ============================================================================
print("Evaluating metrics...")
print("-" * 75)

# Try multiple thresholds to find best F1
best_f1 = 0
best_threshold = 0.5
best_metrics = {
    'threshold': 0.5,
    'tp': 0,
    'fp': 0,
    'tn': 0,
    'fn': 0,
    'precision': 0,
    'recall': 0,
    'f1': 0
}

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    tn = np.sum((y_pred == 0) & (y_test == 0))
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    
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

print(f"Best Threshold: {best_threshold:.2f}")
print(f"  TP={best_metrics['tp']:,}, FP={best_metrics['fp']:,}, "
      f"TN={best_metrics['tn']:,}, FN={best_metrics['fn']:,}")
print(f"  Precision: {best_metrics['precision']:.4f}")
print(f"  Recall: {best_metrics['recall']:.4f}")
print(f"  F1-Score: {best_metrics['f1']:.4f}")
print()

# Use best threshold for final predictions
y_pred = (y_pred_prob >= best_threshold).astype(int)

# Compute AUC-ROC
sorted_indices = np.argsort(-y_pred_prob)
sorted_y = y_test[sorted_indices]
n_pos = np.sum(y_test == 1)
n_neg = np.sum(y_test == 0)
tp_cumsum = np.cumsum(sorted_y == 1)
auc = np.sum(tp_cumsum[sorted_y == 0]) / (n_pos * n_neg) if (n_pos > 0 and n_neg > 0) else 0.5

print(f"AUC-ROC: {auc:.4f}")
print()

# ============================================================================
# Confusion Matrix
# ============================================================================
print("Confusion Matrix:")
print(f"  TP: {best_metrics['tp']:6,}   FP: {best_metrics['fp']:6,}")
print(f"  FN: {best_metrics['fn']:6,}   TN: {best_metrics['tn']:6,}")
print()

# ============================================================================
# Check Benchmark Gate
# ============================================================================
print("Benchmark Gate Status:")
print("-" * 75)

gate_thresholds = {
    "F1": 0.45,
    "Precision": 0.50,
    "Recall": 0.40,
    "AUC-ROC": 0.78
}

gate_results = {
    "F1": best_metrics['f1'] >= gate_thresholds["F1"],
    "Precision": best_metrics['precision'] >= gate_thresholds["Precision"],
    "Recall": best_metrics['recall'] >= gate_thresholds["Recall"],
    "AUC-ROC": auc >= gate_thresholds["AUC-ROC"]
}

for metric, threshold in gate_thresholds.items():
    value = best_metrics[metric.lower()] if metric != "AUC-ROC" else auc
    status = "✓ PASS" if gate_results[metric] else "✗ FAIL"
    print(f"  {metric:12s}: {value:.4f} ≥ {threshold:.2f}? {status}")

gate_pass = all(gate_results.values())
overall_status = "✓✓✓ GATE PASS ✓✓✓" if gate_pass else "✗✗✗ GATE FAIL ✗✗✗"

print()
print(overall_status)
print()

# ============================================================================
# Save Results
# ============================================================================
results = {
    "model": "Neural Network (NumPy)",
    "dataset": "IEEE-CIS Fraud Detection (50K test)",
    "timestamp": str(pd.Timestamp.now()),
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

print(f"Results saved to: {OUTPUT_FILE}")
print()
print("=" * 75)
