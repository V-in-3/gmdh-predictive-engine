#!/usr/bin/env python3
"""
Benchmark evaluation for 200K GMDH model on IEEE-CIS test data.
Shows how 200K model compares to gate thresholds.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Model files
MODEL_200K = DATA_DIR / "fraud_model_coeffs_200k.json"

# Test data - use 50K IEEE test data
TEST_FILE = DATA_DIR / "fraud_transactions_ieee_50k.csv"
OUTPUT_FILE = DATA_DIR / "benchmark_metrics_200k.json"

# Gate thresholds from fraud_detection_dag.py
GATE_THRESHOLDS = {
    'f1': 0.45,
    'precision': 0.50,
    'recall': 0.40,
    'auc_roc': 0.78
}

print("=" * 75)
print("BENCHMARK EVALUATION: 200K GMDH Model")
print("=" * 75)
print()

# Load test data
print(f"Loading test data from {TEST_FILE}...")
df_test = pd.read_csv(TEST_FILE, nrows=50000)
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

print(f"Test set: {X_test.shape[0]} rows × {X_test.shape[1]} features")
print(f"Fraud rate: {y_test.mean()*100:.2f}%")
print()

# Normalize test features
X_test_min = X_test.min(axis=0, keepdims=True)
X_test_max = X_test.max(axis=0, keepdims=True)
X_test_norm = (X_test - X_test_min) / (X_test_max - X_test_min + 1e-8)

# ============================================================================
# Evaluate 200K Model
# ============================================================================
print("Evaluating 200K Model...")
print("-" * 75)

with open(MODEL_200K) as f:
    model_200k = json.load(f)

layer1_nodes = model_200k["layer1_nodes"]
layer2_coeffs = np.array(model_200k["layer2_coeffs"])

print(f"Model: {model_200k['n_samples']:,} training samples, {model_200k['n_features']} features")
print(f"Layers: {len(layer1_nodes)} Layer-1 nodes → 1 Layer-2 meta-model")
print()
layer1_outputs = []
for node in layer1_nodes:
    feat_i = node["feat_i"]
    feat_j = node["feat_j"]
    coeffs = np.array(node["coeffs"])
    
    X_poly = np.column_stack([
        np.ones(len(X_test_norm)),
        X_test_norm[:, feat_i],
        X_test_norm[:, feat_j],
        X_test_norm[:, feat_i] * X_test_norm[:, feat_j]
    ])
    
    output = X_poly @ coeffs
    layer1_outputs.append(output)

# Layer 2 prediction
X_layer2 = np.column_stack(layer1_outputs)
X_meta = np.column_stack([np.ones(len(X_layer2)), X_layer2])
y_pred_200k = np.clip(X_meta @ layer2_coeffs, 0, 1)

# Compute metrics for 200K
y_pred_200k_binary = (y_pred_200k >= 0.5).astype(int)

tp_200k = np.sum((y_pred_200k_binary == 1) & (y_test == 1))
fp_200k = np.sum((y_pred_200k_binary == 1) & (y_test == 0))
fn_200k = np.sum((y_pred_200k_binary == 0) & (y_test == 1))
tn_200k = np.sum((y_pred_200k_binary == 0) & (y_test == 0))

precision_200k = tp_200k / (tp_200k + fp_200k + 1e-8)
recall_200k = tp_200k / (tp_200k + fn_200k + 1e-8)
f1_200k = 2 * precision_200k * recall_200k / (precision_200k + recall_200k + 1e-8)

# Calculate AUC manually - sort by prediction confidence
sorted_indices = np.argsort(-y_pred_200k.ravel())
sorted_y = y_test[sorted_indices]
n_pos = np.sum(y_test == 1)
n_neg = np.sum(y_test == 0)
if n_pos > 0 and n_neg > 0:
    tp_cumsum = np.cumsum(sorted_y == 1)
    auc_200k = np.sum(tp_cumsum[sorted_y == 0]) / (n_pos * n_neg)
else:
    auc_200k = 0.5

print(f"Precision: {precision_200k:.4f}")
print(f"Recall:    {recall_200k:.4f}")
print(f"F1 Score:  {f1_200k:.4f}")
print(f"AUC-ROC:   {auc_200k:.4f}")
print(f"TP: {tp_200k}, FP: {fp_200k}, FN: {fn_200k}, TN: {tn_200k}")
print()


# ============================================================================
# Compare to Gate Thresholds
# ============================================================================
print("=" * 75)
print("GATE THRESHOLD COMPARISON")
print("=" * 75)
print()

metrics = {
    'precision': precision_200k,
    'recall': recall_200k,
    'f1': f1_200k,
    'auc_roc': auc_200k,
}

gate_status = "PASS" if all(metrics[k] >= GATE_THRESHOLDS[k] for k in metrics) else "FAIL"

print(f"{'Metric':<15} {'Threshold':<15} {'Actual':<15} {'Status':<15}")
print("-" * 75)

for metric, threshold in GATE_THRESHOLDS.items():
    actual = metrics[metric]
    status = "✓ PASS" if actual >= threshold else "✗ FAIL"
    print(f"{metric:<15} {threshold:<15.4f} {actual:<15.4f} {status:<15}")

print()
print(f"Overall Gate Status: {gate_status}")
print()

# ============================================================================
# Save 200K metrics
# ============================================================================
metrics_dict = {
    "dataset": "IEEE-CIS 50K test set",
    "model": "200K training samples",
    "model_file": str(MODEL_200K),
    "n_features": model_200k["n_features"],
    "n_layer1_nodes": len(layer1_nodes),
    "precision": float(precision_200k),
    "recall": float(recall_200k),
    "f1": float(f1_200k),
    "auc_roc": float(auc_200k),
    "tp": int(tp_200k),
    "fp": int(fp_200k),
    "fn": int(fn_200k),
    "tn": int(tn_200k),
    "gate_status": gate_status,
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(metrics_dict, f, indent=2)

print(f"Metrics saved to: {OUTPUT_FILE}")
print("=" * 75)
