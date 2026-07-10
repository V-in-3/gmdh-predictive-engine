#!/usr/bin/env python3
"""
Benchmark for GMDH model with polynomial form search.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

MODEL_FILE = DATA_DIR / "fraud_model_coeffs_200k_gmdh_proper.json"
TEST_FILE = DATA_DIR / "fraud_transactions_ieee_50k.csv"
OUTPUT_FILE = DATA_DIR / "benchmark_metrics_200k_gmdh_proper.json"

GATE_THRESHOLDS = {
    'f1': 0.45,
    'precision': 0.50,
    'recall': 0.40,
    'auc_roc': 0.78
}

print("=" * 75)
print("BENCHMARK: GMDH with Polynomial Form Search")
print("=" * 75)
print()

# Load test data
print(f"Loading test data...")
df_test = pd.read_csv(TEST_FILE, nrows=50000)
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

print(f"Test set: {X_test.shape[0]:,} rows × {X_test.shape[1]} features")
print(f"Fraud rate: {y_test.mean()*100:.2f}%")
print()

# Normalize
X_test_min = X_test.min(axis=0, keepdims=True)
X_test_max = X_test.max(axis=0, keepdims=True)
X_test_norm = (X_test - X_test_min) / (X_test_max - X_test_min + 1e-8)

# Load model
with open(MODEL_FILE) as f:
    model_data = json.load(f)

print(f"Model: {model_data['description']}")
print(f"  Evaluated: {model_data['layer1_candidates_evaluated']} candidates")
print(f"  Selected: {model_data['layer1_count']} best nodes")
print()

# ============================================================================
# Helper function to generate polynomial bases
# ============================================================================
def generate_polynomial_bases(xi, xj, form_type):
    n = len(xi)
    ones = np.ones(n)
    
    if form_type == 1:
        return np.column_stack([ones, xi, xj])
    elif form_type == 2:
        return np.column_stack([ones, xi, xj, xi*xj])
    elif form_type == 3:
        return np.column_stack([ones, xi, xj, xi**2, xj**2])
    elif form_type == 4:
        return np.column_stack([ones, xi, xj, xi*xj, xi**2, xj**2])
    elif form_type == 5:
        return np.column_stack([ones, xi, xj, xi*xj, xi**2, xj**2, xi**2 * xj])
    elif form_type == 6:
        return np.column_stack([ones, xi, xj, xi*xj, xi**2, xj**2, xi**2*xj, xi*xj**2])

# ============================================================================
# Generate predictions
# ============================================================================
print("Generating predictions...")
print("-" * 75)

layer1_nodes = model_data["layer1_nodes"]
layer2_coeffs = np.array(model_data["layer2_coeffs"])

# Generate Layer 1 outputs
layer1_outputs = []
for node in layer1_nodes:
    feat_i = node["feat_i"]
    feat_j = node["feat_j"]
    form_type = node["form_type"]
    coeffs = np.array(node["coeffs"])
    
    X_poly = generate_polynomial_bases(
        X_test_norm[:, feat_i],
        X_test_norm[:, feat_j],
        form_type
    )
    
    output = X_poly @ coeffs
    layer1_outputs.append(output)

# Layer 2 prediction
X_layer2 = np.column_stack(layer1_outputs)
X_meta = np.column_stack([np.ones(len(X_layer2)), X_layer2])
y_pred = np.clip(X_meta @ layer2_coeffs, 0, 1)

# ============================================================================
# Compute metrics
# ============================================================================
y_pred_binary = (y_pred >= 0.5).astype(int)

tp = np.sum((y_pred_binary == 1) & (y_test == 1))
fp = np.sum((y_pred_binary == 1) & (y_test == 0))
fn = np.sum((y_pred_binary == 0) & (y_test == 1))
tn = np.sum((y_pred_binary == 0) & (y_test == 0))

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

# AUC
sorted_indices = np.argsort(-y_pred.ravel())
sorted_y = y_test[sorted_indices]
n_pos = np.sum(y_test == 1)
n_neg = np.sum(y_test == 0)
if n_pos > 0 and n_neg > 0:
    tp_cumsum = np.cumsum(sorted_y == 1)
    auc = np.sum(tp_cumsum[sorted_y == 0]) / (n_pos * n_neg)
else:
    auc = 0.5

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")
print(f"Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print()

# ============================================================================
# Gate Validation
# ============================================================================
print("=" * 75)
print("GATE THRESHOLD COMPARISON")
print("=" * 75)
print()

metrics = {
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'auc_roc': auc,
}

gate_status = "PASS" if all(metrics[k] >= GATE_THRESHOLDS[k] for k in metrics) else "FAIL"

print(f"{'Metric':<15} {'Threshold':<15} {'Actual':<15} {'Status':<15}")
print("-" * 75)

for metric, threshold in GATE_THRESHOLDS.items():
    actual = metrics[metric]
    status = "✓ PASS" if actual >= threshold else "✗ FAIL"
    print(f"{metric:<15} {threshold:<15.4f} {actual:<15.4f} {status:<15}")

print()
print(f"Overall Gate: {gate_status}")
print()

# ============================================================================
# Save results
# ============================================================================
results = {
    "model": "GMDH with Polynomial Form Search",
    "candidates_evaluated": 1800,
    "candidates_forms": "6 polynomial forms",
    "nodes_selected": 50,
    "test_set": "IEEE-CIS 50K",
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "auc_roc": float(auc),
    "tp": int(tp),
    "fp": int(fp),
    "fn": int(fn),
    "tn": int(tn),
    "gate_status": gate_status,
    "model_mse": float(model_data["full_mse"]),
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved: {OUTPUT_FILE}")
print("=" * 75)
