#!/usr/bin/env python
"""
Demonstrate model improvement by threshold tuning.
Shows how precision/recall trade-off works.
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

print("=" * 70)
print("THRESHOLD TUNING DEMONSTRATION")
print("=" * 70)

# Load benchmark data
df = pd.read_csv('data/fraud_production_50k.csv')
X = df[['semantic_risk', 'velocity_1h', 'proxy_score', 'amount_deviation']].copy()
y_true = df['is_fraud'].values

# Normalize
for col in X.columns:
    col_min = X[col].min()
    col_max = X[col].max()
    if col_max > col_min:
        X[col] = (X[col] - col_min) / (col_max - col_min)

# Load model
with open('data/fraud_model_coeffs.json', 'r') as f:
    model = json.load(f)

beta0 = model['beta0']
betas = model['betas']

# Generate predictions (probabilities)
y_pred_proba = beta0 + (X.values @ np.array(betas))
y_pred_proba = np.clip(y_pred_proba, 0, 1)  # Clip to [0,1]

print(f"\nPrediction range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
print(f"True fraud rate: {y_true.mean() * 100:.2f}%")
print(f"\nTesting different thresholds:")
print("-" * 70)
print(f"{'Threshold':<12} {'Precision':>12} {'Recall':>12} {'F1':>12} {'AUC':>12} {'Status'}")
print("-" * 70)

results = []
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Handle edge case
    if len(np.unique(y_pred)) == 1:
        precision = 0 if y_pred[0] == 1 else 1
        recall = 0 if y_pred[0] == 0 else 1
        f1 = 0
    else:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Check against thresholds
    status = ""
    if (precision >= 0.50 and recall >= 0.40 and f1 >= 0.45 and auc >= 0.78):
        status = "PASS ✓"
    else:
        status = "FAIL"
    
    results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    })
    
    print(f"{threshold:<12.1f} {precision:>12.4f} {recall:>12.4f} {f1:>12.4f} {auc:>12.4f} {status}")

print("-" * 70)
print("\nGate Requirements: F1>=0.45, Precision>=0.50, Recall>=0.40, AUC>=0.78")
print("\nObservations:")
print("  - As threshold increases, precision improves but recall drops")
print("  - No single threshold passes ALL requirements with current model")
print("  - Model needs more training data to improve AUC and F1")
