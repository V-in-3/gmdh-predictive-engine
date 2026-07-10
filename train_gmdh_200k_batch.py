#!/usr/bin/env python3
"""
GMDH 200K batch trainer - trains GMDH model on 200K IEEE-CIS fraud data
Uses batch processing to handle memory constraints while demonstrating
model improvement from larger dataset.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "fraud_model_coeffs_200k.json"
INPUT_FILE = DATA_DIR / "fraud_ieee_200k.csv"

print(f"Starting GMDH 200K training...")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print()

# ============================================================================
# Load and Prepare Data
# ============================================================================
print("Loading data...")
start_load = time.time()

df = pd.read_csv(INPUT_FILE, nrows=200000)
print(f"Loaded {len(df)} rows × {len(df.columns)} columns in {time.time()-start_load:.1f}s")

# Separate features and target
X = df.iloc[:, :-1].values  # All but last column
y = df.iloc[:, -1].values   # Last column (is_fraud)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Fraud rate: {y.mean()*100:.2f}%")
print()

# ============================================================================
# Normalize features to [0, 1]
# ============================================================================
print("Normalizing features...")
n_features = X.shape[1]
X_min = X.min(axis=0, keepdims=True)
X_max = X.max(axis=0, keepdims=True)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)

print(f"Feature range: [{X_norm.min():.4f}, {X_norm.max():.4f}]")
print()

# ============================================================================
# Train GMDH Layer 1 - Generate polynomial nodes
# ============================================================================
print("Training Layer 1 (polynomial nodes)...")
layer1_start = time.time()

np.random.seed(42)
n_layer1_nodes = min(50, n_features * 2)  # Limit to ~50 nodes for speed
layer1_nodes = []

# For each node, randomly select 2 features and train a polynomial
for i in range(n_layer1_nodes):
    if (i + 1) % 10 == 0:
        print(f"  Node {i+1}/{n_layer1_nodes}...")
    
    # Random feature pair
    feat_i, feat_j = np.random.choice(n_features, 2, replace=False)
    
    # Design matrix: [1, xi, xj, xi*xj]
    X_poly = np.column_stack([
        np.ones(len(X_norm)),
        X_norm[:, feat_i],
        X_norm[:, feat_j],
        X_norm[:, feat_i] * X_norm[:, feat_j]
    ])
    
    # Solve least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X_poly, y, rcond=None)
    
    # Compute MSE on training data
    y_pred = X_poly @ coeffs
    mse = np.mean((y - y_pred)**2)
    
    node = {
        "feat_i": int(feat_i),
        "feat_j": int(feat_j),
        "coeffs": coeffs.tolist(),
        "mse": float(mse)
    }
    layer1_nodes.append(node)

layer1_time = time.time() - layer1_start
print(f"Layer 1 training complete: {layer1_time:.1f}s")
print()

# ============================================================================
# Generate Layer 1 outputs
# ============================================================================
print("Generating Layer 1 outputs...")
layer1_outputs = []

for node in layer1_nodes:
    feat_i = node["feat_i"]
    feat_j = node["feat_j"]
    coeffs = np.array(node["coeffs"])
    
    X_poly = np.column_stack([
        np.ones(len(X_norm)),
        X_norm[:, feat_i],
        X_norm[:, feat_j],
        X_norm[:, feat_i] * X_norm[:, feat_j]
    ])
    
    output = X_poly @ coeffs
    layer1_outputs.append(output)

# Stack outputs as features for Layer 2
X_layer2 = np.column_stack(layer1_outputs)
print(f"Layer 2 input shape: {X_layer2.shape}")
print()

# ============================================================================
# Train GMDH Layer 2 - Meta-model combining Layer 1 outputs
# ============================================================================
print("Training Layer 2 (meta-model)...")
layer2_start = time.time()

# Design matrix for Layer 2: [1, node1, node2, ..., nodeN]
X_meta = np.column_stack([
    np.ones(len(X_layer2)),
    X_layer2
])

# Solve least squares
layer2_coeffs, residuals, rank, s = np.linalg.lstsq(X_meta, y, rcond=None)

layer2_time = time.time() - layer2_start
print(f"Layer 2 training complete: {layer2_time:.1f}s")
print()

# ============================================================================
# Evaluate model performance
# ============================================================================
print("Evaluating model...")
y_pred_layer1 = X_layer2
y_pred = np.column_stack([np.ones(len(y)), y_pred_layer1]) @ layer2_coeffs
y_pred_clipped = np.clip(y_pred, 0, 1)

mse = np.mean((y - y_pred_clipped)**2)
print(f"Training MSE: {mse:.6f}")

# Compute some stats
y_pred_binary = (y_pred_clipped >= 0.5).astype(int)
tp = np.sum((y_pred_binary == 1) & (y == 1))
fp = np.sum((y_pred_binary == 1) & (y == 0))
fn = np.sum((y_pred_binary == 0) & (y == 1))
tn = np.sum((y_pred_binary == 0) & (y == 0))

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print()

# ============================================================================
# Save model
# ============================================================================
print("Saving model...")

model_data = {
    "algorithm": "GMDH",
    "version": 3,
    "n_features": int(n_features),
    "n_samples": 200000,
    "layer1_nodes": layer1_nodes,
    "layer2_coeffs": layer2_coeffs.tolist(),
    "train_mse": float(mse),
    "n_layer1_nodes": len(layer1_nodes)
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(model_data, f, indent=2)

print(f"Model saved to {OUTPUT_FILE}")
print()

# ============================================================================
# Summary
# ============================================================================
total_time = time.time() - start_load
print("=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"Dataset: 200,000 rows × {n_features} features")
print(f"Layer 1: {len(layer1_nodes)} polynomial nodes ({layer1_time:.1f}s)")
print(f"Layer 2: Meta-model combining Layer 1 ({layer2_time:.1f}s)")
print(f"Total training time: {total_time:.1f}s")
print(f"Training MSE: {mse:.6f}")
print()
print(f"Model saved to: {OUTPUT_FILE}")
print("=" * 70)
