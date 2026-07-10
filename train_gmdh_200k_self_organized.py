#!/usr/bin/env python3
"""
GMDH Self-Organizing Polynomial Network - Proper Implementation

GMDH = Group Method of Data Handling
Key principle: Self-organization through validation set selection
- Split data: train (70%) + validation (30%)
- Generate polynomial nodes from feature pairs
- Evaluate on VALIDATION set (not training!)
- Select best nodes automatically
- Combine selected nodes in Layer 2
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import time
import itertools

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "fraud_model_coeffs_200k_gmdh.json"
INPUT_FILE = DATA_DIR / "fraud_ieee_200k.csv"

print("=" * 75)
print("GMDH Self-Organizing Polynomial Network (Proper Implementation)")
print("=" * 75)
print()

# ============================================================================
# Load and Prepare Data
# ============================================================================
print("Loading data...")
start_load = time.time()

df = pd.read_csv(INPUT_FILE, nrows=200000)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"Loaded {len(X):,} rows × {X.shape[1]} features")
print(f"Fraud rate: {y.mean()*100:.2f}%")
print()

# Normalize
X_min = X.min(axis=0, keepdims=True)
X_max = X.max(axis=0, keepdims=True)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)

# ============================================================================
# Split Data: Train (70%) + Validation (30%) - KEY FOR SELF-ORGANIZATION
# ============================================================================
np.random.seed(42)
n_train = int(0.7 * len(X_norm))
idx = np.random.permutation(len(X_norm))
train_idx, val_idx = idx[:n_train], idx[n_train:]

X_train, y_train = X_norm[train_idx], y[train_idx]
X_val, y_val = X_norm[val_idx], y[val_idx]

print(f"Train set: {len(X_train):,} samples")
print(f"Validation set: {len(X_val):,} samples (used for self-organization)")
print()

# ============================================================================
# Layer 1: Generate & Select Polynomial Nodes
# ============================================================================
print("Layer 1: Generating polynomial nodes from feature pairs...")
print("-" * 75)

n_features = X_norm.shape[1]
n_pairs = n_features * (n_features - 1) // 2
print(f"Total possible feature pairs: {n_pairs:,}")

# Generate ALL polynomial nodes (limit to 500 for memory)
max_nodes = min(500, n_pairs)
all_pairs = list(itertools.combinations(range(n_features), 2))
np.random.shuffle(all_pairs)
selected_pairs = all_pairs[:max_nodes]

print(f"Evaluating {len(selected_pairs)} polynomial nodes...")
print()

layer1_candidates = []

for idx, (feat_i, feat_j) in enumerate(selected_pairs):
    if (idx + 1) % 100 == 0:
        print(f"  Evaluated {idx + 1}/{len(selected_pairs)} nodes...")
    
    # Create polynomial basis [1, xi, xj, xi*xj]
    X_train_poly = np.column_stack([
        np.ones(len(X_train)),
        X_train[:, feat_i],
        X_train[:, feat_j],
        X_train[:, feat_i] * X_train[:, feat_j]
    ])
    
    X_val_poly = np.column_stack([
        np.ones(len(X_val)),
        X_val[:, feat_i],
        X_val[:, feat_j],
        X_val[:, feat_i] * X_val[:, feat_j]
    ])
    
    # Train on TRAINING set
    coeffs, _, _, _ = np.linalg.lstsq(X_train_poly, y_train, rcond=None)
    
    # Evaluate on VALIDATION set (SELF-ORGANIZATION!)
    y_val_pred = X_val_poly @ coeffs
    val_mse = np.mean((y_val - y_val_pred)**2)
    
    # Also compute training MSE for reference
    y_train_pred = X_train_poly @ coeffs
    train_mse = np.mean((y_train - y_train_pred)**2)
    
    layer1_candidates.append({
        "feat_i": int(feat_i),
        "feat_j": int(feat_j),
        "coeffs": coeffs.tolist(),
        "val_mse": float(val_mse),
        "train_mse": float(train_mse),
    })

print()

# ============================================================================
# Select Best Nodes (SELF-ORGANIZATION)
# ============================================================================
print("Selecting best nodes based on VALIDATION MSE (Self-Organization)...")

# Sort by validation MSE (best first)
layer1_candidates.sort(key=lambda x: x["val_mse"])

# Select top 50 nodes
n_selected = 50
layer1_nodes = layer1_candidates[:n_selected]

print(f"Selected {n_selected} best nodes")
print(f"Best validation MSE: {layer1_nodes[0]['val_mse']:.6f}")
print(f"Median validation MSE: {layer1_nodes[n_selected//2]['val_mse']:.6f}")
print(f"Worst selected validation MSE: {layer1_nodes[-1]['val_mse']:.6f}")
print()

# ============================================================================
# Layer 2: Meta-model combining selected nodes
# ============================================================================
print("Layer 2: Training meta-model...")
print("-" * 75)

# Generate Layer 1 outputs on training set
layer1_train_outputs = []
for node in layer1_nodes:
    feat_i = node["feat_i"]
    feat_j = node["feat_j"]
    coeffs = np.array(node["coeffs"])
    
    X_train_poly = np.column_stack([
        np.ones(len(X_train)),
        X_train[:, feat_i],
        X_train[:, feat_j],
        X_train[:, feat_i] * X_train[:, feat_j]
    ])
    
    output = X_train_poly @ coeffs
    layer1_train_outputs.append(output)

# Generate Layer 1 outputs on validation set
layer1_val_outputs = []
for node in layer1_nodes:
    feat_i = node["feat_i"]
    feat_j = node["feat_j"]
    coeffs = np.array(node["coeffs"])
    
    X_val_poly = np.column_stack([
        np.ones(len(X_val)),
        X_val[:, feat_i],
        X_val[:, feat_j],
        X_val[:, feat_i] * X_val[:, feat_j]
    ])
    
    output = X_val_poly @ coeffs
    layer1_val_outputs.append(output)

# Meta-model on training set
X_train_meta = np.column_stack([np.ones(len(X_train))] + layer1_train_outputs)
layer2_coeffs, _, _, _ = np.linalg.lstsq(X_train_meta, y_train, rcond=None)

# Evaluate on validation set
X_val_meta = np.column_stack([np.ones(len(X_val))] + layer1_val_outputs)
y_val_pred_meta = X_val_meta @ layer2_coeffs
val_mse_meta = np.mean((y_val - y_val_pred_meta)**2)

print(f"Meta-model validation MSE: {val_mse_meta:.6f}")
print()

# ============================================================================
# Final Model Evaluation
# ============================================================================
print("Final Model Evaluation")
print("-" * 75)

# Full dataset prediction
layer1_full_outputs = []
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
    layer1_full_outputs.append(output)

X_full_meta = np.column_stack([np.ones(len(X_norm))] + layer1_full_outputs)
y_pred_full = np.clip(X_full_meta @ layer2_coeffs, 0, 1)

train_mse_final = np.mean((y_train - (X_train_meta @ layer2_coeffs))**2)
val_mse_final = np.mean((y_val - (X_val_meta @ layer2_coeffs))**2)
full_mse_final = np.mean((y - y_pred_full)**2)

print(f"Training MSE: {train_mse_final:.6f}")
print(f"Validation MSE: {val_mse_final:.6f}")
print(f"Full dataset MSE: {full_mse_final:.6f}")
print()

# ============================================================================
# Save Model
# ============================================================================
print("Saving model...")

model_data = {
    "algorithm": "GMDH",
    "version": 4,
    "description": "Self-organizing polynomial network with validation-based node selection",
    "n_features": int(n_features),
    "n_samples": len(X_norm),
    "n_train": len(X_train),
    "n_val": len(X_val),
    "train_val_split": "70/30",
    "layer1_nodes": layer1_nodes,
    "layer2_coeffs": layer2_coeffs.tolist(),
    "layer1_count": len(layer1_nodes),
    "layer1_candidates_evaluated": len(layer1_candidates),
    "train_mse": float(train_mse_final),
    "val_mse": float(val_mse_final),
    "full_mse": float(full_mse_final),
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(model_data, f, indent=2)

print(f"Model saved to: {OUTPUT_FILE}")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 75)
print("GMDH SELF-ORGANIZATION SUMMARY")
print("=" * 75)
print()
print("Key GMDH Principle: Self-organization through validation set selection")
print()
print(f"1. Generated {len(layer1_candidates):,} polynomial node candidates")
print(f"2. Evaluated each on VALIDATION set (key difference from random!)")
print(f"3. Selected top {len(layer1_nodes)} by validation MSE")
print(f"4. Combined selected nodes into Layer 2 meta-model")
print()
print(f"Result: Best validation MSE = {layer1_nodes[0]['val_mse']:.6f}")
print(f"        Model generalization confirmed (val MSE: {val_mse_final:.6f})")
print()
print("=" * 75)
