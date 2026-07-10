#!/usr/bin/env python3
"""
GMDH Self-Organizing Polynomial Network - WITH POLYNOMIAL FORM SEARCH

GMDH Self-Organization Principle:
1. For each feature pair (xi, xj)
2. Try DIFFERENT polynomial forms
3. Evaluate EACH form on the validation set
4. Automatically select the BEST form
5. True self-organization - search over functions!
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import time
import itertools

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "fraud_model_coeffs_200k_gmdh_proper.json"
INPUT_FILE = DATA_DIR / "fraud_ieee_200k.csv"

print("=" * 75)
print("GMDH Self-Organizing with Polynomial Form Search")
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

# Split Data: Train (70%) + Validation (30%)
np.random.seed(42)
n_train = int(0.7 * len(X_norm))
idx = np.random.permutation(len(X_norm))
train_idx, val_idx = idx[:n_train], idx[n_train:]

X_train, y_train = X_norm[train_idx], y[train_idx]
X_val, y_val = X_norm[val_idx], y[val_idx]

print(f"Train set: {len(X_train):,} samples")
print(f"Validation set: {len(X_val):,} samples")
print()

# ============================================================================
# Define Polynomial Forms (CORE OF GMDH SELF-ORGANIZATION)
# ============================================================================
def generate_polynomial_bases(xi, xj, form_type):
    """Generate different polynomial forms for feature pair (xi, xj)."""
    n = len(xi)
    ones = np.ones(n)
    
    if form_type == 1:
        # Form 1: Linear (3 params)
        return np.column_stack([ones, xi, xj])
    
    elif form_type == 2:
        # Form 2: With interaction (4 params)
        return np.column_stack([ones, xi, xj, xi*xj])
    
    elif form_type == 3:
        # Form 3: With squares (5 params)
        return np.column_stack([ones, xi, xj, xi**2, xj**2])
    
    elif form_type == 4:
        # Form 4: Full quadratic (6 params)
        return np.column_stack([ones, xi, xj, xi*xj, xi**2, xj**2])
    
    elif form_type == 5:
        # Form 5: Higher order (7 params)
        return np.column_stack([ones, xi, xj, xi*xj, xi**2, xj**2, xi**2 * xj])
    
    elif form_type == 6:
        # Form 6: More interactions (8 params)
        return np.column_stack([ones, xi, xj, xi*xj, xi**2, xj**2, xi**2*xj, xi*xj**2])
    
    else:
        raise ValueError(f"Unknown form type: {form_type}")

POLYNOMIAL_FORMS = {
    1: "[1, xi, xj]",
    2: "[1, xi, xj, xi*xj]",
    3: "[1, xi, xj, xi², xj²]",
    4: "[1, xi, xj, xi*xj, xi², xj²]",
    5: "[1, xi, xj, xi*xj, xi², xj², xi²*xj]",
    6: "[1, xi, xj, xi*xj, xi², xj², xi²*xj, xi*xj²]",
}

# ============================================================================
# Layer 1: Generate & Select Polynomial Nodes
# ============================================================================
print("Layer 1: Generating polynomial nodes with form search...")
print("-" * 75)

n_features = X_norm.shape[1]
n_pairs = n_features * (n_features - 1) // 2
print(f"Total possible feature pairs: {n_pairs:,}")

# Sample feature pairs
max_nodes = min(300, n_pairs)
all_pairs = list(itertools.combinations(range(n_features), 2))
np.random.shuffle(all_pairs)
selected_pairs = all_pairs[:max_nodes]

print(f"Evaluating {len(selected_pairs)} feature pairs × {len(POLYNOMIAL_FORMS)} forms = {len(selected_pairs) * len(POLYNOMIAL_FORMS)} candidates")
print()

layer1_candidates = []

for idx, (feat_i, feat_j) in enumerate(selected_pairs):
    if (idx + 1) % 50 == 0:
        print(f"  Pair {idx + 1}/{len(selected_pairs)}...")
    
    # For each pair, try ALL polynomial forms
    for form_type in POLYNOMIAL_FORMS.keys():
        # Generate polynomial basis
        X_train_poly = generate_polynomial_bases(
            X_train[:, feat_i], 
            X_train[:, feat_j], 
            form_type
        )
        
        X_val_poly = generate_polynomial_bases(
            X_val[:, feat_i],
            X_val[:, feat_j],
            form_type
        )
        
        # Train on TRAINING set
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_train_poly, y_train, rcond=None)
        except:
            continue
        
        # Evaluate on VALIDATION set
        y_val_pred = X_val_poly @ coeffs
        val_mse = np.mean((y_val - y_val_pred)**2)
        
        # Training MSE for reference
        y_train_pred = X_train_poly @ coeffs
        train_mse = np.mean((y_train - y_train_pred)**2)
        
        layer1_candidates.append({
            "feat_i": int(feat_i),
            "feat_j": int(feat_j),
            "form_type": int(form_type),
            "form_name": POLYNOMIAL_FORMS[form_type],
            "coeffs": coeffs.tolist(),
            "val_mse": float(val_mse),
            "train_mse": float(train_mse),
            "n_params": len(coeffs),
        })

print()

# ============================================================================
# Select Best Nodes (SELF-ORGANIZATION WITH FORM SELECTION)
# ============================================================================
print("Selecting best nodes (self-organization with form search)...")

# Sort by validation MSE (best first)
layer1_candidates.sort(key=lambda x: x["val_mse"])

# Select top 50 nodes
n_selected = 50
layer1_nodes = layer1_candidates[:n_selected]

print(f"Selected {n_selected} best nodes from {len(layer1_candidates)} candidates")
print()

# Show statistics
form_counts = {}
for node in layer1_nodes:
    form = node["form_type"]
    form_counts[form] = form_counts.get(form, 0) + 1

print("Forms selected in top 50 nodes:")
for form_type in sorted(form_counts.keys()):
    count = form_counts[form_type]
    pct = 100 * count / n_selected
    print(f"  Form {form_type} {POLYNOMIAL_FORMS[form_type]:<40} : {count:3d} nodes ({pct:5.1f}%)")

print()
print(f"Best validation MSE: {layer1_nodes[0]['val_mse']:.6f}")
print(f"  (Form {layer1_nodes[0]['form_type']}: {layer1_nodes[0]['form_name']})")
print(f"Median validation MSE: {layer1_nodes[n_selected//2]['val_mse']:.6f}")
print()

# ============================================================================
# Layer 2: Meta-model
# ============================================================================
print("Layer 2: Training meta-model...")
print("-" * 75)

# Generate Layer 1 outputs on training set
layer1_train_outputs = []
for node in layer1_nodes:
    feat_i = node["feat_i"]
    feat_j = node["feat_j"]
    form_type = node["form_type"]
    coeffs = np.array(node["coeffs"])
    
    X_train_poly = generate_polynomial_bases(
        X_train[:, feat_i],
        X_train[:, feat_j],
        form_type
    )
    
    output = X_train_poly @ coeffs
    layer1_train_outputs.append(output)

# Generate Layer 1 outputs on validation set
layer1_val_outputs = []
for node in layer1_nodes:
    feat_i = node["feat_i"]
    feat_j = node["feat_j"]
    form_type = node["form_type"]
    coeffs = np.array(node["coeffs"])
    
    X_val_poly = generate_polynomial_bases(
        X_val[:, feat_i],
        X_val[:, feat_j],
        form_type
    )
    
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
# Final Evaluation
# ============================================================================
print("Final Model Evaluation")
print("-" * 75)

# Full dataset prediction
layer1_full_outputs = []
for node in layer1_nodes:
    feat_i = node["feat_i"]
    feat_j = node["feat_j"]
    form_type = node["form_type"]
    coeffs = np.array(node["coeffs"])
    
    X_poly = generate_polynomial_bases(
        X_norm[:, feat_i],
        X_norm[:, feat_j],
        form_type
    )
    
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
    "version": 5,
    "description": "Self-organizing polynomial network with polynomial form search",
    "n_features": int(n_features),
    "n_samples": len(X_norm),
    "n_train": len(X_train),
    "n_val": len(X_val),
    "train_val_split": "70/30",
    "polynomial_forms": {str(k): v for k, v in POLYNOMIAL_FORMS.items()},
    "forms_selected": form_counts,
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
print("GMDH WITH POLYNOMIAL FORM SEARCH - SUMMARY")
print("=" * 75)
print()
print("Key GMDH Principle: Self-organization through polynomial form search")
print()
print(f"1. Evaluated {len(layer1_candidates):,} candidates:")
print(f"   - {len(selected_pairs)} feature pairs")
print(f"   - {len(POLYNOMIAL_FORMS)} polynomial forms per pair")
print()
print(f"2. Polynomial forms tested:")
for form_type in sorted(POLYNOMIAL_FORMS.keys()):
    print(f"   Form {form_type}: {POLYNOMIAL_FORMS[form_type]}")
print()
print(f"3. Best forms selected (from {len(layer1_nodes)} nodes):")
for form_type in sorted(form_counts.keys()):
    count = form_counts[form_type]
    print(f"   Form {form_type}: {count} nodes")
print()
print(f"Result: Best validation MSE = {layer1_nodes[0]['val_mse']:.6f}")
print(f"        (Using form {layer1_nodes[0]['form_type']}: {layer1_nodes[0]['form_name']})")
print()
print("=" * 75)
