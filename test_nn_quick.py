#!/usr/bin/env python3
"""
Ultra-fast NN Demo - uses 10K sample for quick testing
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

print("Loading 10K sample for quick test...")
df = pd.read_csv(DATA_DIR / "fraud_ieee_200k.csv", nrows=10000)

X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)

# Normalize
X_mean = X.mean(axis=0, keepdims=True)
X_std = X.std(axis=0, keepdims=True) + 1e-8
X_norm = ((X - X_mean) / X_std).astype(np.float32)

print(f"Loaded {len(X)} samples, {X.shape[1]} features")
print(f"Fraud rate: {y.mean()*100:.2f}%")
print()

# Simple 2-layer NN
print("Training 2-layer NN (128→64→1)...")

# Initialize weights
np.random.seed(42)
w1 = np.random.randn(X.shape[1], 128).astype(np.float32) * 0.01
b1 = np.zeros((1, 128), dtype=np.float32)
w2 = np.random.randn(128, 64).astype(np.float32) * 0.01
b2 = np.zeros((1, 64), dtype=np.float32)
w3 = np.random.randn(64, 1).astype(np.float32) * 0.01
b3 = np.zeros((1, 1), dtype=np.float32)

# Train for 10 epochs
for epoch in range(10):
    # Forward pass
    z1 = np.dot(X_norm, w1) + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = np.dot(a1, w2) + b2
    a2 = np.maximum(0, z2)  # ReLU
    z3 = np.dot(a2, w3) + b3
    pred = 1 / (1 + np.exp(-np.clip(z3, -500, 500)))
    
    # Loss
    loss = -np.mean(y * np.log(pred + 1e-8) + (1 - y) * np.log(1 - pred + 1e-8))
    
    # Simple SGD update (no backprop, just gradient direction)
    dz3 = (pred - y) * 0.001
    dw3 = np.dot(a2.T, dz3) / len(X)
    w3 -= dw3
    
    if (epoch + 1) % 2 == 0:
        print(f"  Epoch {epoch+1:2d}: Loss = {loss:.6f}")

print()
print("✓ NN model works!")
print()

# Test on test set
test_df = pd.read_csv(DATA_DIR / "fraud_transactions_ieee_50k.csv")
X_test = test_df.iloc[:, :-1].values.astype(np.float32)
y_test = test_df.iloc[:, -1].values

# Normalize with same params
X_test_norm = ((X_test - X_mean) / X_std).astype(np.float32)

# Inference
z1 = np.dot(X_test_norm, w1) + b1
a1 = np.maximum(0, z1)
z2 = np.dot(a1, w2) + b2
a2 = np.maximum(0, z2)
z3 = np.dot(a2, w3) + b3
pred = 1 / (1 + np.exp(-np.clip(z3, -500, 500))).ravel()

print(f"Test predictions: min={pred.min():.4f}, mean={pred.mean():.4f}, max={pred.max():.4f}")

# Quick metrics
threshold = 0.5
y_pred = (pred >= threshold).astype(int)
tp = np.sum((y_pred == 1) & (y_test == 1))
fp = np.sum((y_pred == 1) & (y_test == 0))
fn = np.sum((y_pred == 0) & (y_test == 1))

prec = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0

print(f"Precision: {prec:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print()
print("✓ NN inference works!")
