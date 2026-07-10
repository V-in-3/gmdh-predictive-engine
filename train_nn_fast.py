#!/usr/bin/env python3
"""
Neural Network for Fraud Detection - Simplified Fast Version
Optimized for speed on large datasets
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_FILE = DATA_DIR / "fraud_model_nn_200k_fast.json"
INPUT_FILE = DATA_DIR / "fraud_ieee_200k.csv"

print("=" * 75)
print("Neural Network for Fraud Detection - FAST VERSION")
print("=" * 75)
print()

start_total = time.time()

# ============================================================================
# Load and Prepare Data
# ============================================================================
print("Loading data...")
start = time.time()

# Load in chunks to save memory
chunks = []
for chunk in pd.read_csv(INPUT_FILE, chunksize=50000):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(df):,} rows in {time.time()-start:.1f}s")

X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)

print(f"Data shape: {X.shape}, Target shape: {y.shape}")
print(f"Fraud rate: {y.mean()*100:.2f}%")
print()

# Normalize
X_mean = X.mean(axis=0, keepdims=True)
X_std = X.std(axis=0, keepdims=True) + 1e-8
X_norm = ((X - X_mean) / X_std).astype(np.float32)

# Split: Train (70%) + Validation (30%)
np.random.seed(42)
n_train = int(0.7 * len(X_norm))
idx = np.random.permutation(len(X_norm))
train_idx, val_idx = idx[:n_train], idx[n_train:]

X_train, y_train = X_norm[train_idx], y[train_idx]
X_val, y_val = X_norm[val_idx], y[val_idx]

print(f"Train: {len(X_train):,} (fraud: {y_train.mean()*100:.2f}%)")
print(f"Val: {len(X_val):,} (fraud: {y_val.mean()*100:.2f}%)")
print()

# ============================================================================
# Simplified Neural Network (2 hidden layers instead of 4)
# ============================================================================
print("Training Neural Network (Simplified Architecture)")
print("-" * 75)

class SimplifiedNN:
    def __init__(self, input_dim):
        """Initialize with 2 hidden layers: 128 → 64"""
        self.input_dim = input_dim
        
        # Layer 1: input → 128
        self.w1 = np.random.randn(input_dim, 128).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, 128), dtype=np.float32)
        
        # Layer 2: 128 → 64
        self.w2 = np.random.randn(128, 64).astype(np.float32) * np.sqrt(2.0 / 128)
        self.b2 = np.zeros((1, 64), dtype=np.float32)
        
        # Output: 64 → 1
        self.w3 = np.random.randn(64, 1).astype(np.float32) * np.sqrt(2.0 / 64)
        self.b3 = np.zeros((1, 1), dtype=np.float32)
    
    def forward(self, X, training=False):
        """Forward pass"""
        # Hidden 1: ReLU
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        
        if training and np.random.rand() < 0.3:  # 30% dropout
            self.a1 = self.a1 * (np.random.rand(*self.a1.shape) > 0.3) / 0.7
        
        # Hidden 2: ReLU
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = np.maximum(0, self.z2)
        
        if training and np.random.rand() < 0.3:  # 30% dropout
            self.a2 = self.a2 * (np.random.rand(*self.a2.shape) > 0.3) / 0.7
        
        # Output: Sigmoid
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        output = 1 / (1 + np.exp(-np.clip(self.z3, -500, 500)))
        
        return output
    
    def backward(self, y, learning_rate=0.01, class_weight=30):
        """Backward pass"""
        batch_size = y.shape[0]
        
        # Output error
        dz3 = self.forward(X_train[0:batch_size])  - y
        fraud_weight = (y == 1).astype(np.float32) * class_weight + (1 - (y == 1).astype(np.float32))
        dz3 = dz3 * fraud_weight
        
        # Gradients
        dw3 = np.dot(self.a2.T, dz3) / batch_size
        db3 = np.sum(dz3, axis=0, keepdims=True) / batch_size
        
        # Backprop to layer 2
        da2 = np.dot(dz3, self.w3.T)
        dz2 = da2 * (self.z2 > 0)
        
        dw2 = np.dot(self.a1.T, dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size
        
        # Backprop to layer 1
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * (self.z1 > 0)
        
        dw1 = np.dot(X_train[0:batch_size].T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size
        
        # Update weights
        self.w3 -= learning_rate * dw3
        self.b3 -= learning_rate * db3
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
    
    def train(self, X_train, y_train, epochs=50, batch_size=512, learning_rate=0.001):
        """Train network"""
        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(len(X_train))
            X_sh, y_sh = X_train[idx], y_train[idx]
            
            for b_start in range(0, len(X_sh), batch_size):
                b_end = min(b_start + batch_size, len(X_sh))
                X_b = X_sh[b_start:b_end]
                y_b = y_sh[b_start:b_end]
                
                # Forward
                self.z1 = np.dot(X_b, self.w1) + self.b1
                self.a1 = np.maximum(0, self.z1)
                self.z2 = np.dot(self.a1, self.w2) + self.b2
                self.a2 = np.maximum(0, self.z2)
                self.z3 = np.dot(self.a2, self.w3) + self.b3
                pred = 1 / (1 + np.exp(-np.clip(self.z3, -500, 500)))
                
                # Loss
                loss = -np.mean(y_b * np.log(pred + 1e-8) + (1 - y_b) * np.log(1 - pred + 1e-8))
                
                # Backward (simplified)
                dz3 = pred - y_b
                fraud_w = (y_b == 1).astype(np.float32) * 30 + (1 - (y_b == 1).astype(np.float32))
                dz3 = dz3 * fraud_w
                
                dw3 = np.dot(self.a2.T, dz3) / len(X_b)
                db3 = np.sum(dz3) / len(X_b)
                
                self.w3 -= learning_rate * dw3
                self.b3 -= learning_rate * db3
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:2d}/{epochs}: Loss {loss:.6f}")

# Train
nn = SimplifiedNN(X_train.shape[1])
nn.train(X_train, y_train, epochs=50, batch_size=512, learning_rate=0.001)

print()
print(f"Training completed in {time.time()-start_total:.1f} seconds")
print()

# ============================================================================
# Evaluate
# ============================================================================
print("Evaluating model...")

y_train_pred = nn.forward(X_train, training=False)
y_val_pred = nn.forward(X_val, training=False)

train_loss = -np.mean(y_train * np.log(y_train_pred + 1e-8) + (1 - y_train) * np.log(1 - y_train_pred + 1e-8))
val_loss = -np.mean(y_val * np.log(y_val_pred + 1e-8) + (1 - y_val) * np.log(1 - y_val_pred + 1e-8))

print(f"Train Loss: {train_loss:.6f}")
print(f"Val Loss: {val_loss:.6f}")
print()

# ============================================================================
# Save Model
# ============================================================================
model_data = {
    "algorithm": "Neural Network (Simplified)",
    "framework": "NumPy",
    "version": 1,
    "architecture": {
        "input": 432,
        "hidden1": 128,
        "hidden2": 64,
        "output": 1
    },
    "training": {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "epochs": 50,
        "batch_size": 512,
        "learning_rate": 0.001,
        "class_weight": 30
    },
    "performance": {
        "train_loss": float(train_loss),
        "val_loss": float(val_loss)
    },
    "weights": {
        "w1": nn.w1.tolist(),
        "b1": nn.b1.tolist(),
        "w2": nn.w2.tolist(),
        "b2": nn.b2.tolist(),
        "w3": nn.w3.tolist(),
        "b3": nn.b3.tolist()
    },
    "normalization": {
        "X_mean": X_mean.tolist()[0],
        "X_std": X_std.tolist()[0]
    }
}

with open(MODEL_FILE, "w") as f:
    json.dump(model_data, f)

print(f"Model saved to: {MODEL_FILE}")
print()
print("=" * 75)
