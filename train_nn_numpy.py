#!/usr/bin/env python3
"""
Neural Network for Fraud Detection - Pure NumPy Implementation
v5: Batch Normalization + deeper architecture (512->256->128->64->32)
Uses ReLU activations, BatchNorm, dropout, and class weighting.

Architecture:
Input (432) -> Dense(512)+BN+ReLU+Dropout -> Dense(256)+BN+ReLU+Dropout
           -> Dense(128)+BN+ReLU+Dropout -> Dense(64)+BN+ReLU -> Dense(32)+ReLU -> Dense(1)+Sigmoid
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "fraud_model_nn_200k.json"
INPUT_FILE = DATA_DIR / "fraud_transactions_ieee_full.csv"
N_ROWS = 300000  # train on 300K normalized samples

print("=" * 75)
print("Neural Network for Fraud Detection - NumPy Implementation")
print("=" * 75)
print()

# ============================================================================
# Load and Prepare Data
# ============================================================================

# ============================================================================
# Load and Prepare Data
# ============================================================================
print("Loading data...")
start = time.time()

df = pd.read_csv(INPUT_FILE, nrows=N_ROWS)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

print(f"Loaded {len(X):,} rows × {X.shape[1]} features")
print(f"Fraud rate: {y.mean()*100:.2f}%")
print()

# Data is already normalized [0,1] — no re-normalization needed
# Store identity transform so inference code stays compatible
X_mean = np.zeros((1, X.shape[1]))
X_std = np.ones((1, X.shape[1]))

np.random.seed(42)
n_train = int(0.7 * len(X))
idx = np.random.permutation(len(X))
train_idx, val_idx = idx[:n_train], idx[n_train:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_norm = X

print(f"Train set: {len(X_train):,} samples")
print(f"Validation set: {len(X_val):,} samples")
print(f"Train fraud rate: {y_train.mean()*100:.2f}%")
print(f"Val fraud rate: {y_val.mean()*100:.2f}%")
print()

# ============================================================================
# Neural Network Architecture
# ============================================================================
class NeuralNetworkFraud:
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64, 32], dropout_rate=0.2):
        """Initialize NN layers with Batch Normalization."""
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Initialize weights (Xavier initialization)
        self.weights = []
        self.biases = []
        
        # Batch Normalization parameters: gamma, beta, running_mean, running_var
        self.bn_gamma = []
        self.bn_beta = []
        self.bn_running_mean = []
        self.bn_running_var = []
        
        dims = [input_dim] + hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros((1, dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            # BN params for all hidden layers (not output)
            if i < len(dims) - 2:
                self.bn_gamma.append(np.ones((1, dims[i+1])))
                self.bn_beta.append(np.zeros((1, dims[i+1])))
                self.bn_running_mean.append(np.zeros((1, dims[i+1])))
                self.bn_running_var.append(np.ones((1, dims[i+1])))
        
        self.activations = []
        self.z_values = []
        self.bn_cache = []  # cache for backprop: (x_norm, var, x_centered)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def batch_norm_forward(self, x, gamma, beta, running_mean, running_var, training, eps=1e-8, momentum=0.1):
        """Batch normalization forward pass."""
        if training:
            mu = x.mean(axis=0, keepdims=True)
            var = x.var(axis=0, keepdims=True)
            x_centered = x - mu
            x_norm = x_centered / np.sqrt(var + eps)
            out = gamma * x_norm + beta
            # Update running stats
            running_mean[:] = (1 - momentum) * running_mean + momentum * mu
            running_var[:]  = (1 - momentum) * running_var  + momentum * var
            cache = (x_norm, var + eps, x_centered, gamma)
        else:
            x_norm = (x - running_mean) / np.sqrt(running_var + eps)
            out = gamma * x_norm + beta
            cache = None
        return out, cache
    
    def batch_norm_backward(self, dout, cache):
        """Batch norm backward pass."""
        x_norm, var_eps, x_centered, gamma = cache
        N = dout.shape[0]
        dgamma = (dout * x_norm).sum(axis=0, keepdims=True)
        dbeta  = dout.sum(axis=0, keepdims=True)
        dx_norm = dout * gamma
        dvar = (-0.5 * dx_norm * x_centered * (var_eps ** -1.5)).sum(axis=0, keepdims=True)
        dmu  = (-dx_norm / np.sqrt(var_eps)).sum(axis=0, keepdims=True) + dvar * (-2 * x_centered).mean(axis=0, keepdims=True)
        dx   = dx_norm / np.sqrt(var_eps) + dvar * 2 * x_centered / N + dmu / N
        return dx, dgamma, dbeta
    
    def forward(self, X, training=False):
        """Forward pass with BatchNorm + ReLU activations."""
        self.activations = [X]
        self.z_values = []
        self.dropout_masks = []
        self.bn_cache = []
        
        # Hidden layers with BN + ReLU
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Batch Normalization
            z_bn, bn_c = self.batch_norm_forward(
                z,
                self.bn_gamma[i], self.bn_beta[i],
                self.bn_running_mean[i], self.bn_running_var[i],
                training
            )
            self.bn_cache.append(bn_c)
            
            a = self.relu(z_bn)
            
            # Dropout during training
            if training:
                mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape) / (1 - self.dropout_rate)
                a = a * mask
                self.dropout_masks.append(mask)
            
            self.activations.append(a)
        
        # Output layer (no BN)
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self.sigmoid(z)
        self.activations.append(output)
        
        return output
    
    def backward(self, y, learning_rate=0.01, class_weight=1.0):
        """Backward pass with BatchNorm gradients."""
        batch_size = y.shape[0]
        
        # Output layer error
        dz = self.activations[-1] - y
        fraud_mask = (y == 1).astype(float)
        class_weights = fraud_mask * class_weight + (1 - fraud_mask) * 1.0
        dz = dz * class_weights
        
        # Backprop through layers
        for i in range(len(self.weights) - 1, -1, -1):
            dw = np.dot(self.activations[i].T, dz) / batch_size
            db = np.sum(dz, axis=0, keepdims=True) / batch_size
            
            if i > 0:
                dz = np.dot(dz, self.weights[i].T)
                # Dropout gradient
                if len(self.dropout_masks) > i - 1:
                    dz = dz * self.dropout_masks[i - 1]
                # ReLU gradient (before BN, approximate through ReLU)
                dz = dz * self.relu_derivative(self.z_values[i - 1])
                # BatchNorm backward
                if self.bn_cache[i - 1] is not None:
                    dz, dgamma, dbeta = self.batch_norm_backward(dz, self.bn_cache[i - 1])
                    self.bn_gamma[i - 1] -= learning_rate * dgamma
                    self.bn_beta[i - 1]  -= learning_rate * dbeta
            
            self.weights[i] -= learning_rate * dw
            self.biases[i]  -= learning_rate * db
    
    def train(self, X, y, epochs=50, batch_size=256, learning_rate=0.001, X_val=None, y_val=None):
        """Train the network."""
        # Class weight for imbalance
        fraud_rate = y.mean()
        class_weight = (1 - fraud_rate) / (fraud_rate + 1e-8)
        class_weight = min(class_weight, 12.0)  # Cap at 12x (v5: BatchNorm architecture)
        
        print(f"Class weight for fraud: {class_weight:.2f}x")
        print()
        
        losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Shuffle training data
            idx = np.random.permutation(len(X))
            X_shuffled, y_shuffled = X[idx], y[idx]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch gradient descent
            for batch_start in range(0, len(X), batch_size):
                batch_end = min(batch_start + batch_size, len(X))
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Forward pass
                y_pred = self.forward(X_batch, training=True)
                
                # Compute loss
                epsilon = 1e-8
                loss = -np.mean(y_batch * np.log(y_pred + epsilon) + 
                               (1 - y_batch) * np.log(1 - y_pred + epsilon))
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass
                self.backward(y_batch, learning_rate=learning_rate, class_weight=class_weight)
            
            epoch_loss /= n_batches
            losses.append(epoch_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val, training=False)
                val_loss = -np.mean(y_val * np.log(y_val_pred + 1e-8) + 
                                   (1 - y_val) * np.log(1 - y_val_pred + 1e-8))
                val_losses.append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}/{epochs}: Train Loss = {epoch_loss:.6f}, Val Loss = {val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}/{epochs}: Train Loss = {epoch_loss:.6f}")
        
        print()
        return losses, val_losses

# ============================================================================
# Train the Network
# ============================================================================
print("Training Neural Network...")
print("-" * 75)

nn = NeuralNetworkFraud(
    input_dim=X_train.shape[1],
    hidden_dims=[512, 256, 128, 64, 32],
    dropout_rate=0.2
)

losses, val_losses = nn.train(
    X_train, y_train,
    epochs=100,
    batch_size=256,
    learning_rate=0.001,
    X_val=X_val,
    y_val=y_val
)

print("Training complete!")
print()

# ============================================================================
# Evaluate on Full Dataset
# ============================================================================
print("Evaluating model...")
print("-" * 75)

y_train_pred = nn.forward(X_train, training=False)
y_val_pred = nn.forward(X_val, training=False)
y_full_pred = nn.forward(X_norm, training=False)

train_loss = -np.mean(y_train * np.log(y_train_pred + 1e-8) + 
                     (1 - y_train) * np.log(1 - y_train_pred + 1e-8))
val_loss = -np.mean(y_val * np.log(y_val_pred + 1e-8) + 
                   (1 - y_val) * np.log(1 - y_val_pred + 1e-8))

# Compute AUC on training set
sorted_indices = np.argsort(-y_train_pred.ravel())
sorted_y_train = y_train.ravel()[sorted_indices]
n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
tp_cumsum = np.cumsum(sorted_y_train == 1)
auc_train = np.sum(tp_cumsum[sorted_y_train == 0]) / (n_pos * n_neg) if (n_pos > 0 and n_neg > 0) else 0.5

print(f"Train Loss: {train_loss:.6f}")
print(f"Val Loss: {val_loss:.6f}")
print(f"Train AUC: {auc_train:.4f}")
print()

# ============================================================================
# Save Model
# ============================================================================
print("Saving model...")

model_data = {
    "algorithm": "Neural Network",
    "framework": "NumPy (Pure Python)",
    "version": 1,
    "description": "Deep neural network v5: BatchNorm + 512->256->128->64->32 architecture",
    "n_features": int(X_train.shape[1]),
    "n_samples": len(X_norm),
    "n_train": len(X_train),
    "n_val": len(X_val),
    "architecture": {
        "input_dim": int(X_train.shape[1]),
        "hidden_dims": [512, 256, 128, 64, 32],
        "output_dim": 1,
        "activation": "BatchNorm + ReLU (hidden), Sigmoid (output)",
        "dropout_rate": 0.2,
        "batch_norm": True
    },
    "training": {
        "epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.001,
        "optimizer": "SGD with class weights",
        "class_weight": float(min((1 - y.mean()) / (y.mean() + 1e-8), 50.0))
    },
    "performance": {
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "train_auc": float(auc_train)
    },
    "weights": [w.tolist() for w in nn.weights],
    "biases":  [b.tolist() for b in nn.biases],
    "bn_gamma": [g.tolist() for g in nn.bn_gamma],
    "bn_beta":  [b.tolist() for b in nn.bn_beta],
    "bn_running_mean": [m.tolist() for m in nn.bn_running_mean],
    "bn_running_var":  [v.tolist() for v in nn.bn_running_var],
    "normalization": {
        "X_mean": X_mean.reshape(-1).tolist(),
        "X_std": X_std.reshape(-1).tolist()
    }
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(model_data, f, indent=2)

print(f"Model saved to: {OUTPUT_FILE}")
print()

# ============================================================================
# Summary
# ============================================================================
total_time = time.time() - start
print("=" * 75)
print("NEURAL NETWORK TRAINING SUMMARY")
print("=" * 75)
print()
print("Architecture:")
print("  Input: 432 features")
print("  Hidden: 256 → 128 → 64 → 32 neurons (ReLU activation)")
print("  Output: 1 neuron (Sigmoid)")
print()
print("Regularization:")
print("  • Dropout 30% (prevent overfitting)")
print("  • Class weights (handle imbalance)")
print("  • Xavier weight initialization")
print()
print(f"Training: {len(X_train):,} samples, {100} epochs")
print(f"Validation: {len(X_val):,} samples")
print()
print(f"Final Metrics:")
print(f"  Train Loss: {train_loss:.6f}")
print(f"  Val Loss: {val_loss:.6f}")
print(f"  Train AUC: {auc_train:.4f}")
print()
print(f"Total training time: {total_time:.1f} seconds")
print()
print("=" * 75)
