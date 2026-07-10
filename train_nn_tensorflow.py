#!/usr/bin/env python3
"""
Neural Network for Fraud Detection — TensorFlow/Keras Implementation
v6: Adam optimizer + Focal Loss + LR scheduling + EarlyStopping

Same architecture as NumPy v5 (512→256→128→64→32) but with:
- Adam optimizer (adaptive learning rate per parameter)
- Binary Focal Cross-Entropy (focuses on hard examples)
- Cosine LR decay (aggressive start, fine-tune at end)
- EarlyStopping (prevents overfitting, saves time)
"""

import json
import os
import time
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF info messages

import tensorflow as tf
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "fraud_transactions_ieee_full.csv"
OUTPUT_MODEL = DATA_DIR / "fraud_model_nn_tf.json"
OUTPUT_METRICS = DATA_DIR / "benchmark_metrics_nn_tf.json"
N_ROWS = 300000

print("=" * 75)
print("Neural Network for Fraud Detection — TensorFlow Implementation")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print("=" * 75)
print()

# ============================================================================
# Load Data
# ============================================================================
print("Loading data...")
start = time.time()

df = pd.read_csv(INPUT_FILE, nrows=N_ROWS)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32)

print(f"Loaded {len(X):,} rows x {X.shape[1]} features")
print(f"Fraud rate: {y.mean()*100:.2f}%")
print()

# Split
np.random.seed(42)
n_train = int(0.7 * len(X))
idx = np.random.permutation(len(X))
train_idx, val_idx = idx[:n_train], idx[n_train:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

print(f"Train: {len(X_train):,} | Val: {len(X_val):,}")
print(f"Train fraud: {y_train.mean()*100:.2f}% | Val fraud: {y_val.mean()*100:.2f}%")
print()

# ============================================================================
# Build Model
# ============================================================================
print("Building model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(512, kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(256, kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128, kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64, kernel_initializer='he_normal'),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Dense(32, kernel_initializer='he_normal'),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.summary()
print()

# ============================================================================
# Compile with Focal Loss + Adam
# ============================================================================
fraud_rate = y_train.mean()
class_weight_val = min((1 - fraud_rate) / (fraud_rate + 1e-8), 12.0)

print(f"Class weight for fraud: {class_weight_val:.2f}x")
print(f"Using: Adam + BinaryFocalCrossentropy(gamma=2.0)")
print()

# Cosine decay LR
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=n_train // 256 * 100,  # total steps
    alpha=0.0001  # minimum LR
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False),
    metrics=[
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
)

# ============================================================================
# Train
# ============================================================================
print("Training...")
print("-" * 75)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=15,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=256,
    validation_data=(X_val, y_val),
    class_weight={0: 1.0, 1: class_weight_val},
    callbacks=callbacks,
    verbose=1
)

train_time = time.time() - start
print()
print(f"Training complete in {train_time:.1f}s")
print()

# ============================================================================
# Evaluate
# ============================================================================
print("Evaluating on validation set...")
print("-" * 75)

y_val_pred = model.predict(X_val, verbose=0).ravel()

# Find best threshold by F1
best_f1 = 0
best_threshold = 0.5
for t in np.arange(0.1, 0.95, 0.05):
    y_pred_bin = (y_val_pred >= t).astype(int)
    tp = np.sum((y_pred_bin == 1) & (y_val == 1))
    fp = np.sum((y_pred_bin == 1) & (y_val == 0))
    fn = np.sum((y_pred_bin == 0) & (y_val == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# Compute metrics at best threshold
y_pred_bin = (y_val_pred >= best_threshold).astype(int)
tp = int(np.sum((y_pred_bin == 1) & (y_val == 1)))
fp = int(np.sum((y_pred_bin == 1) & (y_val == 0)))
fn = int(np.sum((y_pred_bin == 0) & (y_val == 1)))
tn = int(np.sum((y_pred_bin == 0) & (y_val == 0)))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# AUC
sorted_idx = np.argsort(-y_val_pred)
sorted_y = y_val[sorted_idx]
n_pos = int(np.sum(y_val == 1))
n_neg = int(np.sum(y_val == 0))
tp_cum = np.cumsum(sorted_y == 1)
auc_roc = float(np.sum(tp_cum[sorted_y == 0]) / (n_pos * n_neg)) if (n_pos > 0 and n_neg > 0) else 0.5

print(f"Best threshold: {best_threshold:.2f}")
print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  AUC-ROC:   {auc_roc:.4f}")
print()

# Gate check
print("Benchmark Gate:")
print("-" * 75)
gates = {
    'F1': (f1, 0.45),
    'Precision': (precision, 0.50),
    'Recall': (recall, 0.40),
    'AUC-ROC': (auc_roc, 0.78),
}
all_pass = True
for name, (val, threshold) in gates.items():
    passed = val >= threshold
    status = "PASS" if passed else "FAIL"
    mark = "\u2713" if passed else "\u2717"
    print(f"  {name:12s}: {val:.4f} >= {threshold}? {mark} {status}")
    if not passed:
        all_pass = False

if all_pass:
    print("\u2713\u2713\u2713 GATE PASS \u2713\u2713\u2713")
else:
    print("\u2717\u2717\u2717 GATE FAIL \u2717\u2717\u2717")
print()

# ============================================================================
# Save Results
# ============================================================================
metrics = {
    "model": "Neural Network TensorFlow v6",
    "architecture": "512->256->128->64->32 + BatchNorm + Dropout",
    "optimizer": "Adam + Cosine LR decay",
    "loss": "BinaryFocalCrossentropy(gamma=2.0)",
    "class_weight": float(class_weight_val),
    "best_threshold": float(best_threshold),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "auc_roc": float(auc_roc),
    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    "train_samples": int(n_train),
    "val_samples": int(len(X_val)),
    "epochs_trained": len(history.history['loss']),
    "training_time_sec": float(train_time),
    "gate_pass": all_pass
}

with open(OUTPUT_METRICS, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved: {OUTPUT_METRICS}")

# Save model weights as JSON (compatible with existing inference)
weights_data = {
    "algorithm": "Neural Network (TensorFlow)",
    "version": 6,
    "architecture": {"input_dim": int(X_train.shape[1]), "hidden_dims": [512, 256, 128, 64, 32]},
    "best_threshold": float(best_threshold),
    "performance": {"precision": precision, "recall": recall, "f1": f1, "auc_roc": auc_roc},
    "training_time_sec": float(train_time),
}
with open(OUTPUT_MODEL, 'w') as f:
    json.dump(weights_data, f, indent=2)
print(f"Model metadata saved: {OUTPUT_MODEL}")

# Also save Keras model
keras_path = DATA_DIR / "fraud_model_nn_tf.keras"
model.save(keras_path)
print(f"Keras model saved: {keras_path}")

print()
print("=" * 75)
print("DONE")
print("=" * 75)
