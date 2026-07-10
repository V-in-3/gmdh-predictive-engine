"""Precision-Recall curve analysis for current NN model."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Load model
with open(DATA_DIR / "fraud_model_nn_200k.json") as f:
    m = json.load(f)
weights = [np.array(w) for w in m["weights"]]
biases  = [np.array(b) for b in m["biases"]]

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def predict(X):
    a = X
    for i, (w, b) in enumerate(zip(weights, biases)):
        z = a @ w + b
        a = relu(z) if i < len(weights) - 1 else sigmoid(z)
    return a.flatten()

# Load test set
df = pd.read_csv(DATA_DIR / "fraud_transactions_ieee_50k.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

probs = predict(X)
print(f"Prob range: {probs.min():.4f} - {probs.max():.4f}  mean={probs.mean():.4f}")
print()
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}  Status")
print("-" * 75)

found_pass = False
for t in np.arange(0.05, 0.99, 0.05):
    pred = (probs >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    if prec >= 0.50 and rec >= 0.40:
        status = "<<< PASS ALL GATES"
        found_pass = True
    elif rec >= 0.40:
        status = "<-- recall OK"
    elif prec >= 0.50:
        status = "<-- precision OK"
    else:
        status = ""
    print(f"{t:>10.2f} {prec:>10.4f} {rec:>10.4f} {f1:>8.4f} {tp:>6} {fp:>6} {fn:>6}  {status}")

print()
if found_pass:
    print("RESULT: A valid threshold EXISTS - both gates can be satisfied!")
else:
    print("RESULT: No threshold satisfies BOTH Precision>=0.50 AND Recall>=0.40.")
    print("Architecture or feature changes are needed.")
