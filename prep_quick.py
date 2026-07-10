#!/usr/bin/env python
"""Quick prepare of fraud dataset for GMDH - sample version"""

import pandas as pd
import numpy as np

print("Loading IEEE-CIS fraud dataset...")
df = pd.read_csv('data/final_fraud_dataset.csv', nrows=50000)  # First 50K rows

print(f"Dataset shape: {df.shape}")
print(f"Fraud distribution:\n{df['isFraud'].value_counts()}")
print(f"Fraud rate: {df['isFraud'].mean() * 100:.2f}%")

# Separate and normalize
y = df['isFraud']
X = df.drop('isFraud', axis=1)

# Fill NaN with median
for col in X.columns:
    if X[col].dtype in ['float64', 'int64']:
        X[col] = X[col].fillna(X[col].median())

# Min-Max normalize
print(f"\nNormalizing {X.shape[1]} features...")
for col in X.columns:
    col_min = X[col].min()
    col_max = X[col].max()
    if col_max > col_min:
        X[col] = (X[col] - col_min) / (col_max - col_min)
    else:
        X[col] = 0.0

# Combine
X['is_fraud'] = y.values
print(f"\nSaving to data/fraud_transactions_ieee_50k.csv...")
X.to_csv('data/fraud_transactions_ieee_50k.csv', index=False)

print("\n✓ Done!")
print(f"  Rows: {len(X)}")
print(f"  Features: {X.shape[1] - 1}")
print(f"  File: data/fraud_transactions_ieee_50k.csv")
