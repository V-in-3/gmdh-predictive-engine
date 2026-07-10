#!/usr/bin/env python
"""Analyze final_fraud_dataset.csv structure"""

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/final_fraud_dataset.csv', nrows=100)

print("=" * 60)
print("DATASET STRUCTURE")
print("=" * 60)
print(f"\nShape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\nData types:")
print(df.dtypes)

print(f"\nFirst row sample:")
print(df.iloc[0])

print(f"\nTarget column (fraud label):")
fraud_cols = [col for col in df.columns if 'fraud' in col.lower() or 'isFraud' in col or 'label' in col.lower()]
print(f"  Possible target columns: {fraud_cols}")
