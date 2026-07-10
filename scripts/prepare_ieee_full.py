#!/usr/bin/env python
"""Prepare final_fraud_dataset for GMDH training"""

import pandas as pd
import numpy as np
import sys

def prepare_gmdh_dataset(input_path, output_path, sample_size=None):
    """
    Prepare IEEE-CIS fraud dataset for GMDH training
    
    Args:
        input_path: Path to final_fraud_dataset.csv
        output_path: Path to save prepared dataset
        sample_size: Number of rows to sample (None = all)
    """
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    if sample_size:
        print(f"Sampling {sample_size} rows...")
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud distribution:\n{df['isFraud'].value_counts()}")
    
    # Separate features and target
    y = df['isFraud'].copy()
    X = df.drop('isFraud', axis=1)
    
    # Handle missing values
    print(f"\nHandling missing values...")
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
    
    # Normalize features to [0, 1] manually
    print(f"Normalizing {X.shape[1]} features...")
    X_normalized = X.copy()
    for col in X_normalized.columns:
        col_min = X_normalized[col].min()
        col_max = X_normalized[col].max()
        if col_max > col_min:
            X_normalized[col] = (X_normalized[col] - col_min) / (col_max - col_min)
        else:
            X_normalized[col] = 0.0
    
    # Combine back
    df_prepared = X_normalized.copy()
    df_prepared['is_fraud'] = y.values
    
    # Reorder: features first, target last
    cols = list(X.columns) + ['is_fraud']
    df_prepared = df_prepared[cols]
    
    print(f"\nSaving to {output_path}...")
    df_prepared.to_csv(output_path, index=False)
    
    print(f"\n✓ Prepared dataset:")
    print(f"  Shape: {df_prepared.shape}")
    print(f"  Features: {df_prepared.shape[1] - 1}")
    print(f"  Non-fraud: {(y == 0).sum()}")
    print(f"  Fraud: {(y == 1).sum()}")
    print(f"  Fraud rate: {(y == 1).sum() / len(y) * 100:.2f}%")
    
    return df_prepared

if __name__ == '__main__':
    # Use full dataset
    prepare_gmdh_dataset(
        'data/final_fraud_dataset.csv',
        'data/fraud_transactions_ieee_full.csv',
        sample_size=None  # Use all data
    )
    
    print("\n" + "="*60)
    print("READY FOR GMDH TRAINING")
    print("="*60)
    print("\nUse: data/fraud_transactions_ieee_full.csv")
