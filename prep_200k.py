#!/usr/bin/env python
"""Quick preparation of 200K IEEE samples"""
import pandas as pd
print("Loading...")
df = pd.read_csv('data/final_fraud_dataset.csv', nrows=200000)
print(f"Loaded {len(df)} rows")
X = df.drop('isFraud', axis=1)
X['is_fraud'] = df['isFraud']
X.to_csv('data/fraud_ieee_200k.csv', index=False)
print("Saved: data/fraud_ieee_200k.csv")
