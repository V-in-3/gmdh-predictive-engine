#!/usr/bin/env python
"""Train GMDH on full IEEE-CIS dataset (529K rows)"""

import pandas as pd
import numpy as np
import json
import os
import sys
from itertools import combinations

def train_gmdh_full(data_path, output_path, max_nodes=100):
    """Train GMDH on full dataset with sampling for speed"""
    
    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean() * 100:.2f}%")
    
    y = df['isFraud'].values
    X = df.drop('isFraud', axis=1).values
    
    n_samples, n_features = X.shape
    
    # Normalize
    print("Normalizing...")
    for i in range(n_features):
        col_min = X[:, i].min()
        col_max = X[:, i].max()
        if col_max > col_min:
            X[:, i] = (X[:, i] - col_min) / (col_max - col_min)
    
    # Split
    split = int(n_samples * 0.7)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Layer 1 nodes
    print(f"\nLayer 1: Training {max_nodes} nodes from feature pairs...")
    
    layer1_nodes = []
    feature_pairs = list(combinations(range(n_features), 2))
    
    if len(feature_pairs) > max_nodes:
        np.random.seed(42)
        selected = np.random.choice(len(feature_pairs), max_nodes, replace=False)
        feature_pairs = [feature_pairs[i] for i in selected]
    
    for k, (i, j) in enumerate(feature_pairs):
        if (k+1) % max(1, len(feature_pairs) // 5) == 0:
            print(f"  {k+1}/{len(feature_pairs)}...")
        
        xi_tr, xj_tr = X_train[:, i], X_train[:, j]
        xi_val, xj_val = X_val[:, i], X_val[:, j]
        
        A_tr = np.column_stack([np.ones(len(xi_tr)), xi_tr, xj_tr, xi_tr * xj_tr])
        A_val = np.column_stack([np.ones(len(xi_val)), xi_val, xj_val, xi_val * xj_val])
        
        try:
            coeffs = np.linalg.lstsq(A_tr, y_train, rcond=None)[0]
            y_pred_val = np.clip(A_val @ coeffs, 0, 1)
            mse = np.mean((y_pred_val - y_val) ** 2)
            
            layer1_nodes.append({
                'features': [int(i), int(j)],
                'coeffs': coeffs.tolist(),
                'val_mse': float(mse)
            })
        except:
            continue
    
    print(f"Trained {len(layer1_nodes)} nodes")
    
    # Layer 2
    print(f"\nLayer 2: Training meta-model...")
    
    def get_l1_out(X, nodes):
        outputs = []
        for node in nodes:
            i, j = node['features']
            xi, xj = X[:, i], X[:, j]
            y_node = np.column_stack([np.ones(len(xi)), xi, xj, xi*xj]) @ np.array(node['coeffs'])
            outputs.append(np.clip(y_node, 0, 1))
        return np.column_stack(outputs) if outputs else np.ones((len(X), 1))
    
    X_l1_train = get_l1_out(X_train, layer1_nodes)
    X_l1_val = get_l1_out(X_val, layer1_nodes)
    
    A_final = np.column_stack([np.ones(len(X_l1_train)), X_l1_train])
    final_coeffs = np.linalg.lstsq(A_final, y_train, rcond=None)[0]
    
    y_pred = A_final @ final_coeffs
    train_mse = np.mean((np.clip(y_pred, 0, 1) - y_train) ** 2)
    
    # Validation performance
    A_val_final = np.column_stack([np.ones(len(X_l1_val)), X_l1_val])
    y_val_pred = A_val_final @ final_coeffs
    val_mse = np.mean((np.clip(y_val_pred, 0, 1) - y_val) ** 2)
    
    # Save
    model = {
        'algorithm': 'GMDH',
        'version': 3,
        'n_features': n_features,
        'n_samples': n_samples,
        'layer1_nodes': layer1_nodes[:50],
        'layer2_coeffs': final_coeffs.tolist(),
        'train_mse': float(train_mse),
        'val_mse': float(val_mse),
        'n_layer1_nodes': len(layer1_nodes)
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(model, f, indent=2)
    
    print(f"\nModel saved!")
    print(f"  Training samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Train MSE: {train_mse:.6f}")
    print(f"  Val MSE: {val_mse:.6f}")
    
    return model

if __name__ == '__main__':
    train_gmdh_full(
        'data/final_fraud_dataset.csv',
        'data/fraud_model_coeffs_full.json',
        max_nodes=100
    )
