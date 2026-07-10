"""
Generic GMDH Fraud Trainer - works with any number of features.
Reads CSV with arbitrary features and trains 2-layer polynomial model.
"""
import json
import numpy as np
import os
from itertools import combinations


def train_fraud_model_generic(data_path, output_path, max_features_per_layer=50):
    """
    Train generic GMDH model on fraud dataset with many features.
    
    Args:
        data_path: Path to CSV (last column must be target 'is_fraud')
        output_path: Path to save JSON coefficients
        max_features_per_layer: Limit nodes per layer for speed (default 50)
    """
    
    print(f"Loading data from {data_path}...")
    # Load with pandas for simplicity
    import pandas as pd
    df = pd.read_csv(data_path)
    
    # Assume last column is target
    if 'is_fraud' in df.columns:
        y = df['is_fraud'].values
        X = df.drop('is_fraud', axis=1).values
    else:
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values
    
    n_samples, n_features = X.shape
    print(f"Loaded {n_samples} samples, {n_features} features")
    print(f"Target distribution: {np.sum(y)} positives, {len(y) - np.sum(y)} negatives")
    
    # Train/validation split (70/30)
    split = int(n_samples * 0.7)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Layer 1: Generate nodes from feature pairs
    print(f"\nLayer 1: Training nodes from feature pairs...")
    
    layer1_nodes = []
    max_pairs = min(max_features_per_layer, int(np.sqrt(n_features * 2)))  # Limit pairs
    
    # Sample feature pairs
    feature_pairs = list(combinations(range(n_features), 2))
    if len(feature_pairs) > max_pairs:
        np.random.seed(42)
        selected_indices = np.random.choice(len(feature_pairs), max_pairs, replace=False)
        feature_pairs = [feature_pairs[i] for i in selected_indices]
    
    print(f"Training {len(feature_pairs)} nodes...")
    
    for i, (feat_i, feat_j) in enumerate(feature_pairs):
        if (i + 1) % max(1, len(feature_pairs) // 10) == 0:
            print(f"  {i+1}/{len(feature_pairs)}...")
        
        xi_tr, xj_tr = X_train[:, feat_i], X_train[:, feat_j]
        xi_val, xj_val = X_val[:, feat_i], X_val[:, feat_j]
        
        # Node: y = b0 + b1*xi + b2*xj + b3*xi*xj
        A_tr = np.column_stack([np.ones(len(xi_tr)), xi_tr, xj_tr, xi_tr * xj_tr])
        A_val = np.column_stack([np.ones(len(xi_val)), xi_val, xj_val, xi_val * xj_val])
        
        try:
            # Fit on training
            coeffs = np.linalg.lstsq(A_tr, y_train, rcond=None)[0]
            
            # Score on validation
            y_pred_val = A_val @ coeffs
            y_pred_val = np.clip(y_pred_val, 0, 1)  # Clip to [0,1]
            mse = np.mean((y_pred_val - y_val) ** 2)
            
            layer1_nodes.append({
                'features': [int(feat_i), int(feat_j)],
                'coeffs': coeffs.tolist(),
                'val_mse': float(mse)
            })
        except:
            continue
    
    print(f"Trained {len(layer1_nodes)} nodes in layer 1")
    
    # Layer 2: Meta-model using layer 1 outputs
    print(f"\nLayer 2: Training meta-model from layer 1 outputs...")
    
    # Generate layer 1 outputs
    def get_layer1_outputs(X, nodes):
        outputs = []
        for node in nodes:
            i, j = node['features']
            xi, xj = X[:, i], X[:, j]
            y_node = np.column_stack([np.ones(len(xi)), xi, xj, xi * xj]) @ np.array(node['coeffs'])
            outputs.append(np.clip(y_node, 0, 1))
        return np.column_stack(outputs) if outputs else np.ones((len(X), 1))
    
    X_layer1_train = get_layer1_outputs(X_train, layer1_nodes)
    X_layer1_val = get_layer1_outputs(X_val, layer1_nodes)
    
    # Final model: weighted combination
    A_final = np.column_stack([np.ones(len(X_layer1_train)), X_layer1_train])
    final_coeffs = np.linalg.lstsq(A_final, y_train, rcond=None)[0]
    
    y_pred_final = A_final @ final_coeffs
    y_pred_final = np.clip(y_pred_final, 0, 1)
    final_mse = np.mean((y_pred_final - y_train) ** 2)
    
    print(f"Final model MSE: {final_mse:.6f}")
    
    # Save model
    model = {
        'algorithm': 'GMDH',
        'version': 2,
        'n_features': n_features,
        'layer1_nodes': layer1_nodes[:20],  # Keep top 20 for inference
        'layer2_coeffs': final_coeffs.tolist(),
        'train_mse': float(final_mse),
        'n_layer1_nodes': len(layer1_nodes)
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(model, f, indent=2)
    
    print(f"\nModel saved to {output_path}")
    print(f"  Algorithm: GMDH (2-layer)")
    print(f"  Input features: {n_features}")
    print(f"  Layer 1 nodes: {len(layer1_nodes)}")
    print(f"  Layer 1 used in final: {len(layer1_nodes[:20])}")
    print(f"  Training MSE: {final_mse:.6f}")
    
    return model


if __name__ == '__main__':
    import sys
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/fraud_transactions_ieee_50k.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'data/fraud_model_coeffs.json'
    
    train_fraud_model_generic(data_path, output_path)
