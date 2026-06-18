"""
GMDH Fraud Trainer (Model A) — Python implementation.

2-layer self-organizing polynomial model for fraud detection.
Inputs: semantic_risk (from Bedrock), velocity_1h, proxy_score, amount_deviation
Output: fraud probability (0-1)

Same algorithm as Scala version but runs natively in Airflow container.
"""
import json
import numpy as np
import os


def train_fraud_model(data_path=None, output_path=None):
    """Train 2-layer GMDH model on fraud dataset."""

    data_path = data_path or '/opt/airflow/project/data/fraud_transactions.csv'
    output_path = output_path or '/opt/airflow/project/data/fraud_model_coeffs.json'

    # Load CSV
    import csv
    with open(data_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n = len(rows)
    x1 = np.array([float(r['semantic_risk']) for r in rows])
    x2 = np.array([float(r['velocity_1h']) / 50.0 for r in rows])
    x3 = np.array([float(r['proxy_score']) for r in rows])
    x4 = np.array([float(r['amount_deviation']) for r in rows])
    y = np.array([float(r['is_fraud']) for r in rows])

    # Train/validation split (70/30)
    split = int(n * 0.7)
    X_all = np.column_stack([x1, x2, x3, x4])
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y[:split], y[split:]

    col_names = ['x1', 'x2', 'x3', 'x4']

    def fit_node(X_tr, y_tr, X_vl, y_vl, i, j):
        """Fit one GMDH node: y = b0 + b1*xi + b2*xj + b3*xi*xj"""
        xi_tr, xj_tr = X_tr[:, i], X_tr[:, j]
        xi_vl, xj_vl = X_vl[:, i], X_vl[:, j]

        # Design matrix: [1, xi, xj, xi*xj]
        A_tr = np.column_stack([np.ones(len(xi_tr)), xi_tr, xj_tr, xi_tr * xj_tr])
        A_vl = np.column_stack([np.ones(len(xi_vl)), xi_vl, xj_vl, xi_vl * xj_vl])

        # Least squares
        coeffs, _, _, _ = np.linalg.lstsq(A_tr, y_tr, rcond=None)

        # RMSE on validation
        pred = A_vl @ coeffs
        rmse = np.sqrt(np.mean((pred - y_vl) ** 2))

        return {
            'name': f'node_{col_names[i]}_{col_names[j]}',
            'rmse': rmse,
            'intercept': coeffs[0],
            'coeffs': coeffs[1:].tolist(),
            'inputs': [col_names[i], col_names[j]],
            'i': i, 'j': j
        }

    # Layer 1: All C(4,2) = 6 pairwise nodes
    nodes = []
    for i in range(4):
        for j in range(i + 1, 4):
            node = fit_node(X_train, y_train, X_val, y_val, i, j)
            nodes.append(node)

    # Selection: top 2 by lowest RMSE
    nodes.sort(key=lambda n: n['rmse'])
    w1, w2 = nodes[0], nodes[1]

    print(f"Layer 1 winners: {w1['name']} (RMSE={w1['rmse']:.4f}), {w2['name']} (RMSE={w2['rmse']:.4f})")

    # Compute z1, z2 for Layer 2
    def compute_z(X, node):
        xi, xj = X[:, node['i']], X[:, node['j']]
        return node['intercept'] + node['coeffs'][0] * xi + node['coeffs'][1] * xj + node['coeffs'][2] * xi * xj

    z1_train = compute_z(X_train, w1)
    z2_train = compute_z(X_train, w2)

    # Layer 2: Master node — y = b0 + b1*z1 + b2*z2 + b3*z1*z2
    A_master = np.column_stack([np.ones(len(z1_train)), z1_train, z2_train, z1_train * z2_train])
    master_coeffs, _, _, _ = np.linalg.lstsq(A_master, y_train, rcond=None)

    # Validate final model
    z1_val = compute_z(X_val, w1)
    z2_val = compute_z(X_val, w2)
    A_val = np.column_stack([np.ones(len(z1_val)), z1_val, z2_val, z1_val * z2_val])
    final_pred = A_val @ master_coeffs
    final_rmse = np.sqrt(np.mean((final_pred - y_val) ** 2))

    print(f"Layer 2 master node RMSE: {final_rmse:.4f}")

    # Export model
    model = {
        'beta0': float(master_coeffs[0]),
        'betas': [float(c) for c in master_coeffs[1:]],
        'layer1': [
            {'node': w1['name'], 'intercept': float(w1['intercept']),
             'coeffs': w1['coeffs'], 'inputs': w1['inputs']},
            {'node': w2['name'], 'intercept': float(w2['intercept']),
             'coeffs': w2['coeffs'], 'inputs': w2['inputs']}
        ],
        'final_rmse': float(final_rmse),
        'training_samples': split
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(model, f, indent=2)

    print(f"Fraud Model A saved to: {output_path}")
    return model


if __name__ == '__main__':
    train_fraud_model(
        data_path='data/fraud_transactions.csv',
        output_path='data/fraud_model_coeffs.json'
    )
