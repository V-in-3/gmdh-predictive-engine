"""
GMDH System Health Trainer (Model B) -- Python implementation.

2-layer self-organizing polynomial model for infrastructure efficiency prediction.
Inputs: api_latency (x1), auth_status (x2), cpu_load (x3)
Output: architecture_efficiency (0-1)

Same algorithm as Scala version (dags_backup/gmdh_predictive_engine_it.py)
but runs natively in Airflow container without Spark.
"""
import json
import numpy as np
import os


def train_health_model(data_path=None, output_path=None, health_output_path=None):
    """Train 2-layer GMDH model on infrastructure dataset."""

    data_path = data_path or '/opt/airflow/project/data/fintech_transactions_raw.csv'
    output_path = output_path or '/opt/airflow/project/data/model_b_coeffs.json'
    health_output_path = health_output_path or '/opt/airflow/project/data/model_b_health.json'

    # Load CSV
    import csv
    with open(data_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get('cs_auth_status', '') != '']

    n = len(rows)
    x1 = np.array([float(r['amazon_api_latency']) / 2000.0 for r in rows])
    x2 = np.array([float(r['cs_auth_status']) for r in rows])
    x3 = np.array([float(r['system_cpu_load']) / 100.0 for r in rows])
    y = np.array([float(r['architecture_efficiency']) for r in rows])

    # Train/validation split (70/30)
    split = int(n * 0.7)
    X_all = np.column_stack([x1, x2, x3])
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y[:split], y[split:]

    col_names = ['x1', 'x2', 'x3']

    def fit_node(X_tr, y_tr, X_vl, y_vl, i, j):
        """Fit one GMDH node: y = b0 + b1*xi + b2*xj + b3*xi*xj"""
        xi_tr, xj_tr = X_tr[:, i], X_tr[:, j]
        xi_vl, xj_vl = X_vl[:, i], X_vl[:, j]

        A_tr = np.column_stack([np.ones(len(xi_tr)), xi_tr, xj_tr, xi_tr * xj_tr])
        A_vl = np.column_stack([np.ones(len(xi_vl)), xi_vl, xj_vl, xi_vl * xj_vl])

        coeffs, _, _, _ = np.linalg.lstsq(A_tr, y_tr, rcond=None)

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

    # Layer 1: All C(3,2) = 3 pairwise nodes
    nodes = []
    for i in range(3):
        for j in range(i + 1, 3):
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

    # Layer 2: Master node
    A_master = np.column_stack([np.ones(len(z1_train)), z1_train, z2_train, z1_train * z2_train])
    master_coeffs, _, _, _ = np.linalg.lstsq(A_master, y_train, rcond=None)

    # Validate final model
    z1_val = compute_z(X_val, w1)
    z2_val = compute_z(X_val, w2)
    A_val = np.column_stack([np.ones(len(z1_val)), z1_val, z2_val, z1_val * z2_val])
    final_pred = A_val @ master_coeffs
    final_rmse = np.sqrt(np.mean((final_pred - y_val) ** 2))

    # Compute current health score (mean efficiency on recent data)
    health_score = float(np.mean(final_pred))

    print(f"Layer 2 master node RMSE: {final_rmse:.4f}")
    print(f"Current system health_score: {health_score:.4f}")

    # Export model coefficients
    model = {
        'layers': [
            {
                'node_z1': {'intercept': float(w1['intercept']), 'coeffs': w1['coeffs']},
                'node_z2': {'intercept': float(w2['intercept']), 'coeffs': w2['coeffs']}
            },
            {
                'master_node': {'intercept': float(master_coeffs[0]), 'coeffs': [float(c) for c in master_coeffs[1:]]}
            }
        ],
        'final_rmse': float(final_rmse),
        'training_samples': split
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(model, f, indent=2)

    # Export health score (consumed by fraud_detection_engine)
    health = {
        'health_score': health_score,
        'rmse': float(final_rmse),
        'status': 'OK' if health_score > 0.75 else ('WARN' if health_score > 0.45 else 'CRITICAL')
    }
    with open(health_output_path, 'w') as f:
        json.dump(health, f, indent=2)

    print(f"Model B saved to: {output_path}")
    print(f"Health score saved to: {health_output_path}")
    return model, health


if __name__ == '__main__':
    train_health_model(
        data_path='data/fintech_transactions_raw.csv',
        output_path='data/model_b_coeffs.json',
        health_output_path='data/model_b_health.json'
    )
