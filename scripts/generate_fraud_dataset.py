import pandas as pd
import numpy as np
import json
import os

def generate_fraud_dataset(records=5000):
    rng = np.random.default_rng(seed=99)

    semantic_risk = rng.uniform(0, 1, size=records)
    velocity_1h = rng.integers(1, 50, size=records).astype(float)
    proxy_score = rng.choice([0.0, 0.5, 1.0], size=records, p=[0.7, 0.15, 0.15])
    amount_deviation = rng.uniform(0, 3, size=records)

    fraud_prob = (
        0.4 * semantic_risk
        + 0.05 * (velocity_1h / 50)
        + 0.3 * proxy_score
        + 0.15 * (semantic_risk * amount_deviation)
    )
    noise = rng.normal(0, 0.05, size=records)
    fraud_prob = np.clip(fraud_prob + noise, 0, 1)

    is_fraud = (fraud_prob > 0.55).astype(float)

    df = pd.DataFrame({
        'semantic_risk': semantic_risk,
        'velocity_1h': velocity_1h,
        'proxy_score': proxy_score,
        'amount_deviation': amount_deviation,
        'is_fraud': is_fraud
    })
    return df


if __name__ == '__main__':
    df = generate_fraud_dataset()

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    csv_path = os.path.join(data_dir, 'fraud_transactions.csv')
    df.to_csv(csv_path, index=False)

    # Also save a sample as JSON for Spark json reader
    json_path = os.path.join(data_dir, 'enriched_transaction.json')
    df.head(100).to_json(json_path, orient='records', lines=True)

    print(f"✅ Fraud dataset generated: {len(df)} records → {csv_path}")
    print(f"✅ Sample JSON: 100 records → {json_path}")
