import pandas as pd
import numpy as np
import uuid

def generate_it_dataset(records=10000):
    rng = np.random.default_rng(seed=42)

    customer_ids = [str(uuid.uuid4()) for _ in range(3000)]

    data = {
        'order_id': [f"AMZ-{str(uuid.uuid4())[:8].upper()}" for _ in range(records)],
        'customer_id': rng.choice(customer_ids, size=records),
        'amazon_api_latency': rng.uniform(100, 2000, size=records),
        # 1.0 - Success, 0.5 - Review, 0.0 - Decline
        'cs_auth_status': rng.choice([1.0, 0.5, 0.0], size=records, p=[0.8, 0.1, 0.1]),
        # System Load (CPU %) - x3
        'system_cpu_load': rng.uniform(5, 100, size=records)
    }

    df = pd.DataFrame(data)

    x1 = df['amazon_api_latency'] / 2000
    x2 = df['cs_auth_status']
    x3 = df['system_cpu_load'] / 100

    base_efficiency = (0.6 * (1 - x1)) + (0.3 * x2) - (0.4 * (x1 * x3))

    efficiency = base_efficiency - (0.1 * x3**2)

    noise = rng.normal(loc=0.0, scale=0.03, size=records)
    efficiency += noise

    df['architecture_efficiency'] = np.clip(efficiency, 0, 1)

    null_indices = rng.choice(df.index, size=int(records * 0.02), replace=False)
    df.loc[null_indices, 'cs_auth_status'] = np.nan

    return df

df_final = generate_it_dataset()
df_final.to_csv('fintech_transactions_raw.csv', index=False)

print(f"✅ Dataset for IT generated: {len(df_final)} records.")
print("Columns: order_id, customer_id, amazon_api_latency, cs_auth_status, system_cpu_load, architecture_efficiency")
