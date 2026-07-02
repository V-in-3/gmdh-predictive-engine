"""
Production-like Fraud Dataset Generator (50K records).

Generates a realistic fraud dataset with:
- 97/3 class imbalance (legitimate vs fraud)
- 4 distinct fraud patterns (velocity spike, proxy ring, semantic cluster, combined)
- Temporal drift: last 20% of records shift fraud tactics
- Distributions modeled after IEEE-CIS Fraud Detection dataset characteristics

Output: data/fraud_production_50k.csv
Columns: semantic_risk, velocity_1h, proxy_score, amount_deviation, is_fraud, pattern, epoch
"""
import os
import numpy as np
import pandas as pd


def generate_legitimate(rng, n: int) -> pd.DataFrame:
    """Generate legitimate transactions with realistic distributions."""
    return pd.DataFrame({
        # Most legitimate transactions have low semantic risk (right-skewed)
        'semantic_risk': np.clip(rng.beta(2, 8, size=n), 0, 1),
        # Normal shopping velocity: 1-8 per hour, occasionally higher
        'velocity_1h': np.clip(rng.lognormal(mean=1.2, sigma=0.6, size=n), 1, 50),
        # 90% no proxy, 8% low-risk VPN, 2% suspicious proxy
        'proxy_score': rng.choice([0.0, 0.5, 1.0], size=n, p=[0.90, 0.08, 0.02]),
        # Amount deviation from user's mean: mostly small (< 1 std)
        'amount_deviation': np.clip(rng.exponential(0.4, size=n), 0, 3),
        'is_fraud': np.zeros(n),
    })


def generate_fraud_velocity(rng, n: int) -> pd.DataFrame:
    """Pattern 1: Velocity spike — rapid-fire transactions (card testing)."""
    return pd.DataFrame({
        'semantic_risk': np.clip(rng.beta(3, 5, size=n), 0, 1),  # moderate
        'velocity_1h': np.clip(rng.normal(35, 8, size=n), 15, 50),  # HIGH
        'proxy_score': rng.choice([0.0, 0.5, 1.0], size=n, p=[0.40, 0.30, 0.30]),
        'amount_deviation': np.clip(rng.exponential(0.8, size=n), 0, 3),
        'is_fraud': np.ones(n),
    })


def generate_fraud_proxy(rng, n: int) -> pd.DataFrame:
    """Pattern 2: Proxy ring — high-value orders through anonymous proxies."""
    return pd.DataFrame({
        'semantic_risk': np.clip(rng.beta(4, 4, size=n), 0, 1),  # mid-range
        'velocity_1h': np.clip(rng.normal(8, 4, size=n), 1, 25),  # normal-ish
        'proxy_score': rng.choice([0.0, 0.5, 1.0], size=n, p=[0.05, 0.25, 0.70]),  # HIGH PROXY
        'amount_deviation': np.clip(rng.normal(2.0, 0.5, size=n), 0.5, 3),  # HIGH AMOUNT
        'is_fraud': np.ones(n),
    })


def generate_fraud_semantic(rng, n: int) -> pd.DataFrame:
    """Pattern 3: Semantic cluster — suspicious item descriptions (gift cards, electronics bulk)."""
    return pd.DataFrame({
        'semantic_risk': np.clip(rng.beta(8, 2, size=n), 0.4, 1),  # HIGH SEMANTIC
        'velocity_1h': np.clip(rng.normal(5, 3, size=n), 1, 20),  # normal
        'proxy_score': rng.choice([0.0, 0.5, 1.0], size=n, p=[0.60, 0.25, 0.15]),
        'amount_deviation': np.clip(rng.normal(1.5, 0.6, size=n), 0, 3),
        'is_fraud': np.ones(n),
    })


def generate_fraud_combined(rng, n: int) -> pd.DataFrame:
    """Pattern 4: Combined — all signals fire at once (obvious fraud)."""
    return pd.DataFrame({
        'semantic_risk': np.clip(rng.beta(7, 2, size=n), 0.5, 1),
        'velocity_1h': np.clip(rng.normal(25, 10, size=n), 10, 50),
        'proxy_score': rng.choice([0.0, 0.5, 1.0], size=n, p=[0.10, 0.30, 0.60]),
        'amount_deviation': np.clip(rng.normal(2.2, 0.4, size=n), 1.0, 3),
        'is_fraud': np.ones(n),
    })


def generate_drift_fraud(rng, n: int) -> pd.DataFrame:
    """
    Drifted fraud pattern: fraudsters adapted.
    - Low velocity (learned to slow down)
    - No proxy (using residential IPs)
    - But high semantic risk + amount deviation remain
    This simulates real-world fraud evolution where attackers avoid known signals.
    """
    return pd.DataFrame({
        'semantic_risk': np.clip(rng.beta(6, 3, size=n), 0.3, 1),
        'velocity_1h': np.clip(rng.normal(4, 2, size=n), 1, 12),  # LOW (adapted!)
        'proxy_score': rng.choice([0.0, 0.5, 1.0], size=n, p=[0.85, 0.10, 0.05]),  # LOW (adapted!)
        'amount_deviation': np.clip(rng.normal(1.8, 0.5, size=n), 0.5, 3),  # still high
        'is_fraud': np.ones(n),
    })


def generate_production_dataset(total_records=50000, fraud_rate=0.03, drift_start=0.8, seed=2024):
    """
    Generate production-like dataset with temporal drift.

    Args:
        total_records: Total number of records
        fraud_rate: Fraction of fraud (default 3%)
        drift_start: After this fraction of data, fraud patterns shift
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed=seed)

    n_fraud = int(total_records * fraud_rate)
    n_legit = total_records - n_fraud

    # Split into pre-drift and post-drift epochs
    drift_point = int(total_records * drift_start)
    n_fraud_pre = int(n_fraud * drift_start)
    n_fraud_post = n_fraud - n_fraud_pre
    n_legit_pre = drift_point - n_fraud_pre
    n_legit_post = (total_records - drift_point) - n_fraud_post

    # --- PRE-DRIFT EPOCH (first 80%) ---
    # Distribute fraud across 4 patterns (roughly equal)
    n_per_pattern = n_fraud_pre // 4
    remainder = n_fraud_pre - (n_per_pattern * 4)

    pre_legit = generate_legitimate(rng, n_legit_pre)
    pre_legit['pattern'] = 'legitimate'
    pre_legit['epoch'] = 'stable'

    pre_fraud_vel = generate_fraud_velocity(rng, n_per_pattern)
    pre_fraud_vel['pattern'] = 'velocity_spike'
    pre_fraud_vel['epoch'] = 'stable'

    pre_fraud_proxy = generate_fraud_proxy(rng, n_per_pattern)
    pre_fraud_proxy['pattern'] = 'proxy_ring'
    pre_fraud_proxy['epoch'] = 'stable'

    pre_fraud_sem = generate_fraud_semantic(rng, n_per_pattern)
    pre_fraud_sem['pattern'] = 'semantic_cluster'
    pre_fraud_sem['epoch'] = 'stable'

    pre_fraud_comb = generate_fraud_combined(rng, n_per_pattern + remainder)
    pre_fraud_comb['pattern'] = 'combined'
    pre_fraud_comb['epoch'] = 'stable'

    pre_epoch = pd.concat([pre_legit, pre_fraud_vel, pre_fraud_proxy, pre_fraud_sem, pre_fraud_comb])

    # --- POST-DRIFT EPOCH (last 20%) ---
    # Fraud patterns shift: old patterns reduced, new "drift" pattern dominates
    n_old_patterns = n_fraud_post // 3  # some old fraud remains
    n_drift_fraud = n_fraud_post - n_old_patterns

    post_legit = generate_legitimate(rng, n_legit_post)
    post_legit['pattern'] = 'legitimate'
    post_legit['epoch'] = 'drift'

    # Some old-style fraud persists
    post_fraud_old = generate_fraud_combined(rng, n_old_patterns)
    post_fraud_old['pattern'] = 'combined'
    post_fraud_old['epoch'] = 'drift'

    # New adapted fraud (the drift)
    post_fraud_drift = generate_drift_fraud(rng, n_drift_fraud)
    post_fraud_drift['pattern'] = 'adapted'
    post_fraud_drift['epoch'] = 'drift'

    post_epoch = pd.concat([post_legit, post_fraud_old, post_fraud_drift])

    # --- COMBINE & SHUFFLE within each epoch ---
    pre_epoch = pre_epoch.sample(frac=1, random_state=seed).reset_index(drop=True)
    post_epoch = post_epoch.sample(frac=1, random_state=seed + 1).reset_index(drop=True)

    # Concatenate (temporal order: stable first, then drift)
    df = pd.concat([pre_epoch, post_epoch]).reset_index(drop=True)

    # Round for readability
    df['semantic_risk'] = df['semantic_risk'].round(4)
    df['velocity_1h'] = df['velocity_1h'].round(1)
    df['amount_deviation'] = df['amount_deviation'].round(4)
    df['is_fraud'] = df['is_fraud'].astype(int)

    return df


def print_stats(df: pd.DataFrame):
    """Print dataset statistics."""
    print(f"\n{'='*60}")
    print(f"PRODUCTION DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total records: {len(df):,}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(f"")

    # Per-epoch stats
    for epoch in ['stable', 'drift']:
        subset = df[df['epoch'] == epoch]
        fraud_sub = subset[subset['is_fraud'] == 1]
        print(f"[{epoch.upper()}] Records: {len(subset):,} | Fraud: {len(fraud_sub):,} ({fraud_sub.shape[0]/len(subset)*100:.1f}%)")
        if len(fraud_sub) > 0:
            patterns = fraud_sub['pattern'].value_counts()
            for p, cnt in patterns.items():
                print(f"    {p}: {cnt}")

    print(f"\nFeature distributions (fraud vs legit):")
    for col in ['semantic_risk', 'velocity_1h', 'proxy_score', 'amount_deviation']:
        legit_mean = df[df['is_fraud'] == 0][col].mean()
        fraud_mean = df[df['is_fraud'] == 1][col].mean()
        print(f"  {col:20s} | legit={legit_mean:.3f} | fraud={fraud_mean:.3f} | delta={fraud_mean-legit_mean:+.3f}")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    df = generate_production_dataset()
    print_stats(df)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'fraud_production_50k.csv')
    df.to_csv(csv_path, index=False)

    print(f"Dataset saved: {csv_path}")
    print(f"File size: {os.path.getsize(csv_path) / 1024 / 1024:.1f} MB")
