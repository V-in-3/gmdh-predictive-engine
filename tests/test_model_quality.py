"""
End-to-end model quality tests for GMDH Fraud Engine.

Tests train the model on production-like data (50K records) and assert
minimum quality thresholds that a production fraud system must meet.

Run: pytest tests/test_model_quality.py -v
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from jobs.gmdh_fraud_trainer import train_fraud_model

DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'fraud_production_50k.csv')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str):
    """Load CSV into numpy arrays, return (X, y, epochs)."""
    import csv
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X = np.array([
        [float(r['semantic_risk']),
         float(r['velocity_1h']) / 50.0,
         float(r['proxy_score']),
         float(r['amount_deviation'])]
        for r in rows
    ])
    y = np.array([int(r['is_fraud']) for r in rows])
    epochs = np.array([r.get('epoch', 'stable') for r in rows])
    return X, y, epochs


def predict_with_model(model: dict, X: np.ndarray) -> np.ndarray:
    """Apply trained GMDH model to produce scores."""
    layer1 = model['layer1']
    # Map input names to column indices
    col_map = {'x1': 0, 'x2': 1, 'x3': 2, 'x4': 3}

    def compute_z(node_info, X):
        i = col_map[node_info['inputs'][0]]
        j = col_map[node_info['inputs'][1]]
        xi, xj = X[:, i], X[:, j]
        coeffs = node_info['coeffs']
        return node_info['intercept'] + coeffs[0] * xi + coeffs[1] * xj + coeffs[2] * xi * xj

    z1 = compute_z(layer1[0], X)
    z2 = compute_z(layer1[1], X)

    # Master node: beta0 + betas[0]*z1 + betas[1]*z2 + betas[2]*z1*z2
    scores = model['beta0'] + model['betas'][0] * z1 + model['betas'][1] * z2 + model['betas'][2] * z1 * z2
    return np.clip(scores, 0, 1)


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.55):
    """Compute classification metrics."""
    y_pred = (scores > threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)

    # AUC-ROC (trapezoidal approximation)
    auc = compute_auc_roc(y_true, scores)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'auc_roc': auc,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def compute_auc_roc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC-ROC using trapezoidal rule (no sklearn dependency)."""
    # Sort by score descending
    desc_order = np.argsort(-scores)
    y_sorted = y_true[desc_order]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp_count = 0
    fp_count = 0

    for label in y_sorted:
        if label == 1:
            tp_count += 1
        else:
            fp_count += 1
        tpr_list.append(tp_count / n_pos)
        fpr_list.append(fp_count / n_neg)

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2

    return auc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model():
    """Train GMDH model on production dataset (once per module)."""
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Production dataset not found: {DATA_PATH}. Run: python scripts/generate_production_dataset.py")

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_path = f.name

    try:
        model = train_fraud_model(data_path=DATA_PATH, output_path=output_path)
        yield model
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


@pytest.fixture(scope="module")
def dataset():
    """Load production dataset."""
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Production dataset not found: {DATA_PATH}")
    return load_dataset(DATA_PATH)


# ---------------------------------------------------------------------------
# Tests: Baseline Model Quality
# ---------------------------------------------------------------------------

class TestModelQualityBaseline:
    """Model must meet minimum quality thresholds on production-like data."""

    def test_auc_roc_minimum(self, trained_model, dataset):
        """AUC-ROC >= 0.78 — model must separate fraud from legitimate better than random."""
        X, y, _ = dataset
        scores = predict_with_model(trained_model, X)
        metrics = compute_metrics(y, scores)

        print(f"\n  AUC-ROC: {metrics['auc_roc']:.4f}")
        assert metrics['auc_roc'] >= 0.78, (
            f"AUC-ROC {metrics['auc_roc']:.4f} below minimum 0.78. "
            f"Model cannot distinguish fraud from legitimate transactions."
        )

    def test_precision_minimum(self, trained_model, dataset):
        """Precision >= 0.50 — at least half of flagged transactions must be actual fraud."""
        X, y, _ = dataset
        scores = predict_with_model(trained_model, X)
        metrics = compute_metrics(y, scores)

        print(f"\n  Precision: {metrics['precision']:.4f} (TP={metrics['tp']}, FP={metrics['fp']})")
        assert metrics['precision'] >= 0.50, (
            f"Precision {metrics['precision']:.4f} below minimum 0.50. "
            f"Too many false positives: {metrics['fp']} legitimate transactions blocked."
        )

    def test_recall_minimum(self, trained_model, dataset):
        """Recall >= 0.40 — must catch at least 40% of actual fraud."""
        X, y, _ = dataset
        scores = predict_with_model(trained_model, X)
        metrics = compute_metrics(y, scores)

        print(f"\n  Recall: {metrics['recall']:.4f} (TP={metrics['tp']}, FN={metrics['fn']})")
        assert metrics['recall'] >= 0.40, (
            f"Recall {metrics['recall']:.4f} below minimum 0.40. "
            f"Missing too much fraud: {metrics['fn']} fraudulent transactions passed through."
        )

    def test_f1_minimum(self, trained_model, dataset):
        """F1 >= 0.45 — balanced precision/recall must be acceptable."""
        X, y, _ = dataset
        scores = predict_with_model(trained_model, X)
        metrics = compute_metrics(y, scores)

        print(f"\n  F1: {metrics['f1']:.4f}")
        assert metrics['f1'] >= 0.45, (
            f"F1 score {metrics['f1']:.4f} below minimum 0.45."
        )

    def test_rmse_convergence(self, trained_model):
        """Training RMSE must converge below 0.35 — model learned the patterns."""
        rmse = trained_model['final_rmse']
        print(f"\n  Final RMSE: {rmse:.4f}")
        assert rmse < 0.35, (
            f"Training RMSE {rmse:.4f} too high. Model failed to converge."
        )


# ---------------------------------------------------------------------------
# Tests: Drift Resilience
# ---------------------------------------------------------------------------

class TestDriftResilience:
    """Model trained on stable epoch must not catastrophically fail on drifted data."""

    def test_drift_auc_not_catastrophic(self, trained_model, dataset):
        """AUC-ROC on drift epoch >= 0.62 — degradation OK, complete failure NOT OK."""
        X, y, epochs = dataset
        drift_mask = epochs == 'drift'
        X_drift = X[drift_mask]
        y_drift = y[drift_mask]

        if len(y_drift) == 0:
            pytest.skip("No drift epoch in dataset")

        scores = predict_with_model(trained_model, X_drift)
        metrics = compute_metrics(y_drift, scores)

        print(f"\n  Drift AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  Drift Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        assert metrics['auc_roc'] >= 0.62, (
            f"Drift AUC-ROC {metrics['auc_roc']:.4f} below 0.62. "
            f"Model catastrophically failed on new fraud patterns."
        )

    def test_drift_recall_degradation_bounded(self, trained_model, dataset):
        """Recall drop between stable and drift epoch must be < 40 percentage points."""
        X, y, epochs = dataset

        stable_mask = epochs == 'stable'
        drift_mask = epochs == 'drift'

        scores_stable = predict_with_model(trained_model, X[stable_mask])
        scores_drift = predict_with_model(trained_model, X[drift_mask])

        metrics_stable = compute_metrics(y[stable_mask], scores_stable)
        metrics_drift = compute_metrics(y[drift_mask], scores_drift)

        recall_drop = metrics_stable['recall'] - metrics_drift['recall']
        print(f"\n  Stable recall: {metrics_stable['recall']:.4f}")
        print(f"  Drift recall:  {metrics_drift['recall']:.4f}")
        print(f"  Drop: {recall_drop:.4f}")

        assert recall_drop < 0.40, (
            f"Recall dropped by {recall_drop:.4f} (>{0.40}). "
            f"Model is too brittle to pattern changes."
        )


# ---------------------------------------------------------------------------
# Tests: Per-Pattern Detection
# ---------------------------------------------------------------------------

class TestPatternDetection:
    """Model must detect each known fraud pattern above chance level."""

    @pytest.fixture(scope="class")
    def pattern_data(self, dataset):
        """Split dataset by fraud pattern."""
        import csv
        with open(DATA_PATH, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        patterns = {}
        for r in rows:
            if int(r['is_fraud']) == 1:
                p = r.get('pattern', 'unknown')
                if p not in patterns:
                    patterns[p] = {'X': [], 'y': []}
                patterns[p]['X'].append([
                    float(r['semantic_risk']),
                    float(r['velocity_1h']) / 50.0,
                    float(r['proxy_score']),
                    float(r['amount_deviation'])
                ])
                patterns[p]['y'].append(1)
        # Convert to numpy
        for p in patterns:
            patterns[p]['X'] = np.array(patterns[p]['X'])
            patterns[p]['y'] = np.array(patterns[p]['y'])
        return patterns

    def test_detects_velocity_spike(self, trained_model, pattern_data):
        """Must detect >= 30% of velocity-based fraud.
        Note: velocity alone is a weak signal for 2-layer GMDH on imbalanced data.
        The model prioritizes multi-signal patterns (semantic+proxy+amount).
        """
        if 'velocity_spike' not in pattern_data:
            pytest.skip("velocity_spike pattern not in dataset")
        data = pattern_data['velocity_spike']
        scores = predict_with_model(trained_model, data['X'])
        detected = np.mean(scores > 0.55)
        print(f"\n  Velocity spike detection rate: {detected:.2%}")
        assert detected >= 0.30, f"Only detected {detected:.2%} of velocity fraud"

    def test_detects_proxy_ring(self, trained_model, pattern_data):
        """Must detect >= 30% of proxy-based fraud.
        Note: proxy alone is moderate signal; GMDH node selection may deprioritize
        it if semantic_risk + amount_deviation pair scores better on RMSE.
        """
        if 'proxy_ring' not in pattern_data:
            pytest.skip("proxy_ring pattern not in dataset")
        data = pattern_data['proxy_ring']
        scores = predict_with_model(trained_model, data['X'])
        detected = np.mean(scores > 0.55)
        print(f"\n  Proxy ring detection rate: {detected:.2%}")
        assert detected >= 0.30, f"Only detected {detected:.2%} of proxy fraud"

    def test_detects_semantic_cluster(self, trained_model, pattern_data):
        """Must detect >= 40% of semantic-based fraud."""
        if 'semantic_cluster' not in pattern_data:
            pytest.skip("semantic_cluster pattern not in dataset")
        data = pattern_data['semantic_cluster']
        scores = predict_with_model(trained_model, data['X'])
        detected = np.mean(scores > 0.55)
        print(f"\n  Semantic cluster detection rate: {detected:.2%}")
        assert detected >= 0.40, f"Only detected {detected:.2%} of semantic fraud"

    def test_detects_combined(self, trained_model, pattern_data):
        """Must detect >= 70% of combined-signal fraud (easiest pattern)."""
        if 'combined' not in pattern_data:
            pytest.skip("combined pattern not in dataset")
        data = pattern_data['combined']
        scores = predict_with_model(trained_model, data['X'])
        detected = np.mean(scores > 0.55)
        print(f"\n  Combined fraud detection rate: {detected:.2%}")
        assert detected >= 0.70, f"Only detected {detected:.2%} of combined fraud"


# ---------------------------------------------------------------------------
# Tests: Model Properties
# ---------------------------------------------------------------------------

class TestModelProperties:
    """Structural properties of the trained model."""

    def test_model_has_two_layer1_nodes(self, trained_model):
        """GMDH architecture: exactly 2 winner nodes from Layer 1."""
        assert len(trained_model['layer1']) == 2

    def test_model_coefficients_bounded(self, trained_model):
        """Coefficients should not explode (numerical stability)."""
        all_coeffs = [trained_model['beta0']] + trained_model['betas']
        for node in trained_model['layer1']:
            all_coeffs.append(node['intercept'])
            all_coeffs.extend(node['coeffs'])

        max_abs = max(abs(c) for c in all_coeffs)
        print(f"\n  Max |coefficient|: {max_abs:.4f}")
        assert max_abs < 50.0, (
            f"Coefficient {max_abs:.4f} is too large. Model may be numerically unstable."
        )

    def test_model_size_small(self, trained_model):
        """Model JSON must be < 2KB (portable, hot-reloadable)."""
        model_json = json.dumps(trained_model)
        size_bytes = len(model_json.encode('utf-8'))
        print(f"\n  Model size: {size_bytes} bytes")
        assert size_bytes < 2048, f"Model size {size_bytes} bytes exceeds 2KB limit"
