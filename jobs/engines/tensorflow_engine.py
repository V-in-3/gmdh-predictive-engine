"""TensorFlow fraud scoring engine with a fixed 432-feature input contract."""

import os

from jobs.engines.base import ScoringEngine

DEFAULT_MODEL_PATH = '/opt/airflow/project/data/fraud_model_nn_tf.keras'


class TensorFlowEngine(ScoringEngine):

    @property
    def engine_name(self) -> str:
        return "tensorflow_nn"

    def extract_features(self, transactions: list) -> list:
        raise NotImplementedError("TensorFlow NN is scoring-only")

    def score_transactions(self, transactions: list, model_path: str = None) -> list:
        import numpy as np
        import tensorflow as tf

        model_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TensorFlow model not found: {model_path}")

        model = tf.keras.models.load_model(model_path, compile=False)
        input_size = int(model.input_shape[-1])
        features = np.zeros((len(transactions), input_size), dtype=np.float32)
        for index, tx in enumerate(transactions):
            features[index, 0] = tx.get('semantic_risk', 0.0)
            features[index, 1] = tx.get('velocity_1h', 0.0) / 50.0
            features[index, 2] = tx.get('proxy_score', 0.0)
            features[index, 3] = tx.get('amount_deviation', 0.0) / 3.0

        scores = model.predict(features, verbose=0).reshape(-1)
        return [
            {
                'score': float(score),
                'decision': 'BLOCK' if score > 0.55 else 'ALLOW',
                'engine': self.engine_name,
            }
            for score in scores
        ]
