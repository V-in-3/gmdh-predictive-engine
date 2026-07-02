"""GMDH polynomial scoring engine."""
import json
import os
from jobs.engines.base import ScoringEngine

DEFAULT_MODEL_PATH = '/opt/airflow/project/data/fraud_model_coeffs.json'


class GmdhEngine(ScoringEngine):

    @property
    def engine_name(self) -> str:
        return "gmdh"

    def extract_features(self, transactions: list) -> list:
        """GMDH doesn't extract features — delegates to Bedrock/Ollama."""
        raise NotImplementedError("GMDH is scoring-only. Use BedrockMockEngine or OllamaEngine for feature extraction.")

    def score_transactions(self, transactions: list, model_path: str = None) -> list:
        model_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, 'r') as f:
            model = json.load(f)

        results = []
        for tx in transactions:
            features = [
                tx.get("semantic_risk", 0),
                tx.get("velocity_1h", 0),
                tx.get("proxy_score", 0),
                tx.get("amount_deviation", 0),
            ]
            score = model['beta0'] + sum(b * x for b, x in zip(model['betas'], features))
            score = max(0.0, min(1.0, score))
            results.append({
                "score": score,
                "decision": "BLOCK" if score > 0.55 else "ALLOW",
                "engine": self.engine_name,
            })
        return results
