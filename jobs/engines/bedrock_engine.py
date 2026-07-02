"""Bedrock mock engine — migrated from bedrock_extractor.py."""
import hashlib
from jobs.engines.base import ScoringEngine


class BedrockMockEngine(ScoringEngine):

    @property
    def engine_name(self) -> str:
        return "bedrock_mock"

    def extract_features(self, transactions: list) -> list:
        results = []
        for tx in transactions:
            desc = tx.get("desc", "")
            hash_val = int(hashlib.md5(desc.encode()).hexdigest()[:8], 16)
            semantic_risk = round((hash_val % 100) / 100.0, 2)
            results.append({
                "order_id": tx.get("order_id", "unknown"),
                "semantic_risk": semantic_risk,
                "engine": self.engine_name,
            })
        return results

    def score_transactions(self, transactions: list, model_path: str = None) -> list:
        """Deterministic scoring baseline — hash-based fraud probability."""
        results = []
        for tx in transactions:
            raw = f"{tx.get('semantic_risk', 0)}{tx.get('velocity_1h', 0)}{tx.get('proxy_score', 0)}{tx.get('amount_deviation', 0)}"
            hash_val = int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)
            score = round((hash_val % 100) / 100.0, 2)
            results.append({
                "score": score,
                "decision": "BLOCK" if score > 0.55 else "ALLOW",
                "engine": self.engine_name,
            })
        return results
