"""Base interface for scoring engines."""


class ScoringEngine:
    """Abstract base for all scoring/feature-extraction engines."""

    @property
    def engine_name(self) -> str:
        raise NotImplementedError

    def extract_features(self, transactions: list) -> list:
        """Extract semantic features from raw transactions.
        Returns list of dicts with at least 'order_id' and 'semantic_risk'.
        """
        raise NotImplementedError

    def score_transactions(self, transactions: list, model_path: str = None) -> list:
        """Score transactions for fraud. Returns list of dicts with 'score' and 'decision'."""
        raise NotImplementedError
