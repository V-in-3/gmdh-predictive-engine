"""Unified scoring engine factory."""
import os


def get_engine(engine_type: str = None):
    """
    Factory for scoring engines.
    Reads SCORING_ENGINE env var or accepts explicit param.
    Values: 'bedrock_mock' (default), 'gmdh', 'ollama'
    """
    engine_type = engine_type or os.environ.get('SCORING_ENGINE', 'bedrock_mock')

    if engine_type == 'ollama':
        from jobs.engines.ollama_engine import OllamaEngine
        return OllamaEngine()
    elif engine_type == 'gmdh':
        from jobs.engines.gmdh_engine import GmdhEngine
        return GmdhEngine()
    else:
        from jobs.engines.bedrock_engine import BedrockMockEngine
        return BedrockMockEngine()
