"""Ollama LLM engine — semantic feature extraction and direct fraud scoring."""
import json
import os
import urllib.error
import urllib.request

from jobs.engines.base import ScoringEngine

# DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://gmdh-ollama:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "tinyllama")


class OllamaEngine(ScoringEngine):

    def __init__(self, base_url: str = None, model: str = None):
        self._base_url = base_url or DEFAULT_OLLAMA_URL
        self._model = model or DEFAULT_MODEL

    @property
    def engine_name(self) -> str:
        return "ollama"

    def _call_ollama(self, prompt: str) -> str:
        """Send prompt to Ollama API, return raw text response."""
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read().decode())
                return result.get("response", "")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"Ollama unavailable: {e}")
            return ""

    def _parse_score(self, text: str) -> float:
        """Extract float score from LLM response."""
        for token in text.replace(",", " ").split():
            try:
                val = float(token)
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                continue
        return 0.5  # fallback if LLM returns garbage

    def extract_features(self, transactions: list) -> list:
        results = []
        for tx in transactions:
            desc = tx.get("desc", "")
            prompt = (
                f"Rate the fraud risk of this transaction from 0.0 (safe) to 1.0 (fraud). "
                f"Reply with ONLY a number.\n\nTransaction: {desc}"
            )
            response = self._call_ollama(prompt)
            score = self._parse_score(response)
            results.append({
                "order_id": tx.get("order_id", "unknown"),
                "semantic_risk": score,
                "engine": self.engine_name,
            })
            print(f"   Ollama [{self._model}]: {tx.get('order_id')} -> {score} (raw: {response[:50]})")
        return results

    def score_transactions(self, transactions: list, model_path: str = None) -> list:
        results = []
        for tx in transactions:
            prompt = (
                f"You are a fraud detection system. Score this transaction from 0.0 (legitimate) to 1.0 (fraud). "
                f"Reply with ONLY a number.\n\n"
                f"semantic_risk={tx.get('semantic_risk', 0)}, "
                f"velocity_1h={tx.get('velocity_1h', 0)}, "
                f"proxy_score={tx.get('proxy_score', 0)}, "
                f"amount_deviation={tx.get('amount_deviation', 0)}"
            )
            response = self._call_ollama(prompt)
            score = self._parse_score(response)
            results.append({
                "score": score,
                "decision": "BLOCK" if score > 0.55 else "ALLOW",
                "engine": self.engine_name,
            })
        return results
