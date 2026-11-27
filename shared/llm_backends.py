"""
Pluggable LLM backends for agent reasoning nodes.

Lets a reasoning node (e.g. agents/fraud_investigation_agent.py::llm_reason)
switch between a local model (Ollama) and a cloud model (Groq) via the
LLM_BACKEND env var, without the calling code knowing which one is active.

Usage:
    from shared.llm_backends import build_backend, estimate_cost_usd
    backend = build_backend()
    text, usage = backend.complete(prompt)
"""
import json
import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Tuple


class LLMBackend(ABC):
    name: str
    model: str

    @abstractmethod
    def complete(self, prompt: str) -> Tuple[str, dict]:
        """Returns (response_text, usage) where usage has prompt_tokens /
        completion_tokens keys (either may be None if the backend doesn't
        report them)."""


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "tinyllama")
        self.base_url = base_url or os.environ.get("OLLAMA_URL", "http://gmdh-ollama:11434")

    def complete(self, prompt: str) -> Tuple[str, dict]:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
        usage = {
            "prompt_tokens": result.get("prompt_eval_count"),
            "completion_tokens": result.get("eval_count"),
        }
        return result.get("response", ""), usage


# List price, USD per 1K tokens (prompt_in, completion_out). Used only to
# estimate what a call would cost on Groq's paid tier - while the configured
# key is on the free tier, nothing is actually billed. Verify against
# https://groq.com/pricing before trusting this for a real budget decision.
GROQ_PRICING_PER_1K = {
    "openai/gpt-oss-120b": (0.00015, 0.00075),
}
_UNKNOWN_MODEL_PRICE = (0.0, 0.0)


class GroqBackend(LLMBackend):
    name = "groq"

    def __init__(self, model: str = None):
        from groq import Groq

        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def complete(self, prompt: str) -> Tuple[str, dict]:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
        return resp.choices[0].message.content or "", usage


def estimate_cost_usd(backend_name: str, model: str, prompt_tokens, completion_tokens) -> float:
    """List-price estimate, not an actual charge - informational only."""
    if backend_name != "groq" or prompt_tokens is None or completion_tokens is None:
        return 0.0
    price_in, price_out = GROQ_PRICING_PER_1K.get(model, _UNKNOWN_MODEL_PRICE)
    return round(prompt_tokens / 1000 * price_in + completion_tokens / 1000 * price_out, 6)


def build_backend() -> LLMBackend:
    name = os.environ.get("LLM_BACKEND", "ollama").lower()
    if name == "ollama":
        return OllamaBackend()
    if name == "groq":
        return GroqBackend()
    raise ValueError(f"unknown LLM_BACKEND: {name} (expected 'ollama' or 'groq')")
