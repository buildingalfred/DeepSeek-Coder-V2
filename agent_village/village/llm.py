"""Brains for the villagers.

  ollama[:model]  - a free local model through Ollama (https://ollama.com). Default model:
                    deepseek-coder-v2. Nothing leaves your computer.
  claude[:model]  - Claude through the Anthropic API (needs ANTHROPIC_API_KEY, paid per use).
  none            - no LLM. Villagers fall back to built-in heuristics (random search).
  auto            - Ollama if it is running, else Claude if a key is set, else none.
"""

import os

import requests

DEFAULT_OLLAMA_MODEL = "deepseek-coder-v2"
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, host: str | None = None):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.name = f"ollama:{model}"

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0.7, "num_ctx": 16384},
        }
        if json_mode:
            body["format"] = "json"
        try:
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=600)
        except requests.RequestException as e:
            raise LLMError(f"cannot reach Ollama at {self.host}: {e}") from e
        if r.status_code == 404:
            raise LLMError(f"Ollama has no model '{self.model}'. Run: ollama pull {self.model}")
        if not r.ok:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return r.json()["message"]["content"]

    @staticmethod
    def available(host: str | None = None) -> bool:
        host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not host.startswith("http"):
            host = "http://" + host
        try:
            return requests.get(f"{host}/api/tags", timeout=2).ok
        except requests.RequestException:
            return False


class ClaudeLLM:
    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL, effort: str = "medium"):
        import anthropic  # imported lazily so the package is only needed for this backend

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.name = f"claude:{model}"

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        a = self._anthropic
        try:
            # Server-side fallback re-runs the request on another model if this one declines.
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except a.AuthenticationError as e:
            raise LLMError("Claude API key missing or invalid (set ANTHROPIC_API_KEY)") from e
        except a.RateLimitError as e:
            raise LLMError("Claude API rate limit hit; try again later or use fewer rounds") from e
        except a.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise LLMError(f"cannot reach the Claude API: {e}") from e
        if response.stop_reason == "refusal":
            raise LLMError("Claude declined this request")
        return "".join(b.text for b in response.content if b.type == "text")


def make(spec: str = "auto"):
    """Build a backend from a string like 'ollama:qwen2.5:14b', 'claude', or 'none'."""
    kind, _, model = spec.partition(":")
    kind = kind.lower()
    if kind == "auto":
        if OllamaLLM.available():
            return OllamaLLM()
        if os.environ.get("ANTHROPIC_API_KEY"):
            return ClaudeLLM()
        return None
    if kind == "none":
        return None
    if kind == "ollama":
        return OllamaLLM(model or DEFAULT_OLLAMA_MODEL)
    if kind == "claude":
        return ClaudeLLM(model or DEFAULT_CLAUDE_MODEL)
    raise ValueError(f"unknown --llm '{spec}'. Use auto, none, ollama[:model] or claude[:model]")
