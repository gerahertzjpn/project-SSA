"""Thin Anthropic API wrapper for the regionals LLM layer.

Two call shapes, both deliberately token-frugal:

- ``pick`` — a draft decision. Structured output (JSON schema with the
  available player ids as an enum) so the reply is guaranteed parseable and
  short.
- ``say`` — commentator / flavor text, capped by ``max_tokens`` and prompt
  instructions.

The SDK import is lazy: the whole regionals engine keeps working without the
``anthropic`` package or an API key (draft falls back to auto-picks,
commentary endpoints report the LLM as unavailable).
"""

from __future__ import annotations

import json
import os

DEFAULT_MODEL = "claude-opus-4-8"


class LLMUnavailable(Exception):
    """Raised when the Anthropic SDK or API key is not available."""


class AnthropicLLM:
    """Real Claude-backed client. Construction fails fast with a clear reason."""

    def __init__(self, model: str | None = None):
        try:
            import anthropic  # noqa: PLC0415 - optional dependency
        except ImportError as error:
            raise LLMUnavailable("anthropic package is not installed (pip install anthropic)") from error
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model = model or os.environ.get("REGIONALS_LLM_MODEL", DEFAULT_MODEL)
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def _track(self, response) -> None:
        self.usage["calls"] += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
            self.usage["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)

    def pick(self, system: str, prompt: str, allowed_ids: list[str]) -> dict:
        """Return ``{"player_id": ..., "reason": ...}`` constrained to ``allowed_ids``."""
        schema = {
            "type": "object",
            "properties": {
                "player_id": {"type": "string", "enum": list(allowed_ids)},
                "reason": {"type": "string", "description": "1 короткая фраза, до 15 слов"},
            },
            "required": ["player_id", "reason"],
            "additionalProperties": False,
        }
        response = self._client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        self._track(response)
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)

    def say(self, system: str, prompt: str, max_tokens: int = 400) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        self._track(response)
        return next((b.text for b in response.content if b.type == "text"), "").strip()


def default_llm() -> AnthropicLLM | None:
    """Best-effort real client; ``None`` when unavailable (callers degrade)."""
    try:
        return AnthropicLLM()
    except LLMUnavailable:
        return None


def llm_unavailable_reason() -> str | None:
    try:
        AnthropicLLM()
        return None
    except LLMUnavailable as error:
        return str(error)
