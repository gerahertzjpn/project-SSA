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


def _first_text(response) -> str:
    """The first text block of a response, or a clear error (not a cryptic
    StopIteration) when the model returned no text."""
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise RuntimeError("no text block in API response")
    return text


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
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                      "cache_read_tokens": 0, "cache_creation_tokens": 0}

    @staticmethod
    def _sys(system: str):
        """Wrap the (stable) system prompt as a single cacheable content block.

        LOSSLESS: the model receives the exact same bytes — ``cache_control`` only changes
        billing/latency of the prefix, never the tokens seen or how the model samples, so output
        quality is unaffected. Block-level ``cache_control`` (not the top-level auto form) so it also
        works on Bedrock/Vertex. NOTE: a cache entry is only CREATED once the cached prefix clears the
        per-model minimum (~4096 tokens for Opus/Haiku); a shorter prefix is a silent no-op (no error,
        no saving). See _track() for the cache_read/creation counters that confirm whether it hits."""
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    def _track(self, response) -> None:
        self.usage["calls"] += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
            self.usage["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
            self.usage["cache_read_tokens"] += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            self.usage["cache_creation_tokens"] += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

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
            system=self._sys(system),
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        self._track(response)
        return json.loads(_first_text(response))

    def plan(self, system: str, prompt: str, player_ids: list[str],
             tempo: list[str], win_path: list[str], role: list[str]) -> dict:
        """Return a per-match game plan constrained to the order vocabulary:

        ``{"tempo", "win_path", "roles": {player_id: role}, "rationale"}``.
        """
        schema = {
            "type": "object",
            "properties": {
                "tempo": {"type": "string", "enum": list(tempo)},
                "win_path": {"type": "string", "enum": list(win_path)},
                "roles": {
                    "type": "object",
                    "properties": {pid: {"type": "string", "enum": list(role)} for pid in player_ids},
                    "required": list(player_ids),
                    "additionalProperties": False,
                },
                "rationale": {"type": "string", "description": "1 короткая фраза, до 15 слов"},
            },
            "required": ["tempo", "win_path", "roles", "rationale"],
            "additionalProperties": False,
        }
        response = self._client.messages.create(
            model=self.model,
            max_tokens=300,
            system=self._sys(system),
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        self._track(response)
        return json.loads(_first_text(response))

    def structured(self, system: str, prompt: str, schema: dict,
                   max_tokens: int = 900) -> dict:
        """Generic structured-output call against an arbitrary JSON schema (used by
        the World manager layer for playbooks / market plans / draft picks).

        Floor the budget at 900: a too-small cap truncates the JSON mid-string (Cyrillic
        reasons are token-heavy) -> json.loads raises -> the caller SILENTLY falls back to its
        heuristic. A high cap is free (billed on ACTUAL output), so never starve the call."""
        max_tokens = max(int(max_tokens or 0), 900)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self._sys(system),
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        self._track(response)
        return json.loads(_first_text(response))

    def say(self, system: str, prompt: str, max_tokens: int = 400) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self._sys(system),
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
