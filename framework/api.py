"""
framework/api.py
~~~~~~~~~~~~~~~~
OpenAI-compatible API proxy.

Wraps the openai SDK so the rest of the framework stays endpoint-agnostic.
Supports any OpenAI-compatible backend: Ollama, vLLM, OpenRouter, OpenAI, etc.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Iterator

from openai import OpenAI, APIError, APITimeoutError, APIConnectionError

logger = logging.getLogger(__name__)


class AgentAPI:
    """
    Thin wrapper around the openai SDK client.

    Parameters
    ----------
    base_url : str
        OpenAI-compatible endpoint base URL (e.g. http://localhost:11434/v1).
    api_key : str
        API key for the endpoint. Use any non-empty string for local models.
    model : str
        Default model to use when not overridden per call.
    temperature : float
        Sampling temperature (0.0 = deterministic).
    max_tokens : int
        Maximum tokens per completion.
    timeout : float
        Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        logger.info(
            "AgentAPI initialised — base_url=%s model=%s", base_url, model
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request and return the raw response dict.

        Raises
        ------
        RuntimeError
            Wrapped around any SDK / network errors so callers don't need to
            import openai exceptions.
        """
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            logger.debug("→ chat request | model=%s | %d messages", kwargs["model"], len(messages))
            response = self._client.chat.completions.create(**kwargs)
            logger.debug("← chat response | finish_reason=%s", response.choices[0].finish_reason)
            return response.model_dump()
        except APITimeoutError as exc:
            raise RuntimeError(f"API timeout: {exc}") from exc
        except APIConnectionError as exc:
            raise RuntimeError(f"API connection error: {exc}") from exc
        except APIError as exc:
            raise RuntimeError(f"API error {exc.status_code}: {exc.message}") from exc

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """
        Streaming variant — yields delta content strings as they arrive.
        Useful for live-watching the agent think.
        """
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            logger.debug("→ stream chat | model=%s", kwargs["model"])
            with self._client.chat.completions.stream(**kwargs) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
        except APIError as exc:
            raise RuntimeError(f"Stream API error: {exc}") from exc

    def chat_with_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Stream a chat completion, forward text deltas to a callback, and return
        the fully assembled response dict including tool calls.
        """
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = ""

        try:
            logger.debug("→ chat stream+assemble | model=%s", kwargs["model"])
            stream = self._client.chat.completions.create(**kwargs)
            for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                text_delta = delta.content or ""
                if text_delta:
                    content_parts.append(text_delta)
                    if on_text_delta is not None:
                        on_text_delta(text_delta)

                for tc in delta.tool_calls or []:
                    idx = tc.index if tc.index is not None else len(tool_calls)
                    entry = tool_calls.setdefault(
                        idx,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.type:
                        entry["type"] = tc.type
                    if tc.function:
                        if tc.function.name:
                            entry["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            entry["function"]["arguments"] += tc.function.arguments

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            message: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(content_parts),
            }
            if tool_calls:
                message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]

            return {
                "choices": [
                    {
                        "message": message,
                        "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
                    }
                ]
            }
        except APITimeoutError as exc:
            raise RuntimeError(f"API timeout: {exc}") from exc
        except APIConnectionError as exc:
            raise RuntimeError(f"API connection error: {exc}") from exc
        except APIError as exc:
            raise RuntimeError(f"API error {exc.status_code}: {exc.message}") from exc

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_harness(cls, harness: dict[str, Any], defaults: dict[str, Any]) -> "AgentAPI":
        """
        Build an AgentAPI from a parsed harness YAML dict plus framework defaults.

        ENV variables take precedence over harness YAML values.
        Priority: ENV > harness YAML > framework defaults
        """
        endpoint = harness.get("endpoint", {})
        model_cfg = harness.get("model_defaults", {})

        # Resolve base_url: ENV override → harness YAML → framework default
        env_key_prefix = harness.get("name", "BENCHBOT").upper().replace("-", "_")
        base_url = (
            os.getenv(f"{env_key_prefix}_BASE_URL")
            or os.getenv("BENCHBOT_BASE_URL")
            or endpoint.get("base_url", "http://localhost:11434/v1")
        )

        # Resolve api_key from the env var name declared in harness YAML
        api_key_env = endpoint.get("api_key_env", "BENCHBOT_API_KEY")
        api_key = os.getenv(api_key_env) or os.getenv("BENCHBOT_API_KEY") or "benchbot"

        # BENCHBOT_MODEL is set by the boot screen and acts as a global override
        model = (
            os.getenv("BENCHBOT_MODEL")
            or model_cfg.get("model")
            or defaults.get("model", "llama3")
        )

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=model_cfg.get("temperature", defaults.get("temperature", 0.2)),
            max_tokens=model_cfg.get("max_tokens", defaults.get("max_tokens", 4096)),
            timeout=defaults.get("timeout_s", 60.0),
        )
