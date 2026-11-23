"""LLM chat adapter: OpenAI-compatible chat completions over HTTPModule."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from src.core.modules.http_module import HTTPModule

import src.core.logger as log

# Provider presets supply defaults; endpoint URL and manifest config override them.
PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "requires_api_key": True,
    },
    "openrouter": {
        "model": "openrouter/free",
        "api_key_env": "OPENROUTER_API_KEY",
        "requires_api_key": True,
    },
    "ollama": {
        "model": "llama3.2",
        "api_key_env": None,
        "requires_api_key": False,
    },
    "openai_compatible": {
        "model": "local-model",
        "api_key_env": "LOCAL_LLM_API_KEY",
        "requires_api_key": False,
    },
}


@dataclass
class LLMChatConfig:
    """Resolved configuration for an LLM HTTP endpoint."""

    provider: str = "openai_compatible"
    model: str = "local-model"
    api_key: str | None = None
    api_key_env: str | None = None
    requires_api_key: bool = False
    timeout: float = 60.0
    temperature: float = 0.0
    max_tokens: int | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    options: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None
    image = None

    @classmethod
    def from_endpoint_config(cls, config: dict[str, Any] | None) -> LLMChatConfig:
        """Merge manifest config with provider defaults."""
        raw = dict(config or {})
        provider = raw.pop("provider", "openai_compatible").lower()
        preset = dict(PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["openai_compatible"]))
        preset.update(raw)
        preset["provider"] = provider
        return cls(**{k: v for k, v in preset.items() if k in cls.__dataclass_fields__})


class LLMChatModule:
    """Build chat-completion requests and delegate transport to :class:`HTTPModule`."""

    def __init__(self, chat_url: str, config: dict[str, Any] | None = None):
        self.chat_url = chat_url.rstrip("/")
        self.config = LLMChatConfig.from_endpoint_config(config)
        http_config = dict(config or {})
        http_config.setdefault("timeout", self.config.timeout)
        if self.config.api_key:
            http_config["api_key"] = self.config.api_key
        if self.config.api_key_env:
            http_config["api_key_env"] = self.config.api_key_env
        http_config["headers"] = {
            **self.config.extra_headers,
            **http_config.get("headers", {}),
        }
        self._http = HTTPModule(self.chat_url, http_config)

    def call_sync(self, payload: dict, timeout: float | None = None) -> dict:
        """Run a chat completion.

        Payload keys:
        - ``prompt`` (required unless ``messages`` provided)
        - ``system_prompt`` (optional)
        - ``messages`` (optional OpenAI-style message list)
        - ``model``, ``temperature`` (optional overrides)
        """
        try:
            messages = self._build_messages(payload)
            body = self._build_request_body(payload, messages)
            if self.config.requires_api_key and not self._resolve_api_key():
                raise ValueError(
                    f"API key required for provider '{self.config.provider}' "
                    f"(set config.api_key or env {self.config.api_key_env})"
                )
            import json
            # log.warn("body", json.dumps(body, indent=2, ensure_ascii=False))
            raw = self._http.call_sync(body, timeout=timeout)
            # log.info("raw", json.dumps(raw, indent=2, ensure_ascii=False))
       
            if "error" in raw:
                return raw
            content = self._extract_message_content(raw)
            return {"content": content, "raw": raw}
        except Exception as exc:  # noqa: BLE001
            log.exception("LLM chat call failed for %s: %s", self.chat_url, exc)
            return {"error": str(exc)}

    async def call_async(
        self,
        payload: dict,
        files: dict | None = None,
        callback: Callable[[dict], Any] | None = None,
        timeout: float | None = None,
    ) -> dict:
        """Async wrapper around :meth:`call_sync`."""
        result = await asyncio.to_thread(self.call_sync, payload, timeout)
        if callback:
            callback(result)
        return result

    def _build_messages(self, payload: dict) -> list[dict[str, str]]:
        if "messages" in payload:
            return payload["messages"]

        messages: list[dict[str, str]] = []
        system_prompt = payload.get("system_prompt", "")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        prompt = payload.get("prompt")
        if not prompt:
            raise ValueError("LLM payload requires 'prompt' or 'messages'")
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_request_body(
        self, payload: dict, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        body = {
            "model": payload.get("model", self.config.model),
            "messages": messages,
            "temperature": payload.get("temperature", self.config.temperature),
            "stream": False,
        }
        max_tokens = payload.get("max_tokens", self.config.max_tokens)
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        options = payload.get("options", self.config.options)
        if options is not None:
            body["options"] = options
        extra_body = payload.get("extra_body", self.config.extra_body)
        if extra_body:
            body.update(extra_body)
        return body

    def _resolve_api_key(self) -> str | None:
        if self.config.api_key:
            return self.config.api_key
        if self.config.api_key_env:
            return os.getenv(self.config.api_key_env) or None
        return None

    @staticmethod
    def _extract_message_content(response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not choices:
            raise ValueError("LLM response missing 'choices'")
        message = choices[0].get("message", {})
        content = message.get("content")
        if content is None:
            raise ValueError("LLM response missing message content")
        return content
