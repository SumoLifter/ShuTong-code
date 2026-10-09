from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI


@dataclass
class LLMResponse:
    content: str
    response_format_mode: Optional[str] = None
    schema_name: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    cached_tokens: int = 0
    response_id: Optional[str] = None


class LLMClient:
    def __init__(self, config_path: Path, provider: str, model: str, temperature: float = 0.3):
        self.config = self._load_config(config_path)
        self.provider = provider
        self.model_name = model
        self.temperature = temperature
        self.model_config, self.provider_config = self._resolve_model(provider, model)
        self.api_key = self._resolve_api_key()
        self.client = self._build_client()
        self.extra_body = self.model_config.get("extra_body")
        self.max_tokens = int(self.model_config.get("max_tokens", 4096))
        self.timeout = self.model_config.get("timeout")
        if self.timeout is not None:
            self.timeout = float(self.timeout)
        self.structured_output_mode = self._resolve_structured_output_mode()
        self.json_object_requires_keyword = bool(self.model_config.get("json_object_requires_keyword", False))

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        if not config_path.exists():
            return {}
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    def _resolve_model(self, provider: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        providers = self.config.get("providers", {})
        provider_config = providers.get(provider)
        if not provider_config:
            raise ValueError(f"unsupported provider: {provider}")
        models = provider_config.get("models", [])
        for item in models:
            if item.get("alias") == model:
                return item, provider_config
        for item in models:
            if item.get("id") == model:
                return item, provider_config
        raise ValueError(f"unsupported model provider={provider}, model={model}")

    def _build_client(self) -> OpenAI:
        api_key = self.api_key or ("dummy" if self.provider == "local" else "")
        return OpenAI(
            api_key=api_key,
            base_url=self.provider_config["base_url"],
        )

    def _resolve_api_key(self) -> str:
        env_names = [self.provider_config.get("api_key_env")] + list(self.provider_config.get("api_key_env_fallbacks", []))
        for env_name in env_names:
            if not env_name:
                continue
            value = os.getenv(str(env_name), "").strip()
            if value and not self._looks_like_placeholder(value):
                return value
        return ""

    def _looks_like_placeholder(self, value: str) -> bool:
        normalized = value.strip().lower()
        placeholders = {"sk-xxx", "xxx", "your_api_key", "your-key", "your_key"}
        return normalized in placeholders or normalized.endswith("xxx")

    def _resolve_structured_output_mode(self) -> str:
        explicit = str(self.model_config.get("structured_output_mode", "")).strip().lower()
        if explicit:
            return explicit
        if bool(self.model_config.get("support_json_mode", False)):
            return "json_object"
        return "none"

    def _stringify_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    text = getattr(item, "text", None)
                    if text is not None:
                        parts.append(str(text))
                    else:
                        parts.append(str(item))
            return "".join(parts)
        return str(content)

    def _build_response_format(
        self,
        schema_definition: Dict[str, Any],
        strict: bool,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not schema_definition:
            return None, None

        schema_name = str(schema_definition.get("name", "structured_output")).strip() or "structured_output"
        schema_body = schema_definition.get("schema", {})
        mode = self.structured_output_mode

        if mode == "none":
            if strict:
                raise ValueError(f"model={self.model_name} does not support native structured output")
            return None, "prompt_json"

        if strict and mode != "json_schema":
            raise ValueError(f"model={self.model_name} requires json_schema for strict structured output")

        if mode == "json_schema":
            return (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema_body,
                        "strict": True if strict else False,
                    },
                },
                mode,
            )

        if mode == "json_object":
            if strict:
                raise ValueError(f"model={self.model_name} does not support strict json_schema output")
            return ({"type": "json_object"}, mode)

        raise ValueError(f"unsupported structured_output_mode: {mode}")

    def _normalize_usage(self, usage: Any) -> Tuple[Dict[str, Any], int]:
        if usage is None:
            return {}, 0
        if hasattr(usage, "model_dump"):
            usage_dict = usage.model_dump()
        elif isinstance(usage, dict):
            usage_dict = dict(usage)
        else:
            usage_dict = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details", "completion_tokens_details"):
                value = getattr(usage, key, None)
                if value is not None:
                    usage_dict[key] = value
        cached_tokens = 0
        prompt_details = usage_dict.get("prompt_tokens_details")
        if hasattr(prompt_details, "model_dump"):
            prompt_details = prompt_details.model_dump()
        if isinstance(prompt_details, dict):
            cached_tokens = int(prompt_details.get("cached_tokens", 0) or 0)
        return usage_dict, cached_tokens

    def _ensure_json_keyword(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        combined = "\n".join(str(item.get("content", "")) for item in messages).lower()
        if "json" in combined:
            return messages
        patched = [dict(item) for item in messages]
        for idx, item in enumerate(patched):
            if str(item.get("role", "")).lower() == "system":
                item["content"] = str(item.get("content", "")).rstrip() + "\n\n请仅返回一个合法的 JSON 对象（不要输出任何额外内容）。"
                patched[idx] = item
                return patched
        patched.append({"role": "system", "content": "请仅返回一个合法的 JSON 对象（不要输出任何额外内容）。"})
        return patched

    def chat(
        self,
        messages: List[Dict[str, str]],
        schema: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> LLMResponse:
        if not self.api_key and self.provider != "local":
            return LLMResponse(
                content=json.dumps({"message": "missing_api_key", "provider": self.provider}, ensure_ascii=False),
                response_format_mode=None,
                schema_name=schema.get("name") if schema else None,
            )

        request_messages = list(messages)
        response_format, response_format_mode = self._build_response_format(schema, strict) if schema else (None, None)
        if response_format_mode == "json_object" and self.json_object_requires_keyword:
            request_messages = self._ensure_json_keyword(request_messages)

        request_kwargs: Dict[str, Any] = {
            "model": self.model_config["id"],
            "messages": request_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.timeout is not None:
            request_kwargs["timeout"] = self.timeout
        if self.extra_body:
            request_kwargs["extra_body"] = self.extra_body
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                message = response.choices[0].message if response.choices else None
                content = self._stringify_content(getattr(message, "content", ""))
                usage, cached_tokens = self._normalize_usage(getattr(response, "usage", None))
                return LLMResponse(
                    content=content,
                    response_format_mode=response_format_mode,
                    schema_name=schema.get("name") if schema else None,
                    usage=usage,
                    cached_tokens=cached_tokens,
                    response_id=getattr(response, "id", None),
                )
            except Exception as exc:
                if attempt == 2:
                    return LLMResponse(
                        content=json.dumps({"message": "llm_error", "detail": str(exc)}, ensure_ascii=False),
                        response_format_mode=response_format_mode,
                        schema_name=schema.get("name") if schema else None,
                    )
                time.sleep(1 + attempt)

        raise AssertionError("unreachable")

    def chat_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        rag_context: str = "",
    ) -> LLMResponse:
        return self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
