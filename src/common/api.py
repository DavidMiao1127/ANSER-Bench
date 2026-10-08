from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChatConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: int = 120
    max_retries: int = 3
    retry_backoff: float = 1.5

    @classmethod
    def from_env(cls) -> "ChatConfig":
        from src.common.generation import load_env
        load_env()
        return cls(
            base_url=os.environ.get("GENERATION_BASE_URL", ""),
            api_key=os.environ.get("GENERATION_API_KEY", ""),
            model=os.environ.get("GENERATION_MODEL", ""),
            timeout=int(os.environ.get("GENERATION_TIMEOUT", "120")),
            max_retries=int(os.environ.get("GENERATION_MAX_RETRIES", "3")),
            retry_backoff=float(os.environ.get("GENERATION_RETRY_BACKOFF", "1.5")),
        )


@dataclass(frozen=True)
class ChatResult:
    """A chat completion together with the token usage returned by the API."""

    content: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    finish_reason: Optional[str] = None
    reasoning_content: Optional[str] = None


class ChatClient:
    def __init__(self, config: ChatConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.base_url and self.config.api_key and self.config.model)

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError(
                "Missing generation configuration. Set GENERATION_BASE_URL, GENERATION_API_KEY, and GENERATION_MODEL."
            )

        url = self.config.base_url.rstrip("/") + endpoint
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        last_error: Optional[str] = None
        for attempt in range(1, self.config.max_retries + 1):
            req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                    raw = response.read().decode("utf-8")
                    return json.loads(raw)
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}: {exc.reason}"
                if exc.code in (408, 409, 429, 500, 502, 503, 504) and attempt < self.config.max_retries:
                    backoff = self.config.retry_backoff ** attempt
                    print(
                        f"[Generation] retrying attempt {attempt + 1}/{self.config.max_retries} "
                        f"after {type(exc).__name__}: {last_error}; backoff {backoff:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(backoff)
                    continue
                raise RuntimeError(last_error) from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                last_error = str(exc)
                if attempt < self.config.max_retries:
                    backoff = self.config.retry_backoff ** attempt
                    print(
                        f"[Generation] retrying attempt {attempt + 1}/{self.config.max_retries} "
                        f"after {type(exc).__name__}: {last_error}; backoff {backoff:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(backoff)
                    continue
                raise RuntimeError(last_error) from exc

        raise RuntimeError(last_error or "Unknown generation request failure")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        return_usage: bool = False,
        thinking: Optional[bool] = None,
        allow_empty: bool = False,
    ) -> str | ChatResult:
        payload: Dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # Vendor extensions must not be sent to generic compatible endpoints.
        if thinking is not None:
            from src.common.generation import protocol_for
            protocol = protocol_for(payload['model'])
            if protocol == 'deepseek':
                payload['thinking'] = {'type': 'enabled' if thinking else 'disabled'}
            elif protocol == 'qwen':
                payload['enable_thinking'] = thinking

        obj = self._post("/chat/completions", payload)
        choice = obj["choices"][0]
        message = choice.get("message") if isinstance(choice, dict) else None
        message = message if isinstance(message, dict) else {}
        content = message.get("content")
        content = content if isinstance(content, str) else ""
        usage = obj.get("usage")
        usage = usage if isinstance(usage, dict) else {}

        def token_count(name: str) -> Optional[int]:
            value = usage.get(name)
            return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

        result = ChatResult(
            content=content,
            prompt_tokens=token_count("prompt_tokens"),
            completion_tokens=token_count("completion_tokens"),
            total_tokens=token_count("total_tokens"),
            finish_reason=choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
            reasoning_content=message.get("reasoning_content") if isinstance(message.get("reasoning_content"), str) else None,
        )
        if not result.content.strip() and not allow_empty:
            raise RuntimeError("Model response did not contain answer content")
        return result if return_usage else result.content

    def embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        payload = {
            "model": model or os.environ.get("GENERATION_EMBED_MODEL", "text-embedding-3-large"),
            "input": texts,
        }
        obj = self._post("/embeddings", payload)
        data = obj.get("data", [])
        if not isinstance(data, list):
            raise RuntimeError("Invalid embeddings response")
        result: List[List[float]] = []
        for item in data:
            emb = item.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError("Invalid embedding item")
            result.append([float(x) for x in emb])
        return result
