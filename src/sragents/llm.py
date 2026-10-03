"""Minimal OpenAI-compatible LLM wrapper.

Works with vLLM (OpenAI-compatible API), OpenAI, or any compatible endpoint.
Config via CLI args (--model, --api-base) and env vars (OPENAI_API_BASE, OPENAI_API_KEY).
"""

import os
import re
from contextvars import ContextVar

from openai import OpenAI


def create_llm_client(
    api_base: str | None = None, api_key: str | None = None
) -> OpenAI:
    """Create OpenAI-compatible client (works with vLLM, OpenAI, etc.)."""
    base_url = api_base or os.environ.get("OPENAI_API_BASE")
    key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    return OpenAI(base_url=base_url, api_key=key)


_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)
_USAGE_EVENTS: ContextVar[list[dict]] = ContextVar("sragents_usage_events", default=[])


def strip_think_tags(text: str | None) -> str:
    """Remove <think>...</think> blocks from an LLM response.

    Handles both well-formed `<think>...</think>answer` and truncation
    cases where generation is cut mid-thinking (unclosed tag).
    """
    if text is None:
        return ""
    if "<think>" not in text:
        return text
    text = _THINK_CLOSED_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.lstrip()


def get_extra_body(model: str, thinking: bool = False) -> dict | None:
    """Return per-model extra_body for thinking/reasoning control.

    By default (thinking=False) suppresses thinking on hybrid-thinking
    models so they're comparable to non-reasoning baselines. GPT-5 is a
    pure reasoning model; we always run it at minimal effort.

      - Qwen3: chat_template_kwargs.enable_thinking
      - GLM-5 / Kimi: enable_thinking
      - GPT-5: reasoning_effort="none" (always)
      - Others (Llama, Mistral, MiniMax, ...): no flag
    """
    basename = model.lower().rsplit("/", 1)[-1]

    if "qwen3" in basename:
        return {"enable_thinking": thinking}
    if "gpt-5" in basename:
        return {"reasoning_effort": "none"}
    if "glm-5" in basename or "kimi" in basename:
        return {"enable_thinking": thinking}
    return None


def _uses_completion_token_limit(model: str) -> bool:
    basename = model.lower().rsplit("/", 1)[-1]
    return "gpt-5" in basename


def _chat_kwargs(
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> dict:
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if _uses_completion_token_limit(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    return kwargs


def pop_usage_events() -> list[dict]:
    """Return and clear LLM usage events recorded in the current context."""
    events = list(_USAGE_EVENTS.get())
    _USAGE_EVENTS.set([])
    return events


def summarize_usage(events: list[dict]) -> dict:
    """Aggregate provider token usage events."""
    prompt_tokens = sum(int(e.get("prompt_tokens") or 0) for e in events)
    completion_tokens = sum(int(e.get("completion_tokens") or 0) for e in events)
    total_tokens = sum(
        int(e.get("total_tokens") or 0)
        for e in events
        if e.get("total_tokens") is not None
    )
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "calls": len(events),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "events": events,
    }


def _record_usage(response, *, phase: str) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        event = {
            "phase": phase,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "source": "missing",
        }
    else:
        event = {
            "phase": phase,
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            "source": "provider",
        }
    _USAGE_EVENTS.set([*_USAGE_EVENTS.get(), event])


def chat(
    client: OpenAI,
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stop: list[str] | None = None,
    extra_body: dict | None = None,
) -> str:
    """Send chat completion request, return content string."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = _chat_kwargs(model, messages, temperature, max_tokens)
    if stop:
        kwargs["stop"] = stop
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**kwargs)
    _record_usage(response, phase="chat")
    return response.choices[0].message.content


def chat_messages(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stop: list[str] | None = None,
    extra_body: dict | None = None,
) -> str:
    """Send chat completion with an explicit messages list (for multi-turn)."""
    kwargs = _chat_kwargs(model, messages, temperature, max_tokens)
    if stop:
        kwargs["stop"] = stop
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**kwargs)
    _record_usage(response, phase="chat_messages")
    return response.choices[0].message.content
