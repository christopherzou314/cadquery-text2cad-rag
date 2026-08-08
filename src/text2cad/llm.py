"""Small OpenAI-compatible chat client using only the Python standard library."""

from __future__ import annotations

import json
import http.client
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .prompts import (
    CLARIFICATION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_clarification_prompt,
    build_repair_prompt,
    build_user_prompt,
    extract_python_code,
)


class LLMError(RuntimeError):
    """Raised when the LLM request fails or returns an unexpected response."""


@dataclass(frozen=True)
class StreamingResult:
    content: str
    usage: dict[str, Any] | None = None
    response_id: str | None = None
    provider_request_id: str | None = None
    response_model: str | None = None
    finish_reason: str | None = None
    reasoning_characters: int = 0


def generate_cadquery_code(
    description: str,
    *,
    reference_context: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 300,
    max_retries: int = 3,
    progress_callback: Callable[[str], None] | None = None,
    seed: int | None = None,
    send_seed: bool = False,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Call an OpenAI-compatible chat completions API and return Python code."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    if not api_key:
        raise LLMError("Missing OPENAI_API_KEY. Use --mock for an offline local test.")

    return _chat_completion_code(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(description, reference_context),
            },
        ],
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        progress_callback=progress_callback,
        seed=seed,
        send_seed=send_seed,
        metrics_callback=metrics_callback,
    )


def repair_cadquery_code(
    description: str,
    previous_code: str,
    feedback: str,
    *,
    reference_context: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 300,
    max_retries: int = 3,
    progress_callback: Callable[[str], None] | None = None,
    seed: int | None = None,
    send_seed: bool = False,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
    repair_type: str = "execution",
) -> str:
    """Ask the model to repair code using execution or constraint feedback."""
    return _chat_completion_code(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_repair_prompt(
                    description,
                    previous_code,
                    feedback,
                    reference_context,
                    repair_type,
                ),
            },
        ],
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        progress_callback=progress_callback,
        seed=seed,
        send_seed=send_seed,
        metrics_callback=metrics_callback,
    )


def generate_clarification_response(
    description: str,
    *,
    reference_context: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 300,
    max_retries: int = 3,
    progress_callback: Callable[[str], None] | None = None,
    seed: int | None = None,
    send_seed: bool = False,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Request a natural-language clarification without generating CAD code."""
    return _chat_completion_text(
        [
            {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_clarification_prompt(
                    description, reference_context
                ),
            },
        ],
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        progress_callback=progress_callback,
        seed=seed,
        send_seed=send_seed,
        metrics_callback=metrics_callback,
    )


def _chat_completion_code(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 300,
    max_retries: int = 3,
    progress_callback: Callable[[str], None] | None = None,
    seed: int | None = None,
    send_seed: bool = False,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    content = _chat_completion_text(
        messages,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        progress_callback=progress_callback,
        seed=seed,
        send_seed=send_seed,
        metrics_callback=metrics_callback,
    )
    return extract_python_code(content)


def _chat_completion_text(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 300,
    max_retries: int = 3,
    progress_callback: Callable[[str], None] | None = None,
    seed: int | None = None,
    send_seed: bool = False,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    if not api_key:
        raise LLMError("Missing OPENAI_API_KEY. Use --mock for an offline local test.")

    payload = {
        "model": model,
        "temperature": 0.1,
        "stream": True,
        "messages": messages,
    }
    stream_usage_requested = _supports_stream_usage_option(base_url)
    if stream_usage_requested:
        payload["stream_options"] = {"include_usage": True}
    enable_thinking = _optional_bool_env("LLM_ENABLE_THINKING")
    if (
        enable_thinking is None
        and _is_dashscope_url(base_url)
        and model.lower().startswith("qwen3.7")
    ):
        enable_thinking = True
    if enable_thinking is not None and _is_dashscope_url(base_url):
        payload["enable_thinking"] = enable_thinking
    seed_sent = seed is not None and send_seed
    if seed_sent:
        payload["seed"] = seed
    request_body = json.dumps(payload).encode("utf-8")
    retryable_http_codes = {408, 429, 500, 502, 503, 504}
    transient_errors = (
        http.client.RemoteDisconnected,
        http.client.IncompleteRead,
        ConnectionResetError,
        ConnectionAbortedError,
        BrokenPipeError,
        TimeoutError,
        socket.timeout,
        urllib.error.URLError,
    )

    call_id = str(uuid.uuid4())
    started_at_utc = datetime.now(timezone.utc).isoformat()
    call_started = time.monotonic()
    transport_attempts: list[dict[str, Any]] = []
    backoff_seconds = 0
    result: StreamingResult | None = None

    try:
        for attempt in range(max_retries + 1):
            request = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=request_body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            transport_started = time.monotonic()
            try:
                _report_progress(
                    progress_callback,
                    f"Opening streaming LLM response (attempt {attempt + 1}/{max_retries + 1})...",
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    result = _read_streaming_content(response, progress_callback)
                    http_status = getattr(response, "status", 200)
                transport_attempts.append(
                    _transport_attempt(
                        attempt,
                        transport_started,
                        success=True,
                        http_status=http_status,
                    )
                )
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                error_message = f"LLM HTTP {exc.code}"
                transport_attempts.append(
                    _transport_attempt(
                        attempt,
                        transport_started,
                        success=False,
                        http_status=exc.code,
                        error=f"{error_message}: {body[:1000]}",
                    )
                )
                if exc.code not in retryable_http_codes or attempt >= max_retries:
                    raise LLMError(f"LLM HTTP {exc.code}: {body}") from exc
            except transient_errors as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                transport_attempts.append(
                    _transport_attempt(
                        attempt,
                        transport_started,
                        success=False,
                        error=error_message,
                    )
                )
                if attempt >= max_retries:
                    raise LLMError(
                        f"LLM request failed after {max_retries + 1} attempts: {exc}"
                    ) from exc

            delay = 2 ** (attempt + 1)
            backoff_seconds += delay
            _report_progress(
                progress_callback,
                f"API connection failed ({error_message}). Retrying in {delay}s "
                f"[{attempt + 1}/{max_retries}]...",
            )
            time.sleep(delay)
    except Exception as exc:
        _emit_metrics(
            metrics_callback,
            _call_metrics(
                call_id=call_id,
                started_at_utc=started_at_utc,
                elapsed_seconds=time.monotonic() - call_started,
                model=model,
                base_url=base_url,
                seed=seed,
                seed_sent=seed_sent,
                stream_usage_requested=stream_usage_requested,
                enable_thinking=enable_thinking,
                request_body_bytes=len(request_body),
                input_characters=sum(len(message["content"]) for message in messages),
                transport_attempts=transport_attempts,
                backoff_seconds=backoff_seconds,
                result=None,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise

    if result is None:
        raise LLMError("LLM request ended without a response.")
    _emit_metrics(
        metrics_callback,
        _call_metrics(
            call_id=call_id,
            started_at_utc=started_at_utc,
            elapsed_seconds=time.monotonic() - call_started,
            model=model,
            base_url=base_url,
            seed=seed,
            seed_sent=seed_sent,
            stream_usage_requested=stream_usage_requested,
            enable_thinking=enable_thinking,
            request_body_bytes=len(request_body),
            input_characters=sum(len(message["content"]) for message in messages),
            transport_attempts=transport_attempts,
            backoff_seconds=backoff_seconds,
            result=result,
            error=None,
        ),
    )
    return result.content.strip()


def _read_streaming_content(
    response,
    progress_callback: Callable[[str], None] | None = None,
) -> StreamingResult:
    chunks: list[str] = []
    non_stream_lines: list[str] = []
    received_chars = 0
    next_progress_update = 1000
    stream_completed = False
    usage = None
    response_id = None
    provider_request_id = None
    response_model = None
    finish_reason = None
    reasoning_characters = 0

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            non_stream_lines.append(line)
            continue

        event_text = line[5:].strip()
        if event_text == "[DONE]":
            stream_completed = True
            break

        try:
            event = json.loads(event_text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Invalid JSON in streaming response: {event_text[:200]}") from exc

        if "error" in event:
            raise LLMError(f"LLM streaming error: {event['error']}")

        usage = event.get("usage") or usage
        response_id = event.get("id") or response_id
        provider_request_id = event.get("request_id") or provider_request_id
        response_model = event.get("model") or response_model

        choices = event.get("choices")
        if not choices and "usage" in event:
            continue
        try:
            choice = choices[0]
        except (IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected streaming event: {event}") from exc

        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]
            stream_completed = True

        delta = choice.get("delta") or {}
        reasoning_piece = delta.get("reasoning_content")
        if isinstance(reasoning_piece, str):
            reasoning_characters += len(reasoning_piece)
        piece = delta.get("content")
        if piece is None:
            message = choice.get("message") or {}
            piece = message.get("content")
        if not piece:
            continue
        if not isinstance(piece, str):
            raise LLMError(f"Unexpected streamed content type: {type(piece).__name__}")

        chunks.append(piece)
        received_chars += len(piece)
        if received_chars >= next_progress_update:
            _report_progress(
                progress_callback,
                f"Streaming LLM response: {received_chars} characters received...",
            )
            next_progress_update += 1000

    if chunks and not stream_completed:
        raise http.client.IncompleteRead(
            b"",
            expected="the final streaming event",
        )

    if chunks:
        _report_progress(
            progress_callback,
            f"Streaming LLM response complete: {received_chars} characters.",
        )
        return StreamingResult(
            content="".join(chunks),
            usage=usage,
            response_id=response_id,
            provider_request_id=provider_request_id,
            response_model=response_model,
            finish_reason=finish_reason,
            reasoning_characters=reasoning_characters,
        )

    if non_stream_lines:
        return _read_non_stream_fallback("".join(non_stream_lines))

    raise LLMError("LLM streaming response completed without any content.")


def _read_non_stream_fallback(raw_text: str) -> StreamingResult:
    try:
        data = json.loads(raw_text)
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected non-stream fallback response: {raw_text[:500]}") from exc
    if not isinstance(content, str):
        raise LLMError(f"Unexpected response content type: {type(content).__name__}")
    return StreamingResult(
        content=content,
        usage=data.get("usage"),
        response_id=data.get("id"),
        provider_request_id=data.get("request_id"),
        response_model=data.get("model"),
        finish_reason=choice.get("finish_reason"),
        reasoning_characters=len(
            (choice.get("message") or {}).get("reasoning_content") or ""
        ),
    )


def _transport_attempt(
    attempt: int,
    started: float,
    *,
    success: bool,
    http_status: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "transport_attempt": attempt + 1,
        "success": success,
        "http_status": http_status,
        "duration_seconds": round(time.monotonic() - started, 6),
        "error": error,
    }


def _call_metrics(
    *,
    call_id: str,
    started_at_utc: str,
    elapsed_seconds: float,
    model: str,
    base_url: str,
    seed: int | None,
    seed_sent: bool,
    stream_usage_requested: bool,
    enable_thinking: bool | None,
    request_body_bytes: int,
    input_characters: int,
    transport_attempts: list[dict[str, Any]],
    backoff_seconds: int,
    result: StreamingResult | None,
    error: str | None,
) -> dict[str, Any]:
    usage = _normalize_usage(result.usage if result else None)
    estimated_cost, pricing = _estimate_cost(usage)
    return {
        "call_id": call_id,
        "started_at_utc": started_at_utc,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "success": result is not None and error is None,
        "model_requested": model,
        "model_returned": result.response_model if result else None,
        "base_url_host": urllib.parse.urlparse(base_url).netloc,
        "seed_requested": seed,
        "seed_sent_to_provider": seed_sent,
        "stream_usage_requested": stream_usage_requested,
        "enable_thinking_requested": enable_thinking,
        "temperature": 0.1,
        "latency_seconds": round(elapsed_seconds, 6),
        "backoff_seconds": backoff_seconds,
        "transport_attempt_count": len(transport_attempts),
        "transport_retry_count": max(0, len(transport_attempts) - 1),
        "transport_attempts": transport_attempts,
        "request_body_bytes": request_body_bytes,
        "input_characters": input_characters,
        "response_characters": len(result.content) if result else 0,
        "reasoning_characters": result.reasoning_characters if result else 0,
        "response_id": result.response_id if result else None,
        "provider_request_id": result.provider_request_id if result else None,
        "finish_reason": result.finish_reason if result else None,
        "usage": usage,
        "estimated_cost": estimated_cost,
        "pricing": pricing,
        "error": error,
    }


def _normalize_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {
            "available": False,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cached_prompt_tokens": None,
            "reasoning_tokens": None,
            "text_completion_tokens": None,
            "raw": None,
        }
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "available": True,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_prompt_tokens": prompt_details.get("cached_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
        "text_completion_tokens": completion_details.get("text_tokens"),
        "raw": usage,
    }


def _estimate_cost(
    usage: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    input_rate = _optional_float_env("LLM_INPUT_COST_PER_1M_TOKENS")
    output_rate = _optional_float_env("LLM_OUTPUT_COST_PER_1M_TOKENS")
    currency = os.getenv("LLM_COST_CURRENCY", "CNY")
    pricing = {
        "currency": currency,
        "input_cost_per_1m_tokens": input_rate,
        "output_cost_per_1m_tokens": output_rate,
    }
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if None in (input_rate, output_rate, prompt_tokens, completion_tokens):
        return None, pricing
    amount = (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
    return round(amount, 10), pricing


def _optional_float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _optional_bool_env(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _is_dashscope_url(base_url: str) -> bool:
    host = urllib.parse.urlparse(base_url).netloc.lower()
    return "dashscope" in host or host.endswith(".maas.aliyuncs.com")


def _supports_stream_usage_option(base_url: str) -> bool:
    return _is_dashscope_url(base_url)


def _emit_metrics(
    callback: Callable[[dict[str, Any]], None] | None,
    metrics: dict[str, Any],
) -> None:
    if callback:
        callback(metrics)


def _report_progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback:
        callback(message)
