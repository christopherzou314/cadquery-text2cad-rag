"""Small OpenAI-compatible chat client using only the Python standard library."""

from __future__ import annotations

import json
import http.client
import os
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from .prompts import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt, extract_python_code


class LLMError(RuntimeError):
    """Raised when the LLM request fails or returns an unexpected response."""


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
    )


def repair_cadquery_code(
    description: str,
    previous_code: str,
    traceback: str,
    *,
    reference_context: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 300,
    max_retries: int = 3,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """Ask the model to repair CadQuery code using the execution traceback."""
    return _chat_completion_code(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_repair_prompt(
                    description,
                    previous_code,
                    traceback,
                    reference_context,
                ),
            },
        ],
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        progress_callback=progress_callback,
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
        try:
            _report_progress(
                progress_callback,
                f"Opening streaming LLM response (attempt {attempt + 1}/{max_retries + 1})...",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = _read_streaming_content(response, progress_callback)
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in retryable_http_codes or attempt >= max_retries:
                raise LLMError(f"LLM HTTP {exc.code}: {body}") from exc
            error_message = f"LLM HTTP {exc.code}"
        except transient_errors as exc:
            if attempt >= max_retries:
                raise LLMError(
                    f"LLM request failed after {max_retries + 1} attempts: {exc}"
                ) from exc
            error_message = f"{type(exc).__name__}: {exc}"

        delay = 2 ** (attempt + 1)
        _report_progress(
            progress_callback,
            f"API connection failed ({error_message}). Retrying in {delay}s "
            f"[{attempt + 1}/{max_retries}]...",
        )
        time.sleep(delay)

    return extract_python_code(content)


def _read_streaming_content(
    response,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    chunks: list[str] = []
    non_stream_lines: list[str] = []
    received_chars = 0
    next_progress_update = 1000
    stream_completed = False

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

        choices = event.get("choices")
        if not choices and "usage" in event:
            continue
        try:
            choice = choices[0]
        except (IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected streaming event: {event}") from exc

        if choice.get("finish_reason") is not None:
            stream_completed = True

        delta = choice.get("delta") or {}
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
        return "".join(chunks)

    if non_stream_lines:
        return _read_non_stream_fallback("".join(non_stream_lines))

    raise LLMError("LLM streaming response completed without any content.")


def _read_non_stream_fallback(raw_text: str) -> str:
    try:
        data = json.loads(raw_text)
        content = data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected non-stream fallback response: {raw_text[:500]}") from exc
    if not isinstance(content, str):
        raise LLMError(f"Unexpected response content type: {type(content).__name__}")
    return content


def _report_progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback:
        callback(message)
