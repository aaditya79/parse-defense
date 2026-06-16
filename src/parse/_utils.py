"""Shared utilities for PARSE pipeline modules."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any


def parse_json_response(content: str) -> dict | list:
    """Parse JSON from LLM response, handling markdown code fence wrapping."""
    if not content:
        raise ValueError("Empty response content")

    content = content.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        content = match.group(1).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = content.find(start_char)
        if start != -1:
            end = content.rfind(end_char)
            if end > start:
                try:
                    return json.loads(content[start:end + 1])
                except json.JSONDecodeError:
                    pass

    raise ValueError(f"Could not parse JSON from response: {content[:200]}")


_RETRY_SUFFIX = (
    "IMPORTANT: Return ONLY valid JSON. No markdown, no code fences, "
    "no explanation. Start your response with { or ["
)


def _log_fallback(step_name: str, error: str) -> None:
    log_path = os.path.join("results", "fallback_log.json")
    os.makedirs("results", exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": step_name,
        "error": error[:300],
    }
    existing: list = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(entry)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)


def call_llm_with_json_retry(
    client: Any,
    model: str,
    messages: list[dict],
    temperature: float,
    step_name: str,
    fallback: dict | list,
) -> tuple[dict | list, bool]:
    """
    Call LLM, attempt JSON parse. On failure, retry once with explicit JSON-only
    instruction appended. If retry also fails, log and return fallback.

    Returns (result, used_fallback: bool). Callers should skip caching when
    used_fallback is True so the step is retried on the next run.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content or ""

    try:
        return parse_json_response(content), False
    except ValueError:
        pass

    # Retry with explicit JSON-only instruction
    retry_messages = messages + [
        {"role": "assistant", "content": content},
        {"role": "user", "content": _RETRY_SUFFIX},
    ]
    try:
        retry_resp = client.chat.completions.create(
            model=model,
            messages=retry_messages,
            temperature=temperature,
        )
        retry_content = retry_resp.choices[0].message.content or ""
        return parse_json_response(retry_content), False
    except Exception as e:
        _log_fallback(step_name, f"original: {content[:150]} | retry error: {e}")
        return fallback, True
