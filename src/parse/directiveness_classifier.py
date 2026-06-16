"""Step 1.5: Directiveness Classifier — document-level gate before sentence processing."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openai import OpenAI
from config import CONFIG
from src.parse._utils import call_llm_with_json_retry


class DirectivenessResult(TypedDict):
    directiveness_score: float
    directive_signals: list[str]
    recommendation: str   # 'full_parse' | 'simple_paraphrase'


_CACHE_PATH = os.path.join(CONFIG.CACHE_DIR, "directiveness_classifier.json")

_FALLBACK: DirectivenessResult = {
    "directiveness_score": 0.5,
    "directive_signals": [],
    "recommendation": "full_parse",
}


def _load_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(CONFIG.CACHE_DIR, exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(document: str) -> str:
    return hashlib.sha256(document[:500].encode()).hexdigest()


def classify_directiveness(document: str) -> DirectivenessResult:
    key = _cache_key(document)
    cache = _load_cache()
    if key in cache:
        return cache[key]

    client = OpenAI(
        api_key=CONFIG.OPENROUTER_API_KEY,
        base_url=CONFIG.OPENROUTER_BASE_URL,
    )

    prompt = (
        "Score this document 0.0-1.0 for directiveness — the degree to which "
        "it contains authority claims, recommendations, or instructions that "
        "could redirect an AI agent's behavior.\n\n"
        "0.0 = purely factual, no recommendations or directives\n"
        "0.5 = mix of facts and some guidance/recommendations\n"
        "1.0 = heavily directive, full of authority claims and instructions\n\n"
        "Return JSON: {\"directiveness_score\": float, "
        "\"directive_signals\": [list of 3 phrases that indicate directiveness], "
        "\"recommendation\": \"full_parse\" or \"simple_paraphrase\"}\n\n"
        f"Document:\n{document}"
    )

    result, used_fallback = call_llm_with_json_retry(
        client=client,
        model=CONFIG.DIRECTIVENESS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=CONFIG.TEMPERATURE,
        step_name="directiveness_classifier",
        fallback=dict(_FALLBACK),
    )

    result["directiveness_score"] = float(result.get("directiveness_score", 0.5))
    if not isinstance(result.get("directive_signals"), list):
        result["directive_signals"] = []
    rec = result.get("recommendation", "full_parse")
    if rec not in ("full_parse", "simple_paraphrase"):
        rec = "full_parse" if result["directiveness_score"] >= 0.5 else "simple_paraphrase"
    result["recommendation"] = rec

    if not used_fallback:
        cache[key] = result
        _save_cache(cache)
    return result
