"""Step 1: Domain Classifier — classifies document into financial/legal/medical/devops/general."""

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


class DomainResult(TypedDict):
    domain: str
    confidence: float
    domain_signals: list[str]


_CACHE_PATH = os.path.join(CONFIG.CACHE_DIR, "domain_classifier.json")

_FALLBACK: DomainResult = {
    "domain": "general",
    "confidence": 0.5,
    "domain_signals": [],
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


def classify_domain(document: str) -> DomainResult:
    key = _cache_key(document)
    cache = _load_cache()
    if key in cache:
        return cache[key]

    client = OpenAI(
        api_key=CONFIG.OPENROUTER_API_KEY,
        base_url=CONFIG.OPENROUTER_BASE_URL,
    )

    prompt = (
        "Classify this document into exactly one domain: "
        "financial, legal, medical, devops, or general.\n"
        "Return JSON: {\"domain\": str, \"confidence\": float, "
        "\"domain_signals\": [list of 3 key phrases that indicate the domain]}\n\n"
        f"Document:\n{document}"
    )

    result, used_fallback = call_llm_with_json_retry(
        client=client,
        model=CONFIG.CLASSIFIER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=CONFIG.TEMPERATURE,
        step_name="domain_classifier",
        fallback=dict(_FALLBACK),
    )

    result["domain"] = result.get("domain", "general").lower()
    result["confidence"] = float(result.get("confidence", 0.5))
    if "domain_signals" not in result or not isinstance(result["domain_signals"], list):
        result["domain_signals"] = []

    if not used_fallback:
        cache[key] = result
        _save_cache(cache)
    return result
