"""Step 5: Consistency Checker — verifies all required facts appear in sanitized document."""

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
from src.parse.fact_extractor import FactSet


class FactCoverage(TypedDict):
    fact: str
    present: bool
    found_as: str


class ConsistencyResult(TypedDict):
    passed: bool
    fact_coverage: list[FactCoverage]
    missing_facts: list[str]
    coverage_score: float
    retry_instructions: str


_CACHE_PATH = os.path.join(CONFIG.CACHE_DIR, "consistency_checker.json")

_FALLBACK: ConsistencyResult = {
    "passed": True,
    "fact_coverage": [],
    "missing_facts": [],
    "coverage_score": 1.0,
    "retry_instructions": "",
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


def _cache_key(sanitized_doc: str, facts: str) -> str:
    return hashlib.sha256(f"{sanitized_doc}{facts}".encode()).hexdigest()


def check_consistency(sanitized_document: str, fact_set: FactSet) -> ConsistencyResult:
    if not fact_set["facts"]:
        return dict(_FALLBACK)

    fact_list_str = "\n".join(
        f"- {f['fact_text']} (value: {f['value']})"
        for f in fact_set["facts"]
    )
    key = _cache_key(sanitized_document, fact_list_str)
    cache = _load_cache()
    if key in cache:
        return cache[key]

    client = OpenAI(
        api_key=CONFIG.OPENROUTER_API_KEY,
        base_url=CONFIG.OPENROUTER_BASE_URL,
    )

    prompt = (
        "Verify that all required facts appear in this sanitized document.\n\n"
        f"Required facts:\n{fact_list_str}\n\n"
        f"Sanitized document:\n{sanitized_document}\n\n"
        "For each fact, determine if it is present (exact or semantically equivalent). "
        "Return a JSON object with:\n"
        "{\n"
        "  \"passed\": bool,\n"
        "  \"fact_coverage\": [{\"fact\": str, \"present\": bool, \"found_as\": str}],\n"
        "  \"missing_facts\": [list of missing fact texts],\n"
        "  \"coverage_score\": float,\n"
        "  \"retry_instructions\": str\n"
        "}"
    )

    raw, used_fallback = call_llm_with_json_retry(
        client=client,
        model=CONFIG.CHECKER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=CONFIG.TEMPERATURE,
        step_name="consistency_checker",
        fallback=dict(_FALLBACK),
    )

    result: ConsistencyResult = {
        "passed": bool(raw.get("passed", True)),
        "fact_coverage": raw.get("fact_coverage", []),
        "missing_facts": raw.get("missing_facts", []),
        "coverage_score": float(raw.get("coverage_score", 1.0)),
        "retry_instructions": str(raw.get("retry_instructions", "")),
    }

    if not used_fallback:
        cache[key] = result
        _save_cache(cache)
    return result
