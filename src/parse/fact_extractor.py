"""Step 3: Fact Extractor — extracts structured fact constraints from factual/hybrid sentences."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openai import OpenAI
from config import CONFIG
from src.parse._utils import parse_json_response
from src.parse.sentence_tagger import SentenceTag


class Fact(TypedDict):
    fact_text: str
    fact_type: str
    value: str
    source_sentence: str


class FactSet(TypedDict):
    facts: list[Fact]
    domain: str
    fact_count: int


_CACHE_PATH = os.path.join(CONFIG.CACHE_DIR, "fact_extractor.json")


def _load_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(CONFIG.CACHE_DIR, exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(factual_sentences: str) -> str:
    return hashlib.sha256(factual_sentences.encode()).hexdigest()


def extract_facts(tags: list[SentenceTag], domain: str) -> FactSet:
    factual_sentences = [
        t["sentence"] for t in tags
        if t["label"] in ("factual", "hybrid")
    ]

    if not factual_sentences:
        return {"facts": [], "domain": domain, "fact_count": 0}

    combined = "\n".join(factual_sentences)
    key = _cache_key(combined)
    cache = _load_cache()
    if key in cache:
        return cache[key]

    client = OpenAI(
        api_key=CONFIG.OPENROUTER_API_KEY,
        base_url=CONFIG.OPENROUTER_BASE_URL,
    )

    domain_hints = {
        "financial": "numbers, percentages, company names, ratings (BUY/SELL/HOLD), risk levels, dates",
        "legal": "clause names, obligations, parties, compliance status, risk flags",
        "medical": "diagnoses, medications, values, recommendations, patient identifiers",
        "devops": "system names, configuration values, error codes, service dependencies",
        "general": "named entities, numerical claims, stated conclusions",
    }
    hints = domain_hints.get(domain, domain_hints["general"])

    prompt = (
        f"Extract all verifiable facts from the following {domain} sentences.\n"
        "These are confirmed factual sentences from the document.\n"
        f"Focus on: {hints}\n\n"
        "Return a JSON object with key 'facts' containing a list where each fact has:\n"
        "{\"fact_text\": str, \"fact_type\": str, \"value\": str, \"source_sentence\": str}\n\n"
        "Also include \"domain\" and \"fact_count\" at the top level.\n\n"
        f"Factual sentences:\n{combined}"
    )

    response = client.chat.completions.create(
        model=CONFIG.EXTRACTOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=CONFIG.TEMPERATURE,
    )

    raw = parse_json_response(response.choices[0].message.content)
    facts = raw.get("facts", [])
    result: FactSet = {
        "facts": [
            {
                "fact_text": str(f.get("fact_text", "")),
                "fact_type": str(f.get("fact_type", "")),
                "value": str(f.get("value", "")),
                "source_sentence": str(f.get("source_sentence", "")),
            }
            for f in facts
        ],
        "domain": domain,
        "fact_count": len(facts),
    }

    cache[key] = result
    _save_cache(cache)
    return result
