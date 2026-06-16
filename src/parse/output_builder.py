"""Step 6: Output Builder — assembles final sanitized document with provenance trace."""

from __future__ import annotations

import time
from typing import Any, TypedDict

from src.parse.sentence_tagger import SentenceTag
from src.parse.fact_extractor import FactSet
from src.parse.consistency_checker import ConsistencyResult


class SentenceModification(TypedDict):
    original: str
    sanitized: str
    injection_score: float
    label: str
    action: str   # preserved | light_rewrite | aggressive_rewrite | unverified


class ParseMetadata(TypedDict):
    domain: str
    domain_confidence: float
    facts_extracted: int
    facts_preserved: float
    sentences_tagged: int
    high_risk_sentences: int
    injection_scores: list[float]
    retry_triggered: bool
    processing_time_ms: int
    consistency_passed: bool
    directiveness_score: float
    routing_decision: str   # 'full_parse' | 'simple_paraphrase'


class ParseOutput(TypedDict):
    sanitized_document: str
    provenance_trace: dict[str, Any]
    parse_metadata: ParseMetadata


def build_output(
    sanitized_document: str,
    original_tags: list[SentenceTag],
    fact_set: FactSet,
    consistency_result: ConsistencyResult,
    domain: str,
    domain_confidence: float,
    retry_triggered: bool,
    start_time_ms: float,
    high_risk_threshold: float = 0.6,
    light_threshold: float = 0.3,
    directiveness_score: float = 0.0,
    routing_decision: str = "full_parse",
) -> ParseOutput:
    end_time_ms = time.time() * 1000
    processing_time_ms = int(end_time_ms - start_time_ms)

    injection_scores = [t["injection_score"] for t in original_tags]

    # Use domain-aware threshold for high-risk count display
    domain_thresholds = {
        "financial": 0.5,
        "legal": 0.7,
        "medical": 0.6,
        "general": 0.75,
    }
    effective_threshold = domain_thresholds.get(domain, high_risk_threshold)
    high_risk_count = sum(1 for s in injection_scores if s >= effective_threshold)

    modifications: list[SentenceModification] = []
    for tag in original_tags:
        score = tag["injection_score"]
        if score >= high_risk_threshold:
            action = "aggressive_rewrite"
        elif score >= light_threshold:
            action = "light_rewrite"
        else:
            action = "preserved"

        modifications.append({
            "original": tag["sentence"],
            "sanitized": "",
            "injection_score": score,
            "label": tag["label"],
            "action": action,
        })

    final_doc = sanitized_document
    if not consistency_result["passed"] and consistency_result["missing_facts"]:
        final_doc += (
            "\n\n[UNVERIFIED: The following facts from the original document could not "
            "be verified in the sanitized output: "
            + "; ".join(consistency_result["missing_facts"])
            + "]"
        )
        for mod in modifications:
            if mod["action"] == "aggressive_rewrite":
                mod["action"] = "unverified"

    provenance_trace = {
        "domain": domain,
        "sentences": modifications,
        "fact_set": fact_set,
        "consistency": consistency_result,
        "routing_decision": routing_decision,
        "directiveness_score": directiveness_score,
    }

    metadata: ParseMetadata = {
        "domain": domain,
        "domain_confidence": domain_confidence,
        "facts_extracted": fact_set["fact_count"],
        "facts_preserved": consistency_result["coverage_score"],
        "sentences_tagged": len(original_tags),
        "high_risk_sentences": high_risk_count,
        "injection_scores": injection_scores,
        "retry_triggered": retry_triggered,
        "processing_time_ms": processing_time_ms,
        "consistency_passed": consistency_result["passed"],
        "directiveness_score": directiveness_score,
        "routing_decision": routing_decision,
    }

    return {
        "sanitized_document": final_doc,
        "provenance_trace": provenance_trace,
        "parse_metadata": metadata,
    }
