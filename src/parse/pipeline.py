"""PARSE pipeline orchestrator v2 — parallel classify, combined tagger+extractor, domain routing."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI

from src.parse.domain_classifier import classify_domain
from src.parse.directiveness_classifier import classify_directiveness
from src.parse.tagger_extractor import tag_and_extract
from src.parse.paraphraser import paraphrase
from src.parse.consistency_checker import check_consistency
from src.parse.output_builder import build_output, ParseOutput
from src.parse.fact_extractor import FactSet

from config import CONFIG

# Domain-aware high-risk thresholds (Option C)
_DOMAIN_HIGH_RISK_THRESHOLD = {
    "financial": 0.5,
    "legal": 0.7,
    "medical": 0.6,
    "general": 0.75,
}


def _simple_paraphrase_doc(document: str, domain: str) -> str:
    """Lightweight paraphrase using the paraphraser model — no sentence analysis."""
    client = OpenAI(
        api_key=CONFIG.OPENROUTER_API_KEY,
        base_url=CONFIG.OPENROUTER_BASE_URL,
    )
    response = client.chat.completions.create(
        model=CONFIG.PARAPHRASER_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"Rewrite the following {domain} document in neutral professional language, "
                "preserving all factual information but using different phrasing. "
                "Do not introduce new facts or commentary.\n\n"
                f"Document:\n{document}"
            ),
        }],
        temperature=CONFIG.TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def _build_simple_paraphrase_output(
    sanitized: str,
    domain: str,
    domain_confidence: float,
    directiveness_score: float,
    start_time_ms: float,
) -> ParseOutput:
    """Wrap a simple paraphrase result in a ParseOutput-shaped dict."""
    empty_fact_set: FactSet = {"facts": [], "domain": domain, "fact_count": 0}
    consistency = {
        "passed": True,
        "fact_coverage": [],
        "missing_facts": [],
        "coverage_score": 1.0,
        "retry_instructions": "",
    }
    return build_output(
        sanitized_document=sanitized,
        original_tags=[],
        fact_set=empty_fact_set,
        consistency_result=consistency,
        domain=domain,
        domain_confidence=domain_confidence,
        retry_triggered=False,
        start_time_ms=start_time_ms,
        high_risk_threshold=CONFIG.HIGH_RISK_THRESHOLD,
        light_threshold=CONFIG.LIGHT_REWRITE_THRESHOLD,
        directiveness_score=directiveness_score,
        routing_decision="simple_paraphrase",
    )


def _full_parse_path(
    document: str,
    domain: str,
    domain_confidence: float,
    domain_signals: list[str],
    directiveness_score: float,
    start_time_ms: float,
    routing_decision: str,
    verbose: bool,
) -> ParseOutput:
    """Run the full PARSE pipeline (tagger+extractor → paraphraser → checker)."""
    domain_threshold = _DOMAIN_HIGH_RISK_THRESHOLD.get(domain, CONFIG.HIGH_RISK_THRESHOLD)

    if verbose:
        print("[PARSE] Step 2+3: Sentence tagging + fact extraction (combined)...")
    te_result = tag_and_extract(document, domain, domain_signals)
    tags = te_result["tags"]
    fact_set = te_result["fact_set"]
    high_risk = [t for t in tags if t["injection_score"] >= domain_threshold]
    if verbose:
        print(
            f"  {len(tags)} sentences, {len(high_risk)} high-risk "
            f"(threshold={domain_threshold}), {fact_set['fact_count']} facts"
        )

    if verbose:
        print("[PARSE] Step 4: Paraphrasing (sanitizing)...")
    sanitized = paraphrase(document, tags, fact_set, domain)
    if verbose:
        print(f"  Sanitized document ({len(sanitized)} chars)")

    if verbose:
        print("[PARSE] Step 5: Consistency check...")
    consistency = check_consistency(sanitized, fact_set)
    retry_triggered = False

    if not consistency["passed"]:
        if verbose:
            print(
                f"  Consistency FAILED (score={consistency['coverage_score']:.2f}), retrying..."
            )
        retry_triggered = True
        sanitized = paraphrase(
            document, tags, fact_set, domain,
            retry_instructions=consistency["retry_instructions"],
        )
        consistency = check_consistency(sanitized, fact_set)
        if verbose:
            status = "PASSED" if consistency["passed"] else "FAILED"
            print(f"  Retry {status} (score={consistency['coverage_score']:.2f})")
    else:
        if verbose:
            print(f"  Consistency PASSED (score={consistency['coverage_score']:.2f})")

    return build_output(
        sanitized_document=sanitized,
        original_tags=tags,
        fact_set=fact_set,
        consistency_result=consistency,
        domain=domain,
        domain_confidence=domain_confidence,
        retry_triggered=retry_triggered,
        start_time_ms=start_time_ms,
        high_risk_threshold=CONFIG.HIGH_RISK_THRESHOLD,
        light_threshold=CONFIG.LIGHT_REWRITE_THRESHOLD,
        directiveness_score=directiveness_score,
        routing_decision=routing_decision,
    )


def run_parse(document: str, verbose: bool = False) -> ParseOutput:
    """
    Full PARSE pipeline v2 with all fixes:
    - Steps 1 + 1.5 run in parallel (Fix 4a)
    - Steps 2+3 combined (Fix 4b)
    - Fix 2 directiveness gate: routes to simple_paraphrase when score < 0.5
    - Fix 3 allowlists active in tagger
    - Fix 1 robust JSON in every LLM step
    """
    start_ms = time.time() * 1000

    # Step 1 + 1.5: parallel domain classification and directiveness scan (Fix 4a)
    if verbose:
        print("[PARSE] Steps 1+1.5: Domain classification + directiveness scan (parallel)...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_domain = ex.submit(classify_domain, document)
        fut_direct = ex.submit(classify_directiveness, document)
        domain_result = fut_domain.result()
        direct_result = fut_direct.result()

    domain = domain_result["domain"]
    domain_confidence = domain_result["confidence"]
    domain_signals = domain_result["domain_signals"]
    directiveness_score = direct_result["directiveness_score"]

    if verbose:
        print(
            f"  domain={domain} (conf={domain_confidence:.2f}), "
            f"directiveness={directiveness_score:.2f}"
        )

    # Fix 2 directiveness gate routing
    if directiveness_score >= 0.5:
        routing_decision = "full_parse"
    else:
        routing_decision = "simple_paraphrase"

    if verbose:
        print(f"  routing={routing_decision}")

    if routing_decision == "simple_paraphrase":
        sanitized = _simple_paraphrase_doc(document, domain)
        return _build_simple_paraphrase_output(
            sanitized, domain, domain_confidence, directiveness_score, start_ms
        )

    return _full_parse_path(
        document=document,
        domain=domain,
        domain_confidence=domain_confidence,
        domain_signals=domain_signals,
        directiveness_score=directiveness_score,
        start_time_ms=start_ms,
        routing_decision=routing_decision,
        verbose=verbose,
    )


def run_parse_domain_conditional(document: str, verbose: bool = False) -> ParseOutput:
    """
    PARSE v2 with domain-conditional routing (Option B):
    - financial: always full PARSE
    - general: simple paraphrase unless directiveness >= 0.7
    - legal: full PARSE (legal allowlist active)
    - medical: full PARSE
    - other: simple paraphrase
    """
    start_ms = time.time() * 1000

    if verbose:
        print("[PARSE-DC] Steps 1+1.5: Domain + directiveness (parallel)...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_domain = ex.submit(classify_domain, document)
        fut_direct = ex.submit(classify_directiveness, document)
        domain_result = fut_domain.result()
        direct_result = fut_direct.result()

    domain = domain_result["domain"]
    domain_confidence = domain_result["confidence"]
    domain_signals = domain_result["domain_signals"]
    directiveness_score = direct_result["directiveness_score"]

    # Option B domain-conditional routing
    if domain == "financial":
        routing_decision = "full_parse"
    elif domain == "general":
        routing_decision = "full_parse" if directiveness_score >= 0.7 else "simple_paraphrase"
    elif domain in ("legal", "medical"):
        routing_decision = "full_parse"
    else:
        routing_decision = "simple_paraphrase"

    if verbose:
        print(
            f"  domain={domain} (conf={domain_confidence:.2f}), "
            f"directiveness={directiveness_score:.2f}, routing={routing_decision}"
        )

    if routing_decision == "simple_paraphrase":
        sanitized = _simple_paraphrase_doc(document, domain)
        return _build_simple_paraphrase_output(
            sanitized, domain, domain_confidence, directiveness_score, start_ms
        )

    return _full_parse_path(
        document=document,
        domain=domain,
        domain_confidence=domain_confidence,
        domain_signals=domain_signals,
        directiveness_score=directiveness_score,
        start_time_ms=start_ms,
        routing_decision="domain_conditional",
        verbose=verbose,
    )


def print_provenance_trace(output: ParseOutput, task_id: str = "") -> None:
    """Pretty-print the full provenance trace for a PARSE output."""
    meta = output["parse_metadata"]
    trace = output["provenance_trace"]

    print(f"\n{'='*70}")
    print(f"PARSE Provenance Trace{f' — {task_id}' if task_id else ''}")
    print(f"{'='*70}")
    print(f"Domain:          {meta['domain']} (confidence={meta['domain_confidence']:.2f})")
    print(f"Directiveness:   {meta['directiveness_score']:.2f}")
    print(f"Routing:         {meta['routing_decision']}")
    print(f"Sentences:       {meta['sentences_tagged']} total, {meta['high_risk_sentences']} high-risk")
    print(f"Facts extracted: {meta['facts_extracted']}")
    print(f"Facts preserved: {meta['facts_preserved']:.2%}")
    print(f"Retry triggered: {meta['retry_triggered']}")
    print(f"Consistency:     {'PASSED' if meta['consistency_passed'] else 'FAILED'}")
    print(f"Processing time: {meta['processing_time_ms']}ms")

    print(f"\n--- Sentence-level injection scores ---")
    for i, sent in enumerate(trace["sentences"]):
        bar = "█" * int(sent["injection_score"] * 10)
        flag = " *** HIGH RISK ***" if sent["injection_score"] >= CONFIG.HIGH_RISK_THRESHOLD else ""
        print(
            f"  [{i+1:2d}] score={sent['injection_score']:.2f} |{bar:<10}| "
            f"{sent['label']:8s} {sent['action']:20s}{flag}"
        )
        print(f"       {sent['original'][:90]}")

    print(f"\n--- Extracted facts ({trace['fact_set']['fact_count']}) ---")
    for f in trace["fact_set"]["facts"]:
        print(f"  [{f['fact_type']}] {f['fact_text']} = {f['value']}")

    print(f"\n--- Consistency coverage ---")
    for fc in trace["consistency"].get("fact_coverage", []):
        status = "✓" if fc["present"] else "✗"
        print(f"  {status} {fc['fact']}")
        if fc.get("found_as"):
            print(f"    found as: {fc['found_as']}")

    if trace["consistency"].get("missing_facts"):
        print(f"\n  Missing facts: {trace['consistency']['missing_facts']}")

    print(f"\n--- Sanitized document ---")
    print(output["sanitized_document"][:1000])
    if len(output["sanitized_document"]) > 1000:
        print(f"  ... [{len(output['sanitized_document'])-1000} chars truncated]")
    print(f"{'='*70}\n")
