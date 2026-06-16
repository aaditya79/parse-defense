"""
PARSE analysis script.
Prints comparison table and detailed metrics from parse_eval_trials.jsonl.

Usage:
  python experiments/analyze_parse.py
  python experiments/analyze_parse.py --input results/parse_eval_trials.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG

_CONDITIONS_ORDER = [
    "baseline",
    "paraphrasing",
    "parse",
    "parse_fast",
    "parse_domain_conditional",
    "spotlighting",
    "sandwiching",
    "llamaguard",
]

_PARSE_CONDITIONS = {"parse", "parse_fast", "parse_domain_conditional"}


def load_trials(path: str) -> list[dict]:
    trials = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if "error" not in r:
                    trials.append(r)
    return trials


def compute_metrics(trials: list[dict]) -> tuple[dict, list[str]]:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        by_condition[t["condition"]].append(t)

    domains = sorted(set(t["domain"] for t in trials))

    results = {}
    for condition, ctrials in by_condition.items():
        asr = sum(1 for t in ctrials if t.get("followed_injection")) / len(ctrials) if ctrials else 0
        util = sum(1 for t in ctrials if t.get("task_success")) / len(ctrials) if ctrials else 0

        domain_asr = {}
        for domain in domains:
            dtrials = [t for t in ctrials if t["domain"] == domain]
            domain_asr[domain] = (
                sum(1 for t in dtrials if t.get("followed_injection")) / len(dtrials)
                if dtrials else None
            )

        times = [t.get("processing_time_ms", 0) for t in ctrials]
        avg_time = sum(times) / len(times) if times else 0

        results[condition] = {
            "n_trials": len(ctrials),
            "asr": asr,
            "utility": util,
            "domain_asr": domain_asr,
            "avg_time_ms": avg_time,
        }

        # PARSE-specific metrics (shared across all parse_* conditions)
        if condition in _PARSE_CONDITIONS:
            parse_meta_trials = [t for t in ctrials if "parse_metadata" in t]
            if parse_meta_trials:
                retry_rate = sum(
                    1 for t in parse_meta_trials if t["parse_metadata"].get("retry_triggered")
                ) / len(parse_meta_trials)

                facts_preserved = [
                    t["parse_metadata"].get("facts_preserved", 1.0)
                    for t in parse_meta_trials
                ]
                avg_facts_preserved = sum(facts_preserved) / len(facts_preserved)

                high_risk_counts = [
                    t["parse_metadata"].get("high_risk_sentences", 0)
                    for t in parse_meta_trials
                ]
                avg_high_risk = sum(high_risk_counts) / len(high_risk_counts)

                all_scores = []
                for t in parse_meta_trials:
                    all_scores.extend(t["parse_metadata"].get("injection_scores", []))
                avg_injection_score = sum(all_scores) / len(all_scores) if all_scores else 0

                directiveness_scores = [
                    t["parse_metadata"].get("directiveness_score", 0.0)
                    for t in parse_meta_trials
                ]
                avg_directiveness = sum(directiveness_scores) / len(directiveness_scores)

                routing_counts: dict[str, int] = defaultdict(int)
                for t in parse_meta_trials:
                    routing_counts[t["parse_metadata"].get("routing_decision", "unknown")] += 1

                domain_facts_preserved = {}
                domain_retry_rate = {}
                domain_directiveness = {}
                for domain in domains:
                    dtrials = [t for t in parse_meta_trials if t["domain"] == domain]
                    if dtrials:
                        domain_facts_preserved[domain] = sum(
                            t["parse_metadata"].get("facts_preserved", 1.0) for t in dtrials
                        ) / len(dtrials)
                        domain_retry_rate[domain] = sum(
                            1 for t in dtrials if t["parse_metadata"].get("retry_triggered")
                        ) / len(dtrials)
                        domain_directiveness[domain] = sum(
                            t["parse_metadata"].get("directiveness_score", 0.0) for t in dtrials
                        ) / len(dtrials)

                results[condition].update({
                    "retry_rate": retry_rate,
                    "avg_facts_preserved": avg_facts_preserved,
                    "avg_high_risk_sentences": avg_high_risk,
                    "avg_injection_score": avg_injection_score,
                    "avg_directiveness": avg_directiveness,
                    "routing_counts": dict(routing_counts),
                    "domain_facts_preserved": domain_facts_preserved,
                    "domain_retry_rate": domain_retry_rate,
                    "domain_directiveness": domain_directiveness,
                })

    return results, domains


def compute_injection_detection_accuracy(trials: list[dict]) -> dict:
    parse_trials = [t for t in trials if t["condition"] in _PARSE_CONDITIONS and "parse_metadata" in t]
    if not parse_trials:
        return {}

    total_sentences = 0
    high_risk_sentences = 0
    for t in parse_trials:
        scores = t["parse_metadata"].get("injection_scores", [])
        total_sentences += len(scores)
        high_risk_sentences += sum(1 for s in scores if s >= CONFIG.HIGH_RISK_THRESHOLD)

    if total_sentences == 0:
        return {}

    return {
        "total_sentences": total_sentences,
        "high_risk_flagged": high_risk_sentences,
        "flag_rate": high_risk_sentences / total_sentences,
        "avg_sentences_per_doc": total_sentences / len(parse_trials),
    }


def print_comparison_table(results: dict, domains: list[str]) -> None:
    present = [c for c in _CONDITIONS_ORDER if c in results]

    domain_cols = " | ".join(f"{d[:10]:>12}_ASR" for d in domains)
    width = 22 + len(domains) * 16 + 20
    print(f"\n{'─'*width}")
    print(f"{'Condition':<22} | {domain_cols} | {'overall':>8} | {'utility':>7} | {'avg_ms':>8}")
    print(f"{'─'*width}")

    for condition in present:
        r = results[condition]
        domain_vals = " | ".join(
            f"{r['domain_asr'].get(d, 0)*100:>12.1f}%"
            if r['domain_asr'].get(d) is not None
            else f"{'—':>13}"
            for d in domains
        )
        time_str = f"{r['avg_time_ms']:>8.0f}" if r["avg_time_ms"] else "       -"
        overall = r["asr"] * 100
        print(
            f"{condition:<22} | {domain_vals} | {overall:>7.1f}% | {r['utility']*100:>7.1f}% | {time_str}"
        )

    print(f"{'─'*width}")


def print_parse_details(results: dict, domains: list[str]) -> None:
    for condition in _CONDITIONS_ORDER:
        if condition not in results or condition not in _PARSE_CONDITIONS:
            continue
        r = results[condition]
        if "retry_rate" not in r:
            continue

        print(f"\n--- {condition} metrics ---")
        print(f"  Avg directiveness score:  {r.get('avg_directiveness', 0):.3f}")
        print(f"  Routing decisions:        {r.get('routing_counts', {})}")
        print(f"  Retry rate:               {r.get('retry_rate', 0):.1%}")
        print(f"  Avg facts preserved:      {r.get('avg_facts_preserved', 0):.1%}")
        print(f"  Avg high-risk sentences:  {r.get('avg_high_risk_sentences', 0):.1f}")
        print(f"  Avg injection score:      {r.get('avg_injection_score', 0):.3f}")

        if r.get("domain_directiveness"):
            print(f"  Avg directiveness by domain:")
            for domain in domains:
                val = r["domain_directiveness"].get(domain)
                if val is not None:
                    print(f"    {domain:<15}: {val:.3f}")


def print_example_trace(trials: list[dict]) -> None:
    parse_trials = [t for t in trials if t["condition"] == "parse" and "provenance_trace" in t]
    if not parse_trials:
        return

    fin_trials = [t for t in parse_trials if t["domain"] == "financial"]
    trial = fin_trials[0] if fin_trials else parse_trials[0]

    print(f"\n{'='*70}")
    print(f"Example Provenance Trace — {trial['task_id']} ({trial['domain']})")
    print(f"{'='*70}")

    trace = trial["provenance_trace"]
    meta = trial.get("parse_metadata", {})

    print(f"Domain detected: {meta.get('domain_detected', '?')}")
    print(f"Directiveness:   {meta.get('directiveness_score', '?')}")
    print(f"Routing:         {meta.get('routing_decision', '?')}")
    print(f"Followed injection: {trial['followed_injection']}")
    print(f"Task success: {trial['task_success']}")
    print(f"High-risk sentences: {meta.get('high_risk_sentences', '?')}")
    print(f"Facts extracted: {meta.get('facts_extracted', '?')}")
    if isinstance(meta.get("facts_preserved"), float):
        print(f"Facts preserved: {meta.get('facts_preserved'):.1%}")
    print(f"Retry triggered: {meta.get('retry_triggered', '?')}")

    print(f"\n--- Sentence injection scores ---")
    for i, sent in enumerate(trace.get("sentences", [])[:10]):
        bar = "█" * int(sent["injection_score"] * 10)
        flag = " <<HIGH RISK>>" if sent["injection_score"] >= CONFIG.HIGH_RISK_THRESHOLD else ""
        print(f"  [{i+1:2d}] {sent['injection_score']:.2f} |{bar:<10}| {sent['label']:8s}{flag}")
        print(f"       {sent['original'][:80]}")

    facts = trace.get("fact_set", {}).get("facts", [])
    print(f"\n--- Extracted facts ({len(facts)}) ---")
    for f in facts[:8]:
        print(f"  [{f['fact_type']}] {f['fact_text']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PARSE evaluation results")
    parser.add_argument(
        "--input",
        default=os.path.join(CONFIG.RESULTS_DIR, "parse_eval_trials.jsonl"),
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"No results file found at {args.input}")
        print("Run experiments/eval_parse.py first.")
        sys.exit(1)

    trials = load_trials(args.input)
    print(f"Loaded {len(trials)} trials from {args.input}")

    results, domains = compute_metrics(trials)
    detection = compute_injection_detection_accuracy(trials)

    width = 22 + len(domains) * 16 + 20
    print(f"\n{'='*width}")
    print("PARSE vs. Baselines — Attack Success Rate (ASR) and Utility")
    print(f"{'='*width}")
    print_comparison_table(results, domains)

    print(f"\n--- Overall ASR comparison ---")
    for cond in _CONDITIONS_ORDER:
        if cond in results:
            r = results[cond]
            print(f"  {cond:<25}: ASR={r['asr']:.1%}, Utility={r['utility']:.1%}, n={r['n_trials']}")

    if detection:
        print(f"\n--- Sentence-level injection detection ---")
        print(f"  Total sentences analyzed:  {detection['total_sentences']}")
        print(f"  High-risk flagged (>=0.6):  {detection['high_risk_flagged']}")
        print(f"  Flag rate:                  {detection['flag_rate']:.1%}")
        print(f"  Avg sentences per doc:      {detection['avg_sentences_per_doc']:.1f}")

    print_parse_details(results, domains)
    print_example_trace(trials)


if __name__ == "__main__":
    main()
