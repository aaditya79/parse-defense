"""
Statistical significance tests for PARSE real-document evaluation results.

Tests:
  1. McNemar's exact test (per condition vs baseline, overall)
  2. Fisher's exact test (per condition × domain vs baseline)
  3. Cohen's h effect size
  4. Power analysis (minimum n for 80% power)

Usage:
  python experiments/statistical_tests_real.py
  python experiments/statistical_tests_real.py --input results/real_doc_eval_trials_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Optional

import scipy.stats as stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

DEFAULT_INPUT = os.path.join(CONFIG.RESULTS_DIR, "real_doc_eval_trials_v2.jsonl")
OUTPUT_PATH   = os.path.join(CONFIG.RESULTS_DIR, "statistical_tests_real.json")

CONDITIONS_ORDER = [
    "baseline", "paraphrasing", "parse",
    "parse_fast", "parse_domain_conditional",
    "spotlighting", "sandwiching", "llamaguard",
]
DOMAINS = ["devops", "financial", "legal", "medical", "scientific"]

ALPHA = 0.05


# ── Data loading ──────────────────────────────────────────────────────────────

def load_trials(path: str) -> list[dict]:
    trials = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r and r.get("followed_injection") is not None:
                trials.append(r)
    return trials


def group_by(trials: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        out[t[key]].append(t)
    return out


# ── McNemar's test ────────────────────────────────────────────────────────────
# Paired comparison: for each task, does the condition differ from baseline?
# We match on task_id.

def mcnemar_test(
    baseline_trials: list[dict],
    condition_trials: list[dict],
) -> tuple[int, int, int, int, float]:
    """
    Build 2×2 discordant-pairs table and run McNemar's exact test.
    Returns (b, c, n_paired, n_total, p_value) where:
      b = baseline=1, condition=0 (baseline fails, condition succeeds)
      c = baseline=0, condition=1 (baseline succeeds, condition fails)
    One-sided p: H1 = condition ASR < baseline ASR → condition has fewer 1s
    → we test that c > b (condition causes more failures than baseline)
    → binomtest(c, b+c, 0.5, alternative='greater') gives p(condition worse)
    → we want p(condition better): binomtest(b, b+c, 0.5, alternative='greater')
    """
    base_map = {t["task_id"]: t["followed_injection"] for t in baseline_trials}
    cond_map = {t["task_id"]: t["followed_injection"] for t in condition_trials}

    b = c = 0  # discordant pairs
    n_paired = 0
    for task_id in set(base_map) & set(cond_map):
        base_inj = int(base_map[task_id])
        cond_inj = int(cond_map[task_id])
        n_paired += 1
        if base_inj == 1 and cond_inj == 0:
            b += 1  # baseline injected, condition blocked
        elif base_inj == 0 and cond_inj == 1:
            c += 1  # baseline safe, condition injected

    # One-sided: H1 = condition has fewer injections (b > c)
    # binomtest(b, b+c, 0.5, alternative='greater') = P(X >= b) under H0: p=0.5
    if b + c == 0:
        p_value = 1.0
    else:
        result = stats.binomtest(b, b + c, 0.5, alternative="greater")
        p_value = result.pvalue

    return b, c, n_paired, len(base_map), p_value


# ── Fisher's exact test ───────────────────────────────────────────────────────

def fisher_test(
    baseline_trials: list[dict],
    condition_trials: list[dict],
) -> tuple[float, float]:
    """
    2×2 contingency: [[cond_inj, cond_ok], [base_inj, base_ok]]
    alternative='less': H1 = condition injection rate < baseline
    Returns (odds_ratio, p_value).
    """
    cond_inj = sum(1 for t in condition_trials if t["followed_injection"])
    cond_ok  = len(condition_trials) - cond_inj
    base_inj = sum(1 for t in baseline_trials if t["followed_injection"])
    base_ok  = len(baseline_trials)  - base_inj

    if base_inj + cond_inj == 0 or base_ok + cond_ok == 0:
        return float("nan"), 1.0

    table = [[cond_inj, cond_ok], [base_inj, base_ok]]
    oddsratio, p_value = stats.fisher_exact(table, alternative="less")
    return oddsratio, p_value


# ── Cohen's h ────────────────────────────────────────────────────────────────

def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))."""
    return 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1)))) \
         - 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))


def min_n_for_power(h: float, power: float = 0.80, alpha: float = 0.05) -> Optional[int]:
    """
    Approximate minimum n per group for given power using normal approximation.
    n ≈ (z_alpha + z_beta)^2 / h^2  (one-sided)
    """
    if abs(h) < 0.001:
        return None
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta  = stats.norm.ppf(power)
    return math.ceil((z_alpha + z_beta) ** 2 / h ** 2)


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def sig_marker(p: float, bonf_threshold: float) -> str:
    if p < bonf_threshold:
        return "**"
    if p < ALPHA:
        return "*"
    return "ns"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"No results file: {args.input}")
        sys.exit(1)

    trials = load_trials(args.input)
    print(f"Loaded {len(trials)} trials from {args.input}")

    by_cond   = group_by(trials, "condition")
    baseline  = by_cond.get("baseline", [])
    conditions = [c for c in CONDITIONS_ORDER if c in by_cond and c != "baseline"]

    # Number of non-baseline comparisons (for Bonferroni)
    n_comparisons = len(conditions)
    bonf_threshold = ALPHA / n_comparisons
    print(f"Bonferroni threshold: {ALPHA}/{n_comparisons} = {bonf_threshold:.4f}\n")

    # ── Overall significance table ────────────────────────────────────────────
    print("=" * 80)
    print("Table 1: Overall ASR — McNemar's test vs baseline (one-sided, H1: ASR lower)")
    print("=" * 80)
    print(f"{'Condition':<26} | {'ASR':>6} | {'p_mcnemar':>9} | {'h':>6} | {'min_n':>6} | {'sig':>4} | {'power?':>7}")
    print("─" * 80)

    base_asr = sum(1 for t in baseline if t["followed_injection"]) / len(baseline) if baseline else 0
    print(f"{'baseline':<26} | {base_asr*100:>5.1f}% | {'—':>9} | {'—':>6} | {'—':>6} | {'—':>4} | {'—':>7}")

    results: dict = {"overall": {}, "domain": {}}

    for cond in conditions:
        ct = by_cond[cond]
        cond_asr = sum(1 for t in ct if t["followed_injection"]) / len(ct) if ct else 0

        b, c, n_paired, n_total, p_mc = mcnemar_test(baseline, ct)
        h = cohens_h(cond_asr, base_asr)
        min_n = min_n_for_power(abs(h))
        powered = "YES" if (min_n and n_total >= min_n) else "NO (low n)"
        sig = sig_marker(p_mc, bonf_threshold)

        print(
            f"{cond:<26} | {cond_asr*100:>5.1f}% | {fmt_p(p_mc):>9} | "
            f"{h:>+6.3f} | {str(min_n) if min_n else '∞':>6} | "
            f"{sig:>4} | {powered:>7}"
        )
        results["overall"][cond] = {
            "asr": cond_asr,
            "p_mcnemar": p_mc,
            "cohens_h": h,
            "min_n": min_n,
            "discordant_b": b,
            "discordant_c": c,
            "n_paired": n_paired,
            "significant_bonferroni": p_mc < bonf_threshold,
            "significant_uncorrected": p_mc < ALPHA,
        }

    print("─" * 80)
    print(f"  * p<{ALPHA} (uncorrected)   ** p<{bonf_threshold:.4f} (Bonferroni)")
    print(f"  Cohen's h: 0.2=small, 0.5=medium, 0.8=large")

    # ── Domain-level Fisher table ─────────────────────────────────────────────
    print()
    print("=" * 80)
    print("Table 2: Domain-level ASR — Fisher's exact test vs baseline (one-sided)")
    print("=" * 80)
    # Bonferroni for domain table: conditions × domains
    n_dom_comparisons = len(conditions) * len(DOMAINS)
    bonf_dom = ALPHA / n_dom_comparisons
    print(f"Bonferroni threshold: {ALPHA}/{n_dom_comparisons} = {bonf_dom:.4f}\n")
    print(f"{'Condition':<26} | {'Domain':<12} | {'ASR':>6} | {'p_fisher':>9} | {'h':>6} | {'sig':>4}")
    print("─" * 70)

    results["domain"] = {}
    for cond in conditions:
        results["domain"][cond] = {}
        ct_all = by_cond[cond]
        for domain in DOMAINS:
            base_d = [t for t in baseline if t["domain"] == domain]
            cond_d = [t for t in ct_all if t["domain"] == domain]
            if not base_d or not cond_d:
                continue

            base_d_asr = sum(1 for t in base_d if t["followed_injection"]) / len(base_d)
            cond_d_asr = sum(1 for t in cond_d if t["followed_injection"]) / len(cond_d)
            _, p_fish = fisher_test(base_d, cond_d)
            h = cohens_h(cond_d_asr, base_d_asr)
            sig = sig_marker(p_fish, bonf_dom)

            print(
                f"{cond:<26} | {domain:<12} | {cond_d_asr*100:>5.1f}% | "
                f"{fmt_p(p_fish):>9} | {h:>+6.3f} | {sig:>4}"
            )
            results["domain"][cond][domain] = {
                "asr": cond_d_asr,
                "baseline_asr": base_d_asr,
                "p_fisher": p_fish,
                "cohens_h": h,
                "n_condition": len(cond_d),
                "n_baseline": len(base_d),
                "significant_bonferroni": p_fish < bonf_dom,
                "significant_uncorrected": p_fish < ALPHA,
            }
        print()

    print("─" * 70)
    print(f"  * p<{ALPHA}   ** p<{bonf_dom:.4f} (Bonferroni across {n_dom_comparisons} tests)")

    # ── Power summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("Power Summary")
    print("=" * 80)
    sig_overall = sum(
        1 for v in results["overall"].values()
        if v.get("significant_uncorrected")
    )
    sig_bonf = sum(
        1 for v in results["overall"].values()
        if v.get("significant_bonferroni")
    )
    print(f"Overall tests: {sig_overall}/{len(conditions)} significant at p<0.05 (uncorrected)")
    print(f"             : {sig_bonf}/{len(conditions)} significant after Bonferroni correction")

    # Find best-powered result
    for cond, v in results["overall"].items():
        h   = v.get("cohens_h", 0)
        mn  = v.get("min_n")
        n   = len(by_cond.get(cond, []))
        pwr = "adequately powered" if mn and n >= mn else f"underpowered (need n≥{mn})"
        print(f"  {cond:<26}: |h|={abs(h):.3f}, {pwr}")

    print()
    print("Note: Results flagged as 'exploratory' where n < minimum for 80% power.")
    print("Confidence intervals and bootstrap tests recommended for camera-ready.")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(CONFIG.RESULTS_DIR, exist_ok=True)

    # numpy 2.x bool/int/float scalars are not JSON-serializable; convert via .item()
    import numpy as np

    def _safe(obj):
        if isinstance(obj, np.generic):
            return obj.item()            # → Python native int/float/bool
        if isinstance(obj, bool):
            return 1 if obj else 0
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_safe(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    with open(OUTPUT_PATH, "w") as f:
        json.dump(_safe(results), f, indent=2)
    print(f"\nFull results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
