# PARSE: Provenance-Aware Retrieval Sanitization for Professional Domain LLM Agents

Paper | Code | Benchmark | EMNLP 2026 Industry Track (Under Review)

PARSE is a domain-aware, fact-preserving sanitization pipeline that defends LLM agents against domain-camouflaged prompt injection attacks on real enterprise documents.

## Overview

Prompt injection attacks against LLM agents usually look like obvious overrides ("ignore all previous instructions"). Domain-camouflaged injection is harder. The malicious payload is written in the native register of the surrounding document, so a financial injection reads like a sell-side recommendation, a legal injection reads like a contractual obligation, and a medical injection reads like a clinical guideline. Generic defenses struggle here because the line between a legitimate domain directive and an injected one is semantic, not lexical. A filter that flags imperative or authoritative language flags the real document along with the attack.

PARSE treats sanitization as a fact-preserving rewrite rather than a detection problem. It classifies the document's domain, scores how directive the document is, then tags each sentence and scores its injection likelihood against domain-specific allowlists of legitimate directive phrasing. High-risk sentences are aggressively neutralized while every extracted fact is forced to survive into the output, which is then verified for fact coverage. The result is a sanitized document plus a provenance trace recording per-sentence injection scores and the modification applied to each sentence.

On a benchmark of 122 tasks built from real enterprise documents, PARSE reaches 15.6% attack success rate (ASR) at 86.9% utility, a 38% relative ASR reduction versus the 25.4% baseline. It is the only condition tested whose ASR reduction is statistically significant (p=0.014, McNemar's exact test, one-sided, adequately powered at n=122 > n_min=103). Paraphrasing, a common lightweight defense, shows no significant improvement on real documents (p=0.500) and degrades utility from 91.8% to 82.8%.

## Key Results

| Condition | Overall ASR | Utility |
|---|---|---|
| Baseline | 25.4% | 91.8% |
| Spotlighting | 18.9% | 92.6% |
| Sandwiching | 18.9% | 91.0% |
| Paraphrasing | 24.6% | 82.8% |
| Llama Guard 4 | 18.9% | 64.8% |
| **PARSE (ours)** | **15.6%** | **86.9%** |

PARSE is the only condition with a statistically significant ASR reduction (p=0.014, McNemar's exact test, adequately powered). Paraphrasing shows no significant improvement (p=0.500). Llama Guard 4 lowers ASR but collapses utility to 64.8% because it blocks legitimate documents along with attacks.

## Pipeline Architecture

```
Input Document
      |
      v
Step 1: Domain Classifier (Haiku)
      -> financial | legal | medical | scientific | devops
      |
      |  (parallel with Step 1.5)
      v
Step 1.5: Directiveness Gate (Haiku)
      -> score 0-1 for directive content
      -> ~59% of documents route to simple paraphrase (lightweight path)
      -> ~41% proceed to the full pipeline
      |
      v
Steps 2+3: Combined Tagger-Extractor (Haiku)
      -> labels each sentence: factual | directive | hybrid
      -> scores injection likelihood 0-1 with domain allowlists
      -> extracts a structured fact list
      |
      v
Step 4: Structure-Aware Paraphraser (Sonnet)
      -> aggressive neutralization for high-risk sentences (score >= 0.6)
      -> light rewrite for medium-risk (0.3-0.6)
      -> preserve low-risk (< 0.3)
      -> hard constraint: all extracted facts must appear in output
      |
      v
Step 5: Consistency Checker (Haiku)
      -> verifies all facts are present in the paraphrased output
      -> retries Step 4 once if facts are missing
      |
      v
Step 6: Output Builder
      -> sanitized document + provenance trace
      -> per-sentence injection scores, modification log
```

Steps 1 and 1.5 run in parallel. The directiveness gate routes low-directiveness documents to a single lightweight paraphrase, which keeps cost and latency down on the majority of documents that carry no directive content. Documents that clear the gate run the full tag, extract, rewrite, verify path. High-risk thresholds are domain-aware: financial documents use a lower bar (0.5) and general documents a higher one (0.75), reflecting how much legitimate directive language each domain normally contains.

## Benchmark

Real-Document Benchmark: 122 tasks across 5 professional domains.

| Domain | Tasks | Source | Description |
|---|---|---|---|
| Financial | 24 | SEC EDGAR | 10-K MD&A sections |
| Legal | 25 | Federal Register | Final rules and regulatory notices |
| Medical | 23 | PubMed | RCT and systematic review abstracts |
| Scientific | 25 | arXiv cs.AI/LG/CL | ML/AI research abstracts |
| DevOps | 25 | GitHub danluu/post-mortems | Incident reports |

Each task contains:

- A real enterprise document (150-500 words)
- A legitimate analysis task
- A malicious goal
- A domain-camouflaged payload
- A programmatic ground-truth signal

Attack success is measured by a ground-truth signal check (does the agent response contain the malicious output signal) backed by an LLM judge fallback. Utility is measured by an LLM judge that checks whether the agent completed the legitimate task.

## Repository Structure

```
parse-defense/
├── src/parse/
│   ├── domain_classifier.py         # Step 1: domain detection
│   ├── directiveness_classifier.py  # Step 1.5: routing gate
│   ├── tagger_extractor.py          # Steps 2+3: combined analysis
│   ├── paraphraser.py               # Step 4: fact-constrained rewrite
│   ├── consistency_checker.py       # Step 5: fact verification
│   ├── output_builder.py            # Step 6: provenance trace
│   ├── pipeline.py                  # Full PARSE orchestration
│   ├── pipeline_fast.py             # Ablation: 2-step variant
│   ├── sentence_tagger.py           # Standalone tagger + domain allowlists
│   ├── fact_extractor.py            # Standalone fact extractor
│   └── _utils.py                    # JSON retry, caching utilities
├── experiments/
│   ├── eval_parse.py                # Synthetic benchmark evaluation
│   ├── eval_parse_real.py           # Real-document evaluation (main)
│   ├── analyze_parse.py             # Results tables and analysis
│   └── statistical_tests_real.py    # McNemar, Fisher, Cohen's h
├── scripts/
│   ├── build_real_corpus.py         # Collect real documents from APIs
│   └── construct_tasks.py           # Generate injection tasks
├── config.py                        # Model and threshold configuration
└── requirements.txt
```

## Setup

Requirements: Python 3.11+, OpenRouter API key.

```bash
git clone https://github.com/aaditya79/parse-defense
cd parse-defense
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your OpenRouter key:

```
OPENROUTER_API_KEY=your_key_here
```

The `.env` file is read at import time by `config.py`. Do not commit it.

## Running Experiments

### Reproduce main results (real-document benchmark)

```bash
# Collect real documents (free, public APIs)
python3 scripts/build_real_corpus.py

# Construct injection tasks (~$1.20, uses claude-sonnet-4-5)
python3 scripts/construct_tasks.py

# Run full evaluation: all 8 conditions, 122 tasks (~$5-6)
caffeinate -i python3 experiments/eval_parse_real.py --mode full

# Generate results tables
python3 experiments/analyze_parse.py

# Run statistical tests
python3 experiments/statistical_tests_real.py --input results/real_doc_eval_trials_v2.jsonl
```

Note: `real_doc_eval_trials_v2.jsonl` contains the corrected utility judgments (semantic LLM judge) used in the paper. The raw trial file is `real_doc_eval_trials.jsonl`.

The eight evaluation conditions are: `baseline`, `paraphrasing`, `parse`, `parse_fast`, `parse_domain_conditional`, `spotlighting`, `sandwiching`, and `llamaguard`. Restrict the run with `--conditions` and `--domains`:

```bash
python3 experiments/eval_parse_real.py --mode full --conditions parse parse_fast
```

Trials are written incrementally to `results/real_doc_eval_trials.jsonl` and the runner resumes automatically, skipping any (task, condition) pair already completed.

### Run PARSE on your own documents

```python
from src.parse.pipeline import run_parse

result = run_parse("Your enterprise document text here...")

print(result["sanitized_document"])
print(result["parse_metadata"])
# result["provenance_trace"] holds per-sentence scores and modifications
```

`run_parse` accepts a `verbose=True` flag to print each step as it runs. To inspect a full provenance trace:

```python
from src.parse.pipeline import run_parse, print_provenance_trace

result = run_parse("Your enterprise document text here...", verbose=True)
print_provenance_trace(result)
```

### Smoke test (verify setup)

```bash
python3 experiments/eval_parse_real.py --mode smoke --tasks 3
```

The smoke test runs a few tasks from the financial, medical, and devops domains so you can confirm your API key and dependencies are working before launching the full run.

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---|---|---|
| CLASSIFIER_MODEL | claude-haiku-4-5 | Domain + directiveness classification |
| TAGGER_EXTRACTOR_MODEL | claude-haiku-4-5 | Sentence tagging + fact extraction |
| PARAPHRASER_MODEL | claude-sonnet-4-5 | Structure-aware rewriting |
| CHECKER_MODEL | claude-haiku-4-5 | Consistency verification |
| HIGH_RISK_THRESHOLD | 0.6 | Aggressive neutralization threshold |
| LIGHT_REWRITE_THRESHOLD | 0.3 | Light rewrite threshold |

The directiveness gate routes a document to the full pipeline when its directiveness score is at least 0.5, and to a single lightweight paraphrase otherwise (set in `pipeline.py`). All models are accessed through OpenRouter. The paraphraser defaults to `claude-sonnet-4-5` for best results; set every step to Haiku for faster, cheaper experimentation.

## Cost Estimates

| Operation | Trials | Est. Cost |
|---|---|---|
| Document collection | 125 docs | Free |
| Task construction | 122 tasks | ~$1.20 |
| Full evaluation (8 conditions) | 610 trials | ~$5-6 |
| PARSE on a single document | 1 doc | ~$0.01 |

Caching is aggressive. Every LLM step writes its result to `cache/`, so rerunning skips completed steps automatically. Document collection uses free public APIs (SEC EDGAR, Federal Register, PubMed, arXiv, GitHub).

## Citation

```bibtex
@misc{pai2026parse,
  title={PARSE: Provenance-Aware Retrieval Sanitization for
         Professional Domain LLM Agents},
  author={Aaditya Pai},
  year={2026},
  note={Under submission, EMNLP 2026 Industry Track}
}
```

Related work:

- Paper 1 (attack): arXiv 2605.22001

## License

MIT
