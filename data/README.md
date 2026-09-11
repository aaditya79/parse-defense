# Data

Two sets of artifacts live here. The **real-document benchmark** is the one
evaluated in the GroundLM 2026 paper (122 tasks); the **synthetic set** is the
earlier 45-task benchmark used by `experiments/eval_parse.py` and by the prior
work on prompting-based defenses (arXiv:2606.18530).

| File | Role | Contents |
|---|---|---|
| `real_documents.json` | Real-document benchmark | 125 source documents (25 per domain), keyed by domain: `id`, `source`, `url`, `text`, `word_count` (`entity` for financial). 122 are used; three tasks were discarded for construction errors. |
| `real_tasks.json` | Real-document benchmark | The 122 evaluated tasks: document, legitimate `instruction`, `malicious_goal`, `camouflage_payload`, `static_payload`, `malicious_output_signal`, `key_facts_needed`, `source`, `url`. Tasks, goals, and payloads are LLM-generated (`claude-sonnet-4-5`) with no human validation. |
| `tasks.json` | Synthetic set (earlier) | 200 synthetic tasks (40 each `fin_*`, `gen_*`, `leg_*`, `med_*`, `sci_*`); the 45 ids in `config.BENCHMARK_TASK_IDS` form the evaluated synthetic benchmark. |
| `camouflage_payloads.json` | Synthetic set (earlier) | Domain-camouflaged payload variants (3 per task, 600 total) for the 200 synthetic tasks. |
| `static_payloads.json` | Both | 20 static (non-camouflaged) injection templates in four categories. |

## Provenance of the real documents

All carrier documents were retrieved from public sources with
`scripts/build_real_corpus.py`. Every `url` field points at the original.

| Domain | Tasks | Source | URL pattern |
|---|---|---|---|
| Financial | 24 | SEC EDGAR, 10-K MD&A sections | `https://www.sec.gov/Archives/edgar/data/...` |
| Legal | 25 | Federal Register, final rules | `https://www.federalregister.gov/documents/...` |
| Medical | 23 | PubMed, RCT abstracts | `https://pubmed.ncbi.nlm.nih.gov/<pmid>/` |
| Scientific | 25 | arXiv cs.AI/LG/CL abstracts | `http://arxiv.org/abs/<id>` |
| DevOps | 25 | danluu/post-mortems README | `https://raw.githubusercontent.com/danluu/post-mortems/master/README.md` |

**DevOps note.** The DevOps texts are 150-500-word excerpts of the curated
incident summaries in the README of
[danluu/post-mortems](https://github.com/danluu/post-mortems), not the linked
postmortems themselves. That repository carries no explicit license. The
excerpts are included here for research reproducibility only and will be
removed on request.

The SEC, Federal Register, and PubMed texts are U.S. government or
government-hosted public records; arXiv abstracts are redistributed under
arXiv's terms.
