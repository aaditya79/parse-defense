"""
Phase 1: Real Document Corpus Builder
Collects 20+ real professional documents per domain from public APIs.

Usage:
  python scripts/build_real_corpus.py
  python scripts/build_real_corpus.py --domain financial   # single domain
  python scripts/build_real_corpus.py --target 25          # more docs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from typing import Optional

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

OUTPUT_PATH = os.path.join(CONFIG.DATA_DIR, "real_documents.json")
CACHE_PATH = os.path.join(CONFIG.CACHE_DIR, "corpus_fetcher.json")
MIN_WORDS = 150
MAX_WORDS = 500

_BASE_HEADERS = {"User-Agent": "PARSE Research aup2005@columbia.edu"}

# ── HTTP cache + backoff ───────────────────────────────────────────────────────

def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(CONFIG.CACHE_DIR, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def fetch(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    max_retries: int = 4,
    sleep_between: float = 0.0,
) -> Optional[str]:
    """Cached HTTP GET with exponential backoff on 429/503."""
    cache = _load_cache()
    raw_key = url + json.dumps(params or {}, sort_keys=True)
    key = hashlib.sha256(raw_key.encode()).hexdigest()
    if key in cache:
        return cache[key]

    hdrs = dict(_BASE_HEADERS)
    if headers:
        hdrs.update(headers)

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=hdrs, params=params, timeout=30)
            if resp.status_code in (429, 503):
                wait = 5 * (2 ** attempt)
                print(f"    Rate limited ({resp.status_code}), waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return None
            text = resp.text
            cache[key] = text
            _save_cache(cache)
            if sleep_between:
                time.sleep(sleep_between)
            return text
        except Exception as e:
            wait = 2 * (2 ** attempt)
            if attempt < max_retries - 1:
                time.sleep(wait)
    return None


# ── Quality filters ────────────────────────────────────────────────────────────

_PII_RE = [
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    re.compile(r'\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b'),
]
_NON_EN_RE = re.compile(r'[^\x00-\x7F]{8,}')
_TABLE_RE = re.compile(r'(\d[\d.,\s%$]{5,}){4,}')  # rows of numbers = financial table


def scrub_pii(text: str) -> str:
    for pat in _PII_RE:
        text = pat.sub('[REDACTED]', text)
    return text


def clean_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def truncate(text: str, max_w: int = MAX_WORDS) -> str:
    words = text.split()
    return ' '.join(words[:max_w])


def passes_quality(text: str) -> tuple[bool, str]:
    words = text.split()
    if len(words) < MIN_WORDS:
        return False, f"too short ({len(words)}w)"
    if _NON_EN_RE.search(text):
        return False, "non-English"
    first_sent = re.split(r'[.!?]', text)[0]
    if len(first_sent.split()) < 4:
        return False, "incoherent first sentence"
    # Reject if it's mostly a financial table
    if _TABLE_RE.search(text[:500]):
        return False, "appears to be a data table"
    return True, "ok"


def make_chunk(raw: str) -> Optional[str]:
    """Clean, truncate, return None if too short after cleaning."""
    text = clean_whitespace(scrub_pii(raw))
    words = text.split()
    if len(words) < MIN_WORDS:
        return None
    return truncate(text, MAX_WORDS)


# ── Financial: SEC EDGAR ───────────────────────────────────────────────────────
# Uses EFTS full-text search → filing directory listing → main 10-K HTM
# Targets MD&A (Item 7) — always substantive; falls back to risk factors.

def _edgar_filing_main_doc(cik: str, adsh: str) -> Optional[str]:
    """Return URL of the main 10-K HTM document from the filing index."""
    adsh_clean = adsh.replace("-", "")
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh_clean}/"
        f"{adsh}-index.html"
    )
    raw = fetch(index_url, sleep_between=0.15)
    if not raw:
        return None
    soup = BeautifulSoup(raw, "lxml")
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 4:
            doc_type = cells[3].get_text(strip=True).upper()
            if doc_type == "10-K":
                a = cells[2].find("a", href=True)
                if a:
                    name = a["href"].split("/")[-1]
                    return (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{cik}/{adsh_clean}/{name}"
                    )
    return None


def _extract_section(text: str, patterns: list[str], next_section_re: str) -> Optional[str]:
    """Extract a named section from 10-K plain text."""
    lower = text.lower()
    # Skip the first hit (usually table of contents) by starting from second
    occurrences = [m.start() for p in patterns for m in re.finditer(p, lower)]
    occurrences.sort()

    for occ_idx, start in enumerate(occurrences):
        # Skip hits that are in the table of contents (usually very early)
        if start < 2000 and occ_idx == 0 and len(occurrences) > 1:
            continue
        # Find end of section at next major Item heading
        tail = lower[start + 50:]
        next_m = re.search(next_section_re, tail)
        end = start + 50 + next_m.start() if next_m else start + 4000
        chunk = text[start:end]
        words = chunk.split()
        if len(words) >= MIN_WORDS:
            return ' '.join(words[:MAX_WORDS])
    return None


def collect_financial(target: int) -> list[dict]:
    print(f"\n[financial] Collecting from SEC EDGAR (target={target})...")
    docs: list[dict] = []

    search_raw = fetch(
        "https://efts.sec.gov/LATEST/search-index",
        params={
            "q": '"management discussion" OR "results of operations"',
            "forms": "10-K",
            "dateRange": "custom",
            "startdt": "2024-03-01",
            "enddt": "2025-01-01",
        },
        sleep_between=0.15,
    )
    if not search_raw:
        print("  ERROR: EDGAR EFTS unreachable")
        return docs

    hits = json.loads(search_raw).get("hits", {}).get("hits", [])
    print(f"  {len(hits)} EDGAR search hits")

    for hit in hits:
        if len(docs) >= target:
            break
        src = hit.get("_source", {})
        adsh = src.get("adsh", "")
        ciks = src.get("ciks", [])
        display = src.get("display_names", ["Unknown"])[0]
        if not adsh or not ciks:
            continue
        cik = ciks[0].lstrip("0")

        doc_url = _edgar_filing_main_doc(cik, adsh)
        if not doc_url:
            continue

        doc_raw = fetch(doc_url, sleep_between=0.15)
        if not doc_raw:
            continue

        soup = BeautifulSoup(doc_raw, "lxml")
        full_text = re.sub(r'\s+', ' ', soup.get_text(separator=' '))

        # Try MD&A (Item 7) first — always required and substantive
        chunk = _extract_section(
            full_text,
            patterns=[r"management.{1,10}discussion", r"item\s+7\.?\s"],
            next_section_re=r"item\s+7a\.?\s|item\s+8\.?\s",
        )
        # Fallback: Risk Factors (Item 1A)
        if not chunk:
            chunk = _extract_section(
                full_text,
                patterns=[r"item\s+1a\.?\s*risk\s+factor", r"risk\s+factors"],
                next_section_re=r"item\s+1b\.?\s|item\s+2\.?\s",
            )
        if not chunk:
            continue

        chunk = make_chunk(chunk)
        if not chunk:
            continue
        ok, reason = passes_quality(chunk)
        if not ok:
            continue

        docs.append({
            "id": f"fin_real_{len(docs)+1:03d}",
            "source": "SEC EDGAR",
            "entity": display,
            "url": doc_url,
            "text": chunk,
            "word_count": len(chunk.split()),
        })
        if len(docs) % 5 == 0:
            print(f"  {len(docs)} collected...")

    print(f"  Done: {len(docs)}/{target} financial docs")
    return docs


# ── Legal: Federal Register API ───────────────────────────────────────────────
# Uses https://www.federalregister.gov/api/v1/ — free, no auth, rich legal text.
# Fetches final rules and proposed rules; extracts SUPLINF/PREAMB XML sections.

_FR_XML_TAGS = ["SUPLINF", "PREAMB", "SUPLINFO", "RULE", "PRORULE"]
_FR_TEXT_TAGS = ["P", "FP", "HD", "SECTION"]


def _parse_fr_xml(xml_text: str) -> Optional[str]:
    """Extract prose text from a Federal Register XML document."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    # Collect text from paragraph and section elements
    paragraphs: list[str] = []

    def collect(node: ET.Element, depth: int = 0) -> None:
        tag = node.tag.upper()
        text = (node.text or "").strip()
        if tag in _FR_TEXT_TAGS and text and len(text.split()) >= 8:
            paragraphs.append(text)
        for child in node:
            collect(child, depth + 1)
            tail = (child.tail or "").strip()
            if tail and len(tail.split()) >= 5:
                paragraphs.append(tail)

    collect(root)
    if not paragraphs:
        return None
    # Skip first 2 paragraphs (usually just agency/CFR citation)
    body = " ".join(paragraphs[2:])
    return body if body else None


def collect_legal(target: int) -> list[dict]:
    print(f"\n[legal] Collecting from Federal Register (target={target})...")
    docs: list[dict] = []
    seen_doc_nums: set[str] = set()

    doc_types = ["RULE", "PRORULE", "NOTICE"]
    page = 1

    while len(docs) < target and page <= 8:
        for doc_type in doc_types:
            if len(docs) >= target:
                break
            raw = fetch(
                "https://www.federalregister.gov/api/v1/documents.json",
                params={
                    "conditions[type][]": doc_type,
                    "per_page": 20,
                    "page": page,
                    "order": "newest",
                    "fields[]": [
                        "document_number", "title", "abstract",
                        "html_url", "full_text_xml_url", "type",
                    ],
                },
                sleep_between=0.5,
            )
            if not raw:
                continue
            try:
                results = json.loads(raw).get("results", [])
            except Exception:
                continue

            for doc in results:
                if len(docs) >= target:
                    break
                doc_num = doc.get("document_number", "")
                if doc_num in seen_doc_nums:
                    continue
                seen_doc_nums.add(doc_num)

                xml_url = doc.get("full_text_xml_url", "")
                if not xml_url:
                    continue

                xml_raw = fetch(xml_url, sleep_between=0.5)
                if not xml_raw:
                    continue

                body = _parse_fr_xml(xml_raw)
                if not body:
                    # Fallback: use abstract if long enough
                    body = doc.get("abstract", "")

                chunk = make_chunk(body)
                if not chunk:
                    continue
                ok, _ = passes_quality(chunk)
                if not ok:
                    continue

                docs.append({
                    "id": f"leg_real_{len(docs)+1:03d}",
                    "source": "Federal Register",
                    "title": doc.get("title", ""),
                    "doc_type": doc.get("type", doc_type),
                    "url": doc.get("html_url", xml_url),
                    "text": chunk,
                    "word_count": len(chunk.split()),
                })
                if len(docs) % 5 == 0:
                    print(f"  {len(docs)} collected...")
        page += 1

    print(f"  Done: {len(docs)}/{target} legal docs")
    return docs


# ── Medical: PubMed ────────────────────────────────────────────────────────────

def collect_medical(target: int) -> list[dict]:
    print(f"\n[medical] Collecting from PubMed (target={target})...")
    docs: list[dict] = []

    # Multiple search terms for diversity
    terms = [
        "randomized controlled trial[pt] AND treatment[tiab]",
        "clinical trial[pt] AND diagnosis[tiab]",
        "systematic review[pt] AND intervention[tiab]",
    ]

    pmids_seen: set[str] = set()

    for term in terms:
        if len(docs) >= target:
            break

        search_raw = fetch(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": term, "retmax": 40, "retmode": "json"},
            sleep_between=0.4,
        )
        if not search_raw:
            continue

        pmids = json.loads(search_raw).get("esearchresult", {}).get("idlist", [])
        new_pmids = [p for p in pmids if p not in pmids_seen]
        pmids_seen.update(new_pmids)

        for batch_start in range(0, len(new_pmids), 10):
            if len(docs) >= target:
                break
            batch = new_pmids[batch_start:batch_start + 10]
            xml_raw = fetch(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
                sleep_between=0.4,
            )
            if not xml_raw:
                continue

            try:
                root = ET.fromstring(xml_raw)
            except ET.ParseError:
                continue

            for article in root.findall(".//PubmedArticle"):
                if len(docs) >= target:
                    break
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else "?"

                abstracts = article.findall(".//AbstractText")
                if not abstracts:
                    continue

                parts = []
                for el in abstracts:
                    label = el.get("Label", "")
                    body = (el.text or "").strip()
                    if label:
                        parts.append(f"{label}: {body}")
                    else:
                        parts.append(body)
                abstract = " ".join(parts)

                chunk = make_chunk(abstract)
                if not chunk:
                    continue
                ok, _ = passes_quality(chunk)
                if not ok:
                    continue

                docs.append({
                    "id": f"med_real_{len(docs)+1:03d}",
                    "source": "PubMed",
                    "pmid": pmid,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "text": chunk,
                    "word_count": len(chunk.split()),
                })
                if len(docs) % 5 == 0:
                    print(f"  {len(docs)} collected...")

    print(f"  Done: {len(docs)}/{target} medical docs")
    return docs


# ── Scientific: arXiv ─────────────────────────────────────────────────────────

def _parse_arxiv_atom(raw: str) -> list[dict]:
    """Parse arXiv Atom XML, returning list of {title, abstract, url}."""
    # Strip default namespace to simplify XPath
    raw_clean = re.sub(r'\sxmlns="[^"]+"', '', raw, count=1)
    try:
        root = ET.fromstring(raw_clean)
    except ET.ParseError:
        return []
    entries = []
    for entry in root.findall(".//entry"):
        title = (getattr(entry.find("title"), "text", "") or "").strip()
        summary = (getattr(entry.find("summary"), "text", "") or "").strip()
        url = (getattr(entry.find("id"), "text", "") or "").strip()
        if summary:
            entries.append({"title": title, "abstract": summary, "url": url})
    return entries


def collect_scientific(target: int) -> list[dict]:
    print(f"\n[scientific] Collecting from arXiv (target={target})...")
    docs: list[dict] = []
    seen_urls: set[str] = set()

    categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CR", "stat.ML"]
    for cat in categories:
        if len(docs) >= target:
            break
        raw = fetch(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"cat:{cat}", "max_results": 30,
                    "sortBy": "submittedDate", "sortOrder": "descending"},
            sleep_between=3.0,  # arXiv: max 1 req/3s
        )
        if not raw:
            continue

        entries = _parse_arxiv_atom(raw)
        for e in entries:
            if len(docs) >= target:
                break
            if e["url"] in seen_urls:
                continue
            seen_urls.add(e["url"])

            full = f"{e['title']}. {e['abstract']}" if e["title"] else e["abstract"]
            chunk = make_chunk(full)
            if not chunk:
                continue
            ok, _ = passes_quality(chunk)
            if not ok:
                continue

            docs.append({
                "id": f"sci_real_{len(docs)+1:03d}",
                "source": f"arXiv ({cat})",
                "title": e["title"],
                "url": e["url"],
                "text": chunk,
                "word_count": len(chunk.split()),
            })
        if docs:
            print(f"  {len(docs)} collected (cat={cat})...")

    print(f"  Done: {len(docs)}/{target} scientific docs")
    return docs


# ── DevOps: GitHub postmortems ────────────────────────────────────────────────
# Primary: danluu/post-mortems README (large, real incident descriptions)
# Secondary: GitHub code/repo search for postmortem markdown files
#
# Key fix: strip markdown links [text](url) → just text before quality checks,
# and group consecutive list entries into 150-500-word prose chunks.

_DEVOPS_KEYWORDS = re.compile(
    r'\b(incident|outage|postmortem|root.cause|mitigation|on.call|'
    r'deployment|rollback|degradation|downtime|SLA|failure|recovery|'
    r'service.disruption|alert|error.rate|latency)\b',
    re.IGNORECASE,
)

_MD_LINK_RE = re.compile(r'\[([^\]]+)\]\([^)]+\)')   # [text](url) → text
_MD_HEADING_RE = re.compile(r'^#{1,4}\s+', re.MULTILINE)
_MD_BULLET_RE = re.compile(r'^\s*[-*]\s+', re.MULTILINE)


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting, keep prose text."""
    text = _MD_LINK_RE.sub(r'\1', text)           # links → anchor text
    text = re.sub(r'`[^`]+`', '', text)            # inline code
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # bold/italic
    text = _MD_BULLET_RE.sub('', text)             # bullets
    return text


def _group_list_entries(text: str, target_words: int = 250) -> list[str]:
    """
    Group bullet/paragraph entries in markdown text into ~target_words chunks.
    Handles the danluu README format: `[Company](url). One-sentence description.`
    """
    # Strip markdown links to get clean text
    clean = _strip_markdown(text)
    # Split into lines, filter blanks
    lines = [l.strip() for l in clean.split('\n') if l.strip()]

    chunks: list[str] = []
    current_words: list[str] = []

    for line in lines:
        # Skip heading lines
        if _MD_HEADING_RE.match(line):
            if len(current_words) >= MIN_WORDS:
                chunks.append(' '.join(current_words))
            current_words = []
            continue
        line_words = line.split()
        if not line_words:
            continue
        current_words.extend(line_words)
        if len(current_words) >= target_words:
            chunks.append(' '.join(current_words[:MAX_WORDS]))
            current_words = current_words[MAX_WORDS:]

    if len(current_words) >= MIN_WORDS:
        chunks.append(' '.join(current_words[:MAX_WORDS]))

    return chunks


_KNOWN_RAW_URLS = [
    "https://raw.githubusercontent.com/danluu/post-mortems/master/README.md",
]

_KNOWN_REPOS = [
    ("dastergon", "postmortem-templates", "main"),
    ("danluu", "post-mortems", "master"),
]


def collect_devops(target: int) -> list[dict]:
    print(f"\n[devops] Collecting from GitHub postmortem repositories (target={target})...")
    docs: list[dict] = []
    seen_sigs: set[str] = set()

    def try_add_chunks(chunks: list[str], url: str, source: str) -> None:
        for raw_chunk in chunks:
            if len(docs) >= target:
                return
            if not _DEVOPS_KEYWORDS.search(raw_chunk):
                continue
            chunk = make_chunk(scrub_pii(raw_chunk))
            if not chunk:
                continue
            sig = hashlib.md5(chunk[:80].encode()).hexdigest()
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            # Relaxed quality check: skip first-sentence coherence for devops
            words = chunk.split()
            if len(words) < MIN_WORDS:
                continue
            docs.append({
                "id": f"dev_real_{len(docs)+1:03d}",
                "source": source,
                "url": url,
                "text": chunk,
                "word_count": len(chunk.split()),
            })
            if len(docs) % 5 == 0:
                print(f"  {len(docs)} collected...")

    # Strategy 1: danluu/post-mortems README — large markdown with real incidents
    for url in _KNOWN_RAW_URLS:
        if len(docs) >= target:
            break
        raw = fetch(url, sleep_between=1.0)
        if raw:
            # Split on section headings first, then group entries
            sections = re.split(r'\n##\s+', raw)
            for section in sections:
                if len(docs) >= target:
                    break
                chunks = _group_list_entries(section, target_words=250)
                repo_name = url.split('/')[4] + '/' + url.split('/')[5]
                try_add_chunks(chunks, url, f"GitHub/{repo_name}")

    # Strategy 2: GitHub code search for actual postmortem documents
    if len(docs) < target:
        search_raw = fetch(
            "https://api.github.com/search/code",
            params={
                "q": "postmortem \"root cause\" \"impact\" \"mitigation\" language:markdown",
                "per_page": 20,
                "sort": "indexed",
            },
            sleep_between=2.0,
        )
        if search_raw:
            try:
                items = json.loads(search_raw).get("items", [])
                for item in items:
                    if len(docs) >= target:
                        break
                    raw_url = (
                        item.get("html_url", "")
                        .replace("github.com", "raw.githubusercontent.com")
                        .replace("/blob/", "/")
                    )
                    content = fetch(raw_url, sleep_between=1.0)
                    if content:
                        repo = item.get("repository", {}).get("full_name", "?")
                        chunks = _group_list_entries(content)
                        try_add_chunks(chunks, item.get("html_url", ""), f"GitHub/{repo}")
            except Exception:
                pass

    # Strategy 3: Known postmortem template repos — fetch all markdown files
    if len(docs) < target:
        for owner, repo, branch in _KNOWN_REPOS:
            if len(docs) >= target:
                break
            tree_raw = fetch(
                f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}",
                params={"recursive": "1"},
                sleep_between=1.0,
            )
            if not tree_raw:
                continue
            try:
                files = [
                    f for f in json.loads(tree_raw).get("tree", [])
                    if f.get("path", "").endswith(".md")
                ]
            except Exception:
                continue
            for f in files[:15]:
                if len(docs) >= target:
                    break
                raw_url = (
                    f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{f['path']}"
                )
                content = fetch(raw_url, sleep_between=1.0)
                if content:
                    chunks = _group_list_entries(content)
                    try_add_chunks(chunks, raw_url, f"GitHub/{owner}/{repo}")

    # Strategy 4: GitHub repo search
    if len(docs) < target:
        repo_raw = fetch(
            "https://api.github.com/search/repositories",
            params={
                "q": "postmortem incident-report engineering stars:>20",
                "sort": "stars",
                "per_page": 8,
            },
            sleep_between=2.0,
        )
        if repo_raw:
            try:
                repos = json.loads(repo_raw).get("items", [])
                for r in repos:
                    if len(docs) >= target:
                        break
                    full_name = r.get("full_name", "")
                    branch = r.get("default_branch", "main")
                    for fname in ["README.md", "POSTMORTEM.md", "INCIDENTS.md", "postmortem.md"]:
                        if len(docs) >= target:
                            break
                        url = f"https://raw.githubusercontent.com/{full_name}/{branch}/{fname}"
                        content = fetch(url, sleep_between=1.0)
                        if content:
                            chunks = _group_list_entries(content)
                            try_add_chunks(chunks, url, f"GitHub/{full_name}")
            except Exception:
                pass

    print(f"  Done: {len(docs)}/{target} devops docs")
    return docs


# ── Main ───────────────────────────────────────────────────────────────────────

_COLLECTORS = {
    "financial": collect_financial,
    "legal": collect_legal,
    "medical": collect_medical,
    "scientific": collect_scientific,
    "devops": collect_devops,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=list(_COLLECTORS), default=None,
                        help="Collect a single domain only")
    parser.add_argument("--target", type=int, default=20,
                        help="Documents to collect per domain (default: 20)")
    args = parser.parse_args()

    os.makedirs(CONFIG.DATA_DIR, exist_ok=True)
    os.makedirs(CONFIG.CACHE_DIR, exist_ok=True)

    # Load existing corpus
    corpus: dict[str, list] = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            corpus = json.load(f)
        total_existing = sum(len(v) for v in corpus.values())
        print(f"Loaded existing corpus: {total_existing} documents")

    domains = [args.domain] if args.domain else list(_COLLECTORS)

    for domain in domains:
        existing = corpus.get(domain, [])
        if len(existing) >= args.target:
            print(f"\n[{domain}] Already have {len(existing)} docs (>= {args.target}), skipping.")
            continue
        needed = args.target - len(existing)
        new_docs = _COLLECTORS[domain](needed + 5)  # fetch extra for buffer
        corpus[domain] = existing + new_docs[:needed + 5]
        with open(OUTPUT_PATH, "w") as f:
            json.dump(corpus, f, indent=2)
        print(f"  Checkpoint saved.")

    # Final summary
    print(f"\n{'='*62}")
    print("Collection Summary")
    print(f"{'='*62}")
    print(f"{'Domain':<12} | {'Collected':>9} | {'Avg words':>9} | {'Status':>12}")
    print(f"{'─'*62}")
    total = 0
    for domain in _COLLECTORS:
        docs = corpus.get(domain, [])
        n = len(docs)
        total += n
        avg = sum(d["word_count"] for d in docs) / n if docs else 0
        status = "✓ OK" if n >= args.target else f"⚠  need {args.target - n} more"
        print(f"{domain:<12} | {n:>9} | {avg:>9.0f} | {status:>12}")
    print(f"{'─'*62}")
    print(f"{'TOTAL':<12} | {total:>9}")
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Cache at: {CACHE_PATH}")


if __name__ == "__main__":
    main()
