"""Scrape Etihad Hub policy accordions into a local, citable RAG corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests

from .config import PROJECT_ROOT

POLICY_LIBRARY_URL = "https://www.etihadhub.com/en-ae/policy-library"
CATEGORY_NAMES = ["Book", "Change", "GDS best practice", "Fare audit", "GTAA Addendum"]
ACCORDION_RE = re.compile(
    r"var\s+accordiondata\s*=\s*JSON\.parse\(\"(?P<payload>.*?)\"\);",
    re.DOTALL,
)


class PolicyHTMLParser(HTMLParser):
    """Convert policy HTML to readable text while retaining table/list structure."""

    BLOCKS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._in_ignored = 0
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Keep useful separators and links while ignoring scripts and styles."""
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._in_ignored += 1
            return
        if self._in_ignored:
            return
        if tag in self.BLOCKS or tag in {"br", "ul", "ol", "table", "tr"}:
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        """Close visible blocks so adjacent policy sections do not run together."""
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._in_ignored = max(0, self._in_ignored - 1)
            return
        if not self._in_ignored and tag in self.BLOCKS | {"li", "tr", "table"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Collect only text that should be searchable."""
        if not self._in_ignored:
            self.parts.append(data)

    def text(self) -> str:
        """Normalize spacing without flattening paragraphs, lists, or tables."""
        value = "".join(self.parts).replace("\xa0", " ")
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r" *\| *", " | ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip(" \n|")


def _decode_javascript_string(value: str) -> str:
    # The AEM page stores JSON inside a JavaScript string using \xNN escapes.
    json_compatible = re.sub(r"\\x([0-9a-fA-F]{2})", r"\\u00\1", value)
    return json.loads(f'"{json_compatible}"')


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:72]


def _split_text(text: str, chunk_size: int = 1400, overlap: int = 180) -> list[str]:
    """Structure-aware local splitter that keeps paragraphs intact where possible."""
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidates = [paragraph]
        if len(paragraph) > chunk_size:
            candidates = [
                paragraph[start : start + chunk_size]
                for start in range(0, len(paragraph), chunk_size - overlap)
            ]
        for candidate in candidates:
            joined = f"{current}\n\n{candidate}".strip() if current else candidate
            if current and len(joined) > chunk_size:
                chunks.append(current)
                prefix = current[-overlap:].lstrip()
                current = f"{prefix}\n\n{candidate}".strip()
            else:
                current = joined
    if current:
        chunks.append(current)
    return chunks


def extract_policy_records(page_html: str) -> list[dict[str, Any]]:
    """Decode the five expected AEM accordion groups into policy records."""
    groups = list(ACCORDION_RE.finditer(page_html))
    if len(groups) != len(CATEGORY_NAMES):
        raise ValueError(f"Expected 5 policy groups, found {len(groups)}")

    records: list[dict[str, Any]] = []
    for category, match in zip(CATEGORY_NAMES, groups, strict=True):
        parsed = json.loads(_decode_javascript_string(match.group("payload")))
        accordions = parsed["aemdata"]["accordions"]
        for policy in accordions:
            parser = PolicyHTMLParser()
            parser.feed(policy["description"])
            title = re.sub(r"\s+", " ", policy["heading"]).strip()
            records.append(
                {
                    "category": category,
                    "title": title,
                    "last_modified": policy.get("lastmodified"),
                    "html": policy["description"],
                    "text": parser.text(),
                    "links": parser.links,
                }
            )
    return records


def build_corpus(page_html: str, *, scraped_at: str | None = None) -> dict[str, Any]:
    """Chunk records and attach source, date, and hash metadata for retrieval."""
    records = extract_policy_records(page_html)
    chunks: list[dict[str, Any]] = []
    for record in records:
        title_slug = _slug(record["title"])
        category_slug = _slug(record["category"])
        passages = _split_text(f"{record['title']}\n\n{record['text']}")
        for index, passage in enumerate(passages, start=1):
            chunks.append(
                {
                    "id": f"etihad-{category_slug}-{title_slug}-{index:03d}",
                    "topic": "policy",
                    "document_title": record["title"],
                    "version_id": f"etihadhub-{record['last_modified'] or 'undated'}",
                    "is_current": True,
                    "effective_date_status": "published",
                    "section_path": f"{record['category']} > {record['title']}",
                    "snapshot_page": None,
                    "passage": passage,
                    "table_cell_ids": [],
                    "scope": {},
                    "specificity": 1,
                    "source_url": POLICY_LIBRARY_URL,
                    "fixture": False,
                }
            )

    timestamp = scraped_at or datetime.now(UTC).isoformat()
    digest = hashlib.sha256(page_html.encode("utf-8")).hexdigest()
    return {
        "source_url": POLICY_LIBRARY_URL,
        "scraped_at": timestamp,
        "source_sha256": digest,
        "warning": (
            "Local snapshot of Etihad Hub policy-library content. Check the live source for "
            "changes after the scrape timestamp."
        ),
        "record_count": len(records),
        "chunk_count": len(chunks),
        "categories": {
            category: sum(1 for record in records if record["category"] == category)
            for category in CATEGORY_NAMES
        },
        "records": records,
        "chunks": chunks,
    }


def fetch_policy_page(*, verify_tls: bool = True) -> str:
    """Fetch the public policy page, with TLS verification enabled by default."""
    response = requests.get(
        POLICY_LIBRARY_URL,
        headers={"User-Agent": "EtihadPolicyIndexer/1.0"},
        timeout=90,
        verify=verify_tls,
    )
    response.raise_for_status()
    return response.text


def main() -> None:
    """CLI: fetch or read HTML, then write the generated corpus JSON."""
    parser = argparse.ArgumentParser(description="Scrape and index the Etihad Hub policy library")
    parser.add_argument(
        "--html",
        type=Path,
        help="Use a previously downloaded HTML file instead of making a network request",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/corpus/etihad_hub_policy_library.json",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification only when a managed corporate certificate blocks fetching",
    )
    args = parser.parse_args()

    page_html = (
        args.html.read_text(encoding="utf-8")
        if args.html
        else fetch_policy_page(verify_tls=not args.insecure)
    )
    corpus = build_corpus(page_html)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": corpus["record_count"],
                "chunks": corpus["chunk_count"],
                "categories": corpus["categories"],
                "source_sha256": corpus["source_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
