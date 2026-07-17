"""Lightweight, inspectable retrieval for CadQuery reference snippets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"
LIGHTWEIGHT_LIBRARY_PATH = KNOWLEDGE_DIR / "cadquery_reference.json"
FULL_LIBRARY_PATH = KNOWLEDGE_DIR
DEFAULT_LIBRARY_PATH = FULL_LIBRARY_PATH
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]+")


@dataclass(frozen=True)
class ReferenceMatch:
    id: str
    title: str
    content: str
    example: str
    source_url: str
    score: int
    category: str = "curated"


def retrieve_references(
    query: str,
    *,
    top_k: int = 3,
    library_path: Path = DEFAULT_LIBRARY_PATH,
) -> list[ReferenceMatch]:
    """Return the most relevant reference entries for an English or Chinese query."""
    entries = _load_entries(library_path)
    query_lower = query.lower()
    query_tokens = _expand_tokens(_TOKEN_RE.findall(query_lower))
    ranked: list[ReferenceMatch] = []

    for entry in entries:
        keywords = [keyword.lower() for keyword in entry["keywords"]]
        search_terms = [term.lower() for term in entry.get("search_terms", [])]
        searchable = " ".join(keywords + search_terms + [entry["title"].lower()])
        searchable_tokens = _expand_tokens(_TOKEN_RE.findall(searchable))
        score = 0
        for keyword in keywords:
            if _keyword_matches(keyword, query_lower, query_tokens):
                score += 5
        score += len(query_tokens & searchable_tokens)

        if score > 0:
            ranked.append(
                ReferenceMatch(
                    id=entry["id"],
                    title=entry["title"],
                    content=entry["content"],
                    example=entry["example"],
                    source_url=entry["source_url"],
                    score=score,
                    category=entry.get("category", "curated"),
                )
            )

    ranked.sort(key=lambda match: (-match.score, match.id))
    if ranked:
        return ranked[: max(1, top_k)]

    fallback_ids = {"workplane_primitives", "parameterization"}
    return [
        ReferenceMatch(
            id=entry["id"],
            title=entry["title"],
            content=entry["content"],
            example=entry["example"],
            source_url=entry["source_url"],
            score=0,
            category=entry.get("category", "curated"),
        )
        for entry in entries
        if entry["id"] in fallback_ids
    ][: max(1, top_k)]


def format_reference_context(
    matches: list[ReferenceMatch],
    *,
    max_chars_per_reference: int = 3500,
) -> str:
    sections = []
    for index, match in enumerate(matches, start=1):
        content = match.content
        if len(content) > max_chars_per_reference:
            content = content[:max_chars_per_reference].rstrip() + "\n[Reference truncated for prompt size]"
        example_section = ""
        if match.example:
            example_section = f"\nExample:\n```python\n{match.example}\n```"
        sections.append(
            f"Reference {index}: {match.title} [{match.category}]\n"
            f"Guidance: {content}"
            f"{example_section}"
        )
    return "\n\n".join(sections)


def write_retrieval_log(
    path: Path,
    *,
    query: str,
    matches: list[ReferenceMatch],
) -> None:
    payload = {
        "query": query,
        "top_k": len(matches),
        "matches": [
            {
                "id": match.id,
                "title": match.title,
                "score": match.score,
                "source_url": match.source_url,
                "category": match.category,
                "content": match.content,
                "example": match.example,
            }
            for match in matches
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_entries(library_path: Path) -> list[dict]:
    if library_path.is_file():
        paths = [library_path]
    else:
        paths = sorted(
            path
            for path in library_path.glob("cadquery_*.json")
            if path.name != "cadquery_reference_manifest.json"
        )

    entries: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        entries.extend(payload)
    return entries


def _keyword_matches(keyword: str, query: str, query_tokens: set[str]) -> bool:
    if not keyword.isascii():
        return keyword in query

    keyword_tokens = _TOKEN_RE.findall(keyword)
    if not keyword_tokens:
        return False
    if len(keyword_tokens) == 1:
        return bool(_expand_tokens(keyword_tokens) & query_tokens)

    pattern = r"(?<![A-Za-z0-9])" + r"[\s_-]+".join(
        re.escape(token) for token in keyword_tokens
    ) + r"(?![A-Za-z0-9])"
    return re.search(pattern, query) is not None


def _expand_tokens(tokens: list[str]) -> set[str]:
    expanded: set[str] = set()
    for token in tokens:
        expanded.add(token)
        if token.endswith("ies") and len(token) > 4:
            expanded.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
        if token.endswith("ed") and len(token) > 4:
            stem = token[:-2]
            expanded.add(stem)
            if len(stem) > 2 and stem[-1] == stem[-2]:
                expanded.add(stem[:-1])
        if token.endswith("ing") and len(token) > 5:
            expanded.add(token[:-3])
    return expanded
