from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class BiomarkerEntry:
    canonical_name: str
    synonyms: tuple[str, ...]
    category: str
    pathway: str
    hallmark: str


class BiomarkerMatcher:
    def __init__(self, entries: Iterable[BiomarkerEntry]) -> None:
        self.entries = list(entries)
        self._compiled: list[tuple[BiomarkerEntry, list[re.Pattern[str]], set[str]]] = []
        for entry in self.entries:
            patterns = [_compile_term_pattern(term) for term in entry.synonyms]
            normalized_terms = {_normalize_term(term) for term in entry.synonyms}
            normalized_terms.add(_normalize_term(entry.canonical_name))
            self._compiled.append((entry, patterns, normalized_terms))

    def match_text(self, text: str, annotations: Iterable[str] | None = None) -> list[BiomarkerEntry]:
        annotation_terms = {_normalize_term(term) for term in (annotations or []) if term}
        matches: list[BiomarkerEntry] = []
        for entry, patterns, normalized_terms in self._compiled:
            text_match = any(pattern.search(text) for pattern in patterns)
            annotation_match = bool(annotation_terms & normalized_terms)
            if text_match or annotation_match:
                matches.append(entry)
        return matches


def load_lexicon(path: str | Path) -> list[BiomarkerEntry]:
    rows: list[BiomarkerEntry] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            synonyms = tuple(_split_synonyms(row["synonyms"], row["canonical_name"]))
            rows.append(
                BiomarkerEntry(
                    canonical_name=row["canonical_name"].strip(),
                    synonyms=synonyms,
                    category=row["category"].strip(),
                    pathway=row["pathway"].strip(),
                    hallmark=row["hallmark"].strip(),
                )
            )
    return rows


def _split_synonyms(raw: str, canonical_name: str) -> list[str]:
    items = [item.strip() for item in raw.split("|") if item.strip()]
    if canonical_name not in items:
        items.insert(0, canonical_name)
    return items


def _compile_term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE)


def _normalize_term(text: str) -> str:
    lowered = text.lower().replace("-", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()
