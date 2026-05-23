from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

from .biomarker_matcher import BiomarkerMatcher, load_lexicon
from .pubmed_client import Article, PubMedClient
from .pubtator_client import PubTatorClient

CLINICAL_TERMS = {
    "clinical trial",
    "randomized",
    "placebo",
    "participants",
    "patients",
    "human",
    "older adults",
    "intervention",
}
AGING_TERMS = {
    "aging",
    "ageing",
    "longevity",
    "healthspan",
    "older adults",
    "geroscience",
    "frailty",
}
ENDPOINT_TERMS = {
    "biomarker",
    "endpoint",
    "outcome",
    "surrogate",
    "serum",
    "plasma",
    "measure",
}
CATEGORY_BASE_SURROGATE = {
    "epigenetic aging": 0.80,
    "metabolic": 0.72,
    "inflammation": 0.65,
    "functional": 0.68,
    "senescence": 0.70,
    "mitochondrial": 0.58,
    "stress response": 0.55,
    "cardiovascular": 0.60,
    "body composition": 0.57,
    "growth signaling": 0.63,
    "organ function": 0.52,
    "cellular aging": 0.56,
    "stress axis": 0.54,
}
RELATION_TYPE = "intervention_to_surrogate"


@dataclass(frozen=True)
class SeedIntervention:
    intervention: str
    mechanism_hint: str
    pubmed_query: str
    match_terms: tuple[str, ...]
    priority: str


@dataclass
class EvidenceRow:
    intervention: str
    mechanism_hint: str
    surrogate: str
    surrogate_category: str
    pathway: str
    hallmark: str
    relation_type: str
    pmid: str
    title: str
    journal: str
    year: str
    study_type: str
    article_score: float
    surrogate_signal: float
    sample_snippet: str
    doi: str

    def asdict(self) -> dict[str, object]:
        return {
            "pair_id": make_pair_id(self.intervention, self.surrogate),
            "intervention": self.intervention,
            "mechanism_hint": self.mechanism_hint,
            "surrogate": self.surrogate,
            "surrogate_category": self.surrogate_category,
            "pathway": self.pathway,
            "hallmark": self.hallmark,
            "relation_type": self.relation_type,
            "pmid": self.pmid,
            "title": self.title,
            "journal": self.journal,
            "year": self.year,
            "study_type": self.study_type,
            "article_score": round(self.article_score, 3),
            "surrogate_signal": round(self.surrogate_signal, 3),
            "sample_snippet": self.sample_snippet,
            "doi": self.doi,
        }


def load_seeds(path: str | Path) -> list[SeedIntervention]:
    seeds: list[SeedIntervention] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seeds.append(
                SeedIntervention(
                    intervention=row["intervention"].strip(),
                    mechanism_hint=row["mechanism_hint"].strip(),
                    pubmed_query=row["pubmed_query"].strip(),
                    match_terms=tuple(term.strip().lower() for term in row["match_terms"].split("|") if term.strip()),
                    priority=row["priority"].strip(),
                )
            )
    return seeds


def build_dataset(
    seed_path: str | Path,
    lexicon_path: str | Path,
    raw_output_path: str | Path,
    evidence_output_path: str | Path,
    curated_output_path: str | Path,
    retmax: int = 20,
    email: str | None = None,
    api_key: str | None = None,
    use_pubtator: bool = False,
    mindate: str | None = None,
    maxdate: str | None = None,
) -> tuple[list[EvidenceRow], list[dict[str, object]]]:
    seeds = load_seeds(seed_path)
    matcher = BiomarkerMatcher(load_lexicon(lexicon_path))
    pubmed = PubMedClient(email=email, api_key=api_key)
    pubtator = PubTatorClient() if use_pubtator else None

    articles_for_dump: list[dict[str, object]] = []
    evidence_rows: list[EvidenceRow] = []

    for seed in seeds:
        pmids = pubmed.search(seed.pubmed_query, retmax=retmax, mindate=mindate, maxdate=maxdate)
        articles = pubmed.fetch_details(pmids)
        annotations = pubtator.fetch_annotations(pmids) if pubtator is not None else {}

        for article in articles:
            articles_for_dump.append({"intervention": seed.intervention, **article.asdict()})
        evidence_rows.extend(extract_evidence_rows(seed, articles, matcher, annotations))

    write_jsonl(raw_output_path, articles_for_dump)
    write_csv(evidence_output_path, [row.asdict() for row in evidence_rows])
    curated_rows = aggregate_evidence_rows(evidence_rows)
    write_csv(curated_output_path, curated_rows)
    return evidence_rows, curated_rows


def extract_evidence_rows(
    seed: SeedIntervention,
    articles: Iterable[Article],
    matcher: BiomarkerMatcher,
    annotations_by_pmid: dict[str, list[str]] | None = None,
) -> list[EvidenceRow]:
    results: list[EvidenceRow] = []
    annotations_by_pmid = annotations_by_pmid or {}

    for article in articles:
        combined_text = f"{article.title}. {article.abstract}".strip()
        if not article_mentions_intervention(combined_text, seed.match_terms):
            continue
        matches = matcher.match_text(combined_text, annotations_by_pmid.get(article.pmid, []))
        study_type = infer_study_type(article)
        for entry in matches:
            score = score_article(article, entry.canonical_name, seed.intervention, study_type)
            surrogate_signal = score_surrogate_signal(article, entry.category, study_type)
            results.append(
                EvidenceRow(
                    intervention=seed.intervention,
                    mechanism_hint=seed.mechanism_hint,
                    surrogate=entry.canonical_name,
                    surrogate_category=entry.category,
                    pathway=entry.pathway,
                    hallmark=entry.hallmark,
                    relation_type=RELATION_TYPE,
                    pmid=article.pmid,
                    title=article.title,
                    journal=article.journal,
                    year=article.year,
                    study_type=study_type,
                    article_score=score,
                    surrogate_signal=surrogate_signal,
                    sample_snippet=make_snippet(combined_text, entry.canonical_name),
                    doi=article.doi,
                )
            )
    return results


def infer_study_type(article: Article) -> str:
    publication_types = " ".join(article.publication_types).lower()
    text = f"{article.title} {article.abstract}".lower()
    if "randomized controlled trial" in publication_types or "randomized" in text:
        return "randomized_trial"
    if "clinical trial" in publication_types or "placebo" in text or "participants" in text:
        return "clinical_trial"
    if "meta-analysis" in publication_types or "systematic review" in publication_types:
        return "meta_analysis"
    if "review" in publication_types:
        return "review"
    if any(term in text for term in ["mouse", "mice", "murine", "rat", "rats"]):
        return "preclinical"
    return "observational"


def score_article(article: Article, surrogate: str, intervention: str, study_type: str) -> float:
    title = article.title.lower()
    abstract = article.abstract.lower()
    text = f"{title} {abstract}"
    score = 0.20
    if surrogate.lower() in title:
        score += 0.20
    if intervention.lower() in title:
        score += 0.10
    if contains_any(text, CLINICAL_TERMS):
        score += 0.15
    if contains_any(text, AGING_TERMS):
        score += 0.15
    if contains_any(text, ENDPOINT_TERMS):
        score += 0.10
    if study_type in {"clinical_trial", "randomized_trial", "meta_analysis"}:
        score += 0.15
    if article.year.isdigit() and int(article.year) >= 2015:
        score += 0.05
    return round(min(score, 1.0), 3)


def score_surrogate_signal(article: Article, surrogate_category: str, study_type: str) -> float:
    base = CATEGORY_BASE_SURROGATE.get(surrogate_category, 0.50)
    text = f"{article.title} {article.abstract}".lower()
    bonus = 0.0
    if contains_any(text, ENDPOINT_TERMS):
        bonus += 0.10
    if contains_any(text, AGING_TERMS):
        bonus += 0.05
    if study_type in {"clinical_trial", "randomized_trial", "meta_analysis"}:
        bonus += 0.10
    return round(min(base + bonus, 1.0), 3)


def aggregate_evidence_rows(rows: Iterable[EvidenceRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[EvidenceRow]] = {}
    for row in rows:
        key = (row.intervention, row.surrogate)
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, object]] = []
    for group in grouped.values():
        group.sort(key=lambda item: item.article_score, reverse=True)
        top = group[0]
        article_count = len({row.pmid for row in group})
        clinical_count = len({row.pmid for row in group if row.study_type in {"clinical_trial", "randomized_trial"}})
        mean_article_score = sum(row.article_score for row in group) / len(group)
        mean_surrogate_signal = sum(row.surrogate_signal for row in group) / len(group)
        evidence_score = min(1.0, mean_article_score * 0.70 + min(article_count, 10) / 10 * 0.30)
        surrogate_confidence = min(1.0, mean_surrogate_signal * 0.80 + min(clinical_count, 5) / 5 * 0.20)
        literature_support = min(1.0, math.log1p(article_count) / math.log(9))
        aggregated.append(
            {
                "pair_id": make_pair_id(top.intervention, top.surrogate),
                "intervention": top.intervention,
                "mechanism_hint": top.mechanism_hint,
                "surrogate": top.surrogate,
                "surrogate_category": top.surrogate_category,
                "pathway": top.pathway,
                "hallmark": top.hallmark,
                "relation_type": top.relation_type,
                "evidence_score": round(evidence_score, 3),
                "surrogate_confidence": round(surrogate_confidence, 3),
                "literature_support": round(literature_support, 3),
                "article_count": article_count,
                "supporting_pmids": ";".join(unique_in_order(row.pmid for row in group)),
                "sample_titles": " | ".join(unique_in_order(row.title for row in group)[:3]),
                "sample_snippet": top.sample_snippet,
            }
        )

    aggregated.sort(key=lambda row: (row["intervention"], -float(row["evidence_score"]), -int(row["article_count"]), row["surrogate"]))
    return aggregated


def write_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    row_list = list(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not row_list:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = list(row_list[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def article_mentions_intervention(text: str, match_terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in match_terms)


def make_snippet(text: str, surrogate: str, width: int = 220) -> str:
    lowered = text.lower()
    target = surrogate.lower()
    idx = lowered.find(target)
    if idx == -1:
        return text[:width].strip()
    start = max(0, idx - width // 3)
    end = min(len(text), idx + len(surrogate) + width // 2)
    return text[start:end].strip()


def make_pair_id(intervention: str, surrogate: str) -> str:
    combined = f"{intervention}__{surrogate}".lower()
    combined = re.sub(r"[^a-z0-9]+", "_", combined)
    return combined.strip("_")


def unique_in_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build an intervention-surrogate dataset for longevity interventions.")
    parser.add_argument("--seeds", default=str(root / "data" / "seed_interventions.csv"))
    parser.add_argument("--lexicon", default=str(root / "data" / "biomarker_lexicon.csv"))
    parser.add_argument("--raw-output", default=str(root / "data" / "raw" / "pubmed_articles.jsonl"))
    parser.add_argument("--evidence-output", default=str(root / "data" / "processed" / "article_surrogate_evidence.csv"))
    parser.add_argument("--curated-output", default=str(root / "data" / "processed" / "curated_biomarkers.csv"))
    parser.add_argument("--retmax", type=int, default=20)
    parser.add_argument("--email", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--mindate", default="")
    parser.add_argument("--maxdate", default="")
    parser.add_argument("--use-pubtator", action="store_true")
    args = parser.parse_args(argv)

    try:
        evidence_rows, curated_rows = build_dataset(
            seed_path=args.seeds,
            lexicon_path=args.lexicon,
            raw_output_path=args.raw_output,
            evidence_output_path=args.evidence_output,
            curated_output_path=args.curated_output,
            retmax=args.retmax,
            email=args.email or None,
            api_key=args.api_key or None,
            use_pubtator=args.use_pubtator,
            mindate=args.mindate or None,
            maxdate=args.maxdate or None,
        )
    except Exception as exc:
        print(f"dataset build failed: {exc}", file=sys.stderr)
        return 1

    print(f"built {len(evidence_rows)} intervention-surrogate-article evidence rows")
    print(f"built {len(curated_rows)} intervention-surrogate rows")
    return 0
