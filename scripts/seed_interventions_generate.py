import json
import os
import re
import time
from collections import Counter, defaultdict

import pandas as pd
import requests
from Bio import Entrez, Medline
from tqdm import tqdm


API_KEY = os.getenv("LITELLM_API_KEY", "") # read your API key from the LITELLM_API_KEY environment variable
BASE_URL = os.getenv("LITELLM_BASE_URL", "") # set your base url via the LITELLM_BASE_URL environment variable
MODEL = os.getenv("LITELLM_MODEL", "openai/gpt-5.4")  # LLM model name, e.g., "openai/gpt-4" or "openai/gpt-5.4"
EMAIL = os.getenv("ENTREZ_EMAIL", "") # set your email via the ENTREZ_EMAIL environment variable

PAPERS_PER_QUERY = int(os.getenv("PAPERS_PER_QUERY", "20"))
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "250"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "3"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
ENTREZ_SLEEP_SECONDS = float(os.getenv("ENTREZ_SLEEP_SECONDS", "0.34"))

Entrez.email = EMAIL


SEED_SEARCH_TERMS = [
    '("aging"[Title/Abstract] OR "ageing"[Title/Abstract] OR "healthy aging"[Title/Abstract] OR "longevity"[Title/Abstract] OR "geroscience"[Title/Abstract]) AND ("drug"[Title/Abstract] OR "pharmacological"[Title/Abstract] OR "intervention"[Title/Abstract] OR "treatment"[Title/Abstract] OR "clinical trial"[Title/Abstract])',
    '("older adults"[Title/Abstract] OR "older people"[Title/Abstract] OR "elderly"[Title/Abstract]) AND ("plasma"[Title/Abstract] OR "serum"[Title/Abstract]) AND ("protein"[Title/Abstract] OR "proteins"[Title/Abstract] OR "proteomics"[Title/Abstract] OR "biomarker"[Title/Abstract])',
    '("aging"[Title/Abstract] OR "longevity"[Title/Abstract]) AND ("plasma biomarker"[Title/Abstract] OR "serum biomarker"[Title/Abstract] OR "proteomics"[Title/Abstract] OR "surrogate endpoint"[Title/Abstract])',
    '("aging"[Title/Abstract] OR "healthy aging"[Title/Abstract]) AND ("randomized"[Title/Abstract] OR "trial"[Title/Abstract] OR "intervention"[Title/Abstract]) AND ("plasma"[Title/Abstract] OR "serum"[Title/Abstract])',
    '("frailty"[Title/Abstract] OR "biological age"[Title/Abstract] OR "functional decline"[Title/Abstract]) AND ("plasma protein"[Title/Abstract] OR "serum protein"[Title/Abstract] OR "circulating protein"[Title/Abstract])',
    '("geroscience"[Title/Abstract] OR "longevity"[Title/Abstract]) AND ("biomarker"[Title/Abstract] OR "surrogate endpoint"[Title/Abstract]) AND ("intervention"[Title/Abstract] OR "trial"[Title/Abstract])',
    '("anti-aging"[Title/Abstract] OR "aging"[Title/Abstract]) AND ("proteomics"[Title/Abstract] OR "plasma proteomics"[Title/Abstract] OR "serum proteomics"[Title/Abstract])',
    '("circulating proteins"[Title/Abstract] OR "plasma proteins"[Title/Abstract] OR "serum proteins"[Title/Abstract]) AND ("aging"[Title/Abstract] OR "older adults"[Title/Abstract]) AND ("drug"[Title/Abstract] OR "treatment"[Title/Abstract] OR "intervention"[Title/Abstract])',
]


GENERIC_INTERVENTION_TERMS = {
    "drug", "drugs", "therapy", "therapies", "treatment", "treatments", "intervention",
    "interventions", "compound", "compounds", "molecule", "molecules", "small molecule",
    "small molecules", "pharmacological intervention", "agent", "agents", "biologic",
    "biologics", "medication", "medications", "supplement", "supplements", "diet",
    "exercise", "lifestyle intervention", "senolytic", "senolytics", "geroprotector",
    "geroprotectors"
}

GENERIC_BIOMARKER_TERMS = {
    "protein", "proteins", "plasma protein", "plasma proteins", "serum protein", "serum proteins",
    "circulating protein", "circulating proteins", "biomarker", "biomarkers", "marker",
    "markers", "inflammation", "aging biomarker", "aging biomarkers", "surrogate endpoint",
    "surrogate endpoints", "proteomics", "plasma proteomics", "serum proteomics", "cytokine",
    "cytokines", "chemokine", "chemokines", "growth factor", "growth factors"
}


ARTICLE_TRIAGE_PROMPT = """
You are screening a biomedical abstract for a dataset about anti-aging interventions and plasma/serum protein surrogate biomarkers.

Return ONLY JSON:
{
  "is_relevant": true,
  "article_type": "trial|intervention_study|observational|review|preclinical|methodology|irrelevant",
  "study_type": "rct|clinical_trial|observational|preclinical|review|unknown",
  "species": "human|mouse|rat|cell|mixed|unknown",
  "human_flag": true,
  "contains_intervention": true,
  "contains_plasma_or_serum": true,
  "contains_protein_biomarker": true,
  "aging_context": true,
  "outcome_context": true,
  "population": "",
  "sample_size_text": ""
}

Rules:
- is_relevant should be true only if the article is meaningfully relevant to aging/longevity/older adults and either intervention evidence or plasma/serum protein biomarker evidence.
- contains_protein_biomarker should be true only if the text mentions specific or candidate proteins/biomarkers, not just generic "biology".
- outcome_context should be true if the text mentions clinical, functional, aging-related, frailty, survival, biological age, morbidity, or similar outcome context.
"""

INTERVENTION_PROMPT = """
Extract ONLY explicitly named pharmacological interventions from the text.

Return ONLY JSON:
{
  "interventions": [
    {
      "name": "",
      "type": "small_molecule|biologic|antibody|peptide|unknown"
    }
  ]
}

Include:
- specific approved drugs
- specific investigational compounds
- specific biologics
- named peptides
- monoclonal antibodies

Exclude:
- generic words like drug, treatment, intervention, therapy, compound
- drug classes unless a specific named agent is present
- diets, exercise, fasting, lifestyle interventions
- diseases, pathways, cell types

Do not infer.
"""

BIOMARKER_PROMPT = """
Extract ONLY explicitly named plasma/serum protein biomarkers from the text.

Return ONLY JSON:
{
  "biomarkers": [
    {
      "name": "",
      "sample_type": "plasma|serum|circulating|unknown"
    }
  ]
}

Include:
- specific named proteins measurable in plasma/serum/circulation
- cytokines, apolipoproteins, growth factors, soluble receptors, named proteomic proteins

Exclude:
- generic terms like biomarker, inflammation, proteomics, protein, marker
- non-protein biomarkers
- pathways, phenotypes, diseases
- tissue-only markers unless clearly circulating/plasma/serum related

Do not infer.
"""

RELATION_PROMPT = """
You are extracting structured evidence for a drug intervention and plasma/serum protein biomarker dataset.


Return ONLY JSON:
{
  "relations": [
    {
      "intervention": "",
      "biomarker": "",
      "effect_direction": "increase|decrease|no_change|mixed|unknown",
      "study_type": "rct|clinical_trial|observational|preclinical|review|unknown",
      "species": "human|mouse|rat|cell|mixed|unknown",
      "sample_type": "plasma|serum|circulating|unknown",
      "dose": "",
      "duration": "",
      "assay": "",
      "statistical_significance": "",
      "outcome_context": "",
      "evidence_sentence": "",
      "confidence": "low|medium|high"
    }
  ]
}

Rules:
- Include only explicit intervention -> biomarker relationships.
- Include only specific named interventions and specific named proteins.
- Exclude generic terms.
- If the paper is a review, extract only if the statement is explicit and specific.
- evidence_sentence should be short and copied or tightly paraphrased from the text.
- Do not infer missing details.
"""

INTERVENTION_PROFILE_PROMPT = """
Create a structured profile for the intervention using ONLY the evidence provided.

INTERVENTION: {intervention}
LINKED BIOMARKERS: {biomarkers}

Return ONLY JSON:
{{
  "intervention": "",
  "synonyms": [],
  "intervention_type": "",
  "biologic_or_small_molecule": "",
  "mechanism_of_action": "",
  "primary_targets": [],
  "drug_class": "",
  "approval_status": "",
  "route_of_administration": "",
  "clinical_uses": [],
  "aging_rationale": "",
  "safety_notes": "",
  "evidence_summary": ""
}}

EVIDENCE:
{evidence}
"""

BIOMARKER_PROFILE_PROMPT = """
Create a structured profile for the plasma/serum protein biomarker using ONLY the evidence provided.

BIOMARKER: {biomarker}
LINKED INTERVENTIONS: {interventions}

Return ONLY JSON:
{{
  "biomarker": "",
  "gene_symbol": "",
  "protein_full_name": "",
  "sample_type": "",
  "biomarker_class": "",
  "biological_function": "",
  "aging_relevance": "",
  "surrogate_endpoint_notes": "",
  "assay_notes": "",
  "confounder_notes": "",
  "evidence_summary": ""
}}

EVIDENCE:
{evidence}
"""


def norm(x):
    return x.lower().strip() if isinstance(x, str) else ""


def clean_text(x):
    if not isinstance(x, str):
        return ""
    return re.sub(r"\s+", " ", x).strip()


def canonicalize_name(name):
    name = clean_text(name)
    if not name:
        return ""
    return name.lower()


def unique_clean_strings(items, banned_terms=None):
    banned_terms = banned_terms or set()
    out = []
    seen = set()

    for item in items or []:
        v = canonicalize_name(item)
        if not v or v in banned_terms:
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)

    return out


def safe_json_load(raw_text):
    if not raw_text:
        return None

    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw_text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def ask_llm(prompt, json_mode=True, retries=LLM_RETRIES):
    if not API_KEY:
        raise ValueError("Set LITELLM_API_KEY environment variable")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for _ in range(retries):
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=LLM_TIMEOUT,
            )

            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]

            print("LLM ERROR:", r.status_code, r.text)
            time.sleep(3)
        except Exception as e:
            print("LLM EXCEPTION:", e)
            time.sleep(3)

    return None


def search_pubmed(query, n=20):
    time.sleep(ENTREZ_SLEEP_SECONDS)
    handle = Entrez.esearch(db="pubmed", term=query, retmax=n)
    res = Entrez.read(handle)
    handle.close()
    return res.get("IdList", [])


def fetch_article(pmid):
    time.sleep(ENTREZ_SLEEP_SECONDS)
    handle = Entrez.efetch(
        db="pubmed",
        id=pmid,
        rettype="medline",
        retmode="text",
    )
    records = list(Medline.parse(handle))
    handle.close()

    if not records:
        return None

    r = records[0]
    return {
        "pmid": str(pmid),
        "title": r.get("TI", "") or "",
        "abstract": r.get("AB", "") or "",
        "journal": r.get("JT", "") or "",
        "year": r.get("DP", "") or "",
        "publication_types": "; ".join(r.get("PT", []) or []),
        "authors": "; ".join(r.get("AU", []) or []),
    }


def article_text(article):
    return clean_text(
        (article.get("title", "") or "") + "\n" + (article.get("abstract", "") or "")
    )


def triage_article(text):
    raw = ask_llm(ARTICLE_TRIAGE_PROMPT + "\n\nTEXT:\n" + text, json_mode=True)
    data = safe_json_load(raw)
    if not data:
        return {
            "is_relevant": False,
            "article_type": "unknown",
            "study_type": "unknown",
            "species": "unknown",
            "human_flag": False,
            "contains_intervention": False,
            "contains_plasma_or_serum": False,
            "contains_protein_biomarker": False,
            "aging_context": False,
            "outcome_context": False,
            "population": "",
            "sample_size_text": "",
        }
    return data


def extract_interventions(text):
    raw = ask_llm(INTERVENTION_PROMPT + "\n\nTEXT:\n" + text, json_mode=True)
    data = safe_json_load(raw)
    if not data:
        return []

    cleaned = []
    seen = set()

    for item in data.get("interventions", []):
        name = canonicalize_name(item.get("name", ""))
        if not name or name in GENERIC_INTERVENTION_TERMS:
            continue
        if name in seen:
            continue
        seen.add(name)
        cleaned.append(
            {
                "name": name,
                "type": clean_text(item.get("type", "")) or "unknown",
            }
        )

    return cleaned


def extract_biomarkers(text):
    raw = ask_llm(BIOMARKER_PROMPT + "\n\nTEXT:\n" + text, json_mode=True)
    data = safe_json_load(raw)
    if not data:
        return []

    cleaned = []
    seen = set()

    for item in data.get("biomarkers", []):
        name = canonicalize_name(item.get("name", ""))
        if not name or name in GENERIC_BIOMARKER_TERMS:
            continue
        if name in seen:
            continue
        seen.add(name)
        cleaned.append(
            {
                "name": name,
                "sample_type": clean_text(item.get("sample_type", "")) or "unknown",
            }
        )

    return cleaned


def extract_relations(text):
    raw = ask_llm(RELATION_PROMPT + "\n\nTEXT:\n" + text, json_mode=True)
    data = safe_json_load(raw)
    if not data:
        return []

    cleaned = []
    seen = set()

    for rel in data.get("relations", []):
        intervention = canonicalize_name(rel.get("intervention", ""))
        biomarker = canonicalize_name(rel.get("biomarker", ""))

        if not intervention or intervention in GENERIC_INTERVENTION_TERMS:
            continue
        if not biomarker or biomarker in GENERIC_BIOMARKER_TERMS:
            continue

        row = {
            "intervention": intervention,
            "biomarker": biomarker,
            "effect_direction": clean_text(rel.get("effect_direction", "")) or "unknown",
            "study_type": clean_text(rel.get("study_type", "")) or "unknown",
            "species": clean_text(rel.get("species", "")) or "unknown",
            "sample_type": clean_text(rel.get("sample_type", "")) or "unknown",
            "dose": clean_text(rel.get("dose", "")),
            "duration": clean_text(rel.get("duration", "")),
            "assay": clean_text(rel.get("assay", "")),
            "statistical_significance": clean_text(rel.get("statistical_significance", "")),
            "outcome_context": clean_text(rel.get("outcome_context", "")),
            "evidence_sentence": clean_text(rel.get("evidence_sentence", "")),
            "confidence": clean_text(rel.get("confidence", "")) or "low",
        }

        key = (
            row["intervention"],
            row["biomarker"],
            row["effect_direction"],
            row["evidence_sentence"],
        )
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(row)

    return cleaned


def make_evidence_block(articles, max_articles=8, max_chars=12000):
    chunks = []
    total = 0

    for art in articles[:max_articles]:
        chunk = (
            f"PMID: {art.get('pmid', '')}\n"
            f"TITLE: {art.get('title', '')}\n"
            f"ABSTRACT: {art.get('abstract', '')}\n"
        )
        if total + len(chunk) > max_chars:
            break
        chunks.append(chunk)
        total += len(chunk)

    return "\n---\n".join(chunks)


def build_intervention_profile(intervention, linked_biomarkers, evidence_articles):
    evidence = make_evidence_block(evidence_articles)
    prompt = INTERVENTION_PROFILE_PROMPT.format(
        intervention=intervention,
        biomarkers=", ".join(sorted(linked_biomarkers)),
        evidence=evidence,
    )
    raw = ask_llm(prompt, json_mode=True)
    data = safe_json_load(raw)

    if not data:
        return {
            "intervention": intervention,
            "synonyms": [],
            "intervention_type": "",
            "biologic_or_small_molecule": "",
            "mechanism_of_action": "",
            "primary_targets": [],
            "drug_class": "",
            "approval_status": "",
            "route_of_administration": "",
            "clinical_uses": [],
            "aging_rationale": "",
            "safety_notes": "",
            "evidence_summary": "",
        }

    data["intervention"] = data.get("intervention") or intervention
    return data


def build_biomarker_profile(biomarker, linked_interventions, evidence_articles):
    evidence = make_evidence_block(evidence_articles)
    prompt = BIOMARKER_PROFILE_PROMPT.format(
        biomarker=biomarker,
        interventions=", ".join(sorted(linked_interventions)),
        evidence=evidence,
    )
    raw = ask_llm(prompt, json_mode=True)
    data = safe_json_load(raw)

    if not data:
        return {
            "biomarker": biomarker,
            "gene_symbol": "",
            "protein_full_name": "",
            "sample_type": "",
            "biomarker_class": "",
            "biological_function": "",
            "aging_relevance": "",
            "surrogate_endpoint_notes": "",
            "assay_notes": "",
            "confounder_notes": "",
            "evidence_summary": "",
        }

    data["biomarker"] = data.get("biomarker") or biomarker
    return data


def list_to_text(x):
    if isinstance(x, list):
        return "; ".join(str(i) for i in x)
    if x is None:
        return ""
    return str(x)


def collect_corpus(seed_terms, papers_per_query=PAPERS_PER_QUERY, max_articles=MAX_ARTICLES):
    all_pmids = []
    pmid_to_query = defaultdict(list)

    for term in seed_terms:
        print("SEARCH:", term)
        pmids = search_pubmed(term, n=papers_per_query)
        for pmid in pmids:
            all_pmids.append(str(pmid))
            pmid_to_query[str(pmid)].append(term)

    unique_pmids = []
    seen = set()
    for pmid in all_pmids:
        if pmid not in seen:
            seen.add(pmid)
            unique_pmids.append(pmid)

    unique_pmids = unique_pmids[:max_articles]

    papers = []
    intervention_mentions = []
    biomarker_mentions = []
    relation_rows = []

    intervention_articles = defaultdict(list)
    biomarker_articles = defaultdict(list)

    for pmid in tqdm(unique_pmids, desc="Fetch + extract"):
        art = fetch_article(pmid)
        if not art:
            continue

        text = article_text(art)
        if not text:
            continue

        triage = triage_article(text)

        paper_row = {
            "paper_id": f"PMID:{art['pmid']}",
            "pmid": art["pmid"],
            "title": art["title"],
            "abstract": art["abstract"],
            "journal": art["journal"],
            "year": art["year"],
            "publication_types": art["publication_types"],
            "seed_queries": " || ".join(pmid_to_query.get(art["pmid"], [])),
            "is_relevant": triage.get("is_relevant", False),
            "article_type": triage.get("article_type", "unknown"),
            "study_type": triage.get("study_type", "unknown"),
            "species": triage.get("species", "unknown"),
            "human_flag": triage.get("human_flag", False),
            "contains_intervention": triage.get("contains_intervention", False),
            "contains_plasma_or_serum": triage.get("contains_plasma_or_serum", False),
            "contains_protein_biomarker": triage.get("contains_protein_biomarker", False),
            "aging_context": triage.get("aging_context", False),
            "outcome_context": triage.get("outcome_context", False),
            "population": triage.get("population", ""),
            "sample_size_text": triage.get("sample_size_text", ""),
        }
        papers.append(paper_row)

        if not triage.get("is_relevant", False):
            continue

        interventions = extract_interventions(text)
        biomarkers = extract_biomarkers(text)
        relations = extract_relations(text)

        for iv in interventions:
            intervention_mentions.append(
                {
                    "paper_id": paper_row["paper_id"],
                    "pmid": art["pmid"],
                    "intervention": iv["name"],
                    "intervention_type": iv["type"],
                    "title": art["title"],
                    "journal": art["journal"],
                    "year": art["year"],
                }
            )
            intervention_articles[iv["name"]].append(art)

        for bm in biomarkers:
            biomarker_mentions.append(
                {
                    "paper_id": paper_row["paper_id"],
                    "pmid": art["pmid"],
                    "biomarker": bm["name"],
                    "sample_type": bm["sample_type"],
                    "title": art["title"],
                    "journal": art["journal"],
                    "year": art["year"],
                }
            )
            biomarker_articles[bm["name"]].append(art)

        for rel in relations:
            relation_rows.append(
                {
                    "relation_id": f"{art['pmid']}::{rel['intervention']}::{rel['biomarker']}::{len(relation_rows)+1}",
                    "paper_id": paper_row["paper_id"],
                    "pmid": art["pmid"],
                    "title": art["title"],
                    "journal": art["journal"],
                    "year": art["year"],
                    "intervention": rel["intervention"],
                    "biomarker": rel["biomarker"],
                    "effect_direction": rel["effect_direction"],
                    "study_type": rel["study_type"] if rel["study_type"] != "unknown" else triage.get("study_type", "unknown"),
                    "species": rel["species"] if rel["species"] != "unknown" else triage.get("species", "unknown"),
                    "sample_type": rel["sample_type"],
                    "dose": rel["dose"],
                    "duration": rel["duration"],
                    "assay": rel["assay"],
                    "statistical_significance": rel["statistical_significance"],
                    "outcome_context": rel["outcome_context"],
                    "evidence_sentence": rel["evidence_sentence"],
                    "confidence": rel["confidence"],
                    "article_type": triage.get("article_type", "unknown"),
                    "human_flag": triage.get("human_flag", False),
                    "population": triage.get("population", ""),
                    "sample_size_text": triage.get("sample_size_text", ""),
                    "seed_queries": paper_row["seed_queries"],
                }
            )
            intervention_articles[rel["intervention"]].append(art)
            biomarker_articles[rel["biomarker"]].append(art)

    df_papers = pd.DataFrame(papers).drop_duplicates(subset=["pmid"])
    df_intervention_mentions = pd.DataFrame(intervention_mentions)
    df_biomarker_mentions = pd.DataFrame(biomarker_mentions)
    df_relations = pd.DataFrame(relation_rows)

    if len(df_intervention_mentions) > 0:
        df_intervention_mentions = df_intervention_mentions.drop_duplicates(
            subset=["pmid", "intervention"]
        )

    if len(df_biomarker_mentions) > 0:
        df_biomarker_mentions = df_biomarker_mentions.drop_duplicates(
            subset=["pmid", "biomarker"]
        )

    if len(df_relations) > 0:
        df_relations = df_relations.drop_duplicates(
            subset=[
                "pmid",
                "intervention",
                "biomarker",
                "effect_direction",
                "evidence_sentence",
            ]
        )

    return (
        df_papers,
        df_intervention_mentions,
        df_biomarker_mentions,
        df_relations,
        intervention_articles,
        biomarker_articles,
    )


def dedupe_articles_by_pmid(articles):
    out = []
    seen = set()

    for art in articles or []:
        pmid = str(art.get("pmid", ""))
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        out.append(art)

    return out


def build_entity_tables(df_intervention_mentions, df_biomarker_mentions, df_relations):
    intervention_names = set()
    biomarker_names = set()

    if len(df_intervention_mentions) > 0:
        intervention_names.update(df_intervention_mentions["intervention"].dropna().tolist())
    if len(df_relations) > 0:
        intervention_names.update(df_relations["intervention"].dropna().tolist())

    if len(df_biomarker_mentions) > 0:
        biomarker_names.update(df_biomarker_mentions["biomarker"].dropna().tolist())
    if len(df_relations) > 0:
        biomarker_names.update(df_relations["biomarker"].dropna().tolist())

    intervention_names = sorted(canonicalize_name(x) for x in intervention_names if x)
    biomarker_names = sorted(canonicalize_name(x) for x in biomarker_names if x)

    intervention_rows = []
    intervention_id_map = {}

    for idx, name in enumerate(intervention_names, start=1):
        subset_mentions = (
            df_intervention_mentions[df_intervention_mentions["intervention"] == name]
            if len(df_intervention_mentions) > 0 else pd.DataFrame()
        )
        subset_rel = (
            df_relations[df_relations["intervention"] == name]
            if len(df_relations) > 0 else pd.DataFrame()
        )

        mention_count = len(subset_mentions)
        paper_count = len(set(subset_mentions["pmid"].tolist())) if len(subset_mentions) > 0 else 0
        linked_biomarkers = sorted(set(subset_rel["biomarker"].tolist())) if len(subset_rel) > 0 else []

        dominant_type = "unknown"
        if len(subset_mentions) > 0 and "intervention_type" in subset_mentions.columns:
            vals = [x for x in subset_mentions["intervention_type"].tolist() if x]
            if vals:
                dominant_type = Counter(vals).most_common(1)[0][0]

        intervention_id = f"INTV_{idx:05d}"
        intervention_id_map[name] = intervention_id

        intervention_rows.append(
            {
                "intervention_id": intervention_id,
                "canonical_name": name,
                "dominant_intervention_type": dominant_type,
                "mention_count": mention_count,
                "paper_count": paper_count,
                "linked_biomarker_count": len(linked_biomarkers),
                "linked_biomarkers": "; ".join(linked_biomarkers),
            }
        )

    biomarker_rows = []
    biomarker_id_map = {}

    for idx, name in enumerate(biomarker_names, start=1):
        subset_mentions = (
            df_biomarker_mentions[df_biomarker_mentions["biomarker"] == name]
            if len(df_biomarker_mentions) > 0 else pd.DataFrame()
        )
        subset_rel = (
            df_relations[df_relations["biomarker"] == name]
            if len(df_relations) > 0 else pd.DataFrame()
        )

        mention_count = len(subset_mentions)
        paper_count = len(set(subset_mentions["pmid"].tolist())) if len(subset_mentions) > 0 else 0
        linked_interventions = sorted(set(subset_rel["intervention"].tolist())) if len(subset_rel) > 0 else []

        dominant_sample_type = "unknown"
        if len(subset_mentions) > 0 and "sample_type" in subset_mentions.columns:
            vals = [x for x in subset_mentions["sample_type"].tolist() if x]
            if vals:
                dominant_sample_type = Counter(vals).most_common(1)[0][0]

        biomarker_id = f"BMK_{idx:05d}"
        biomarker_id_map[name] = biomarker_id

        biomarker_rows.append(
            {
                "biomarker_id": biomarker_id,
                "canonical_name": name,
                "dominant_sample_type": dominant_sample_type,
                "mention_count": mention_count,
                "paper_count": paper_count,
                "linked_intervention_count": len(linked_interventions),
                "linked_interventions": "; ".join(linked_interventions),
            }
        )

    df_interventions = pd.DataFrame(intervention_rows)
    df_biomarkers = pd.DataFrame(biomarker_rows)

    if len(df_relations) > 0:
        df_relations = df_relations.copy()
        df_relations["intervention_id"] = df_relations["intervention"].map(intervention_id_map)
        df_relations["biomarker_id"] = df_relations["biomarker"].map(biomarker_id_map)

    return df_interventions, df_biomarkers, df_relations, intervention_id_map, biomarker_id_map


def build_intervention_profiles(df_interventions, df_relations, intervention_articles):
    rows = []
    json_rows = []

    if len(df_interventions) == 0:
        return pd.DataFrame(), []

    for _, row in tqdm(df_interventions.iterrows(), total=len(df_interventions), desc="Intervention profiles"):
        intervention = row["canonical_name"]

        subset_rel = (
            df_relations[df_relations["intervention"] == intervention]
            if len(df_relations) > 0 else pd.DataFrame()
        )
        linked_biomarkers = sorted(set(subset_rel["biomarker"].tolist())) if len(subset_rel) > 0 else []
        evidence_articles = dedupe_articles_by_pmid(intervention_articles.get(intervention, []))

        profile = build_intervention_profile(intervention, linked_biomarkers, evidence_articles)
        profile["intervention_id"] = row["intervention_id"]
        profile["linked_biomarkers"] = linked_biomarkers
        profile["evidence_count"] = len(evidence_articles)

        rows.append(
            {
                "intervention_id": row["intervention_id"],
                "intervention": profile.get("intervention", intervention),
                "synonyms": list_to_text(profile.get("synonyms", [])),
                "intervention_type": profile.get("intervention_type", ""),
                "biologic_or_small_molecule": profile.get("biologic_or_small_molecule", ""),
                "mechanism_of_action": profile.get("mechanism_of_action", ""),
                "primary_targets": list_to_text(profile.get("primary_targets", [])),
                "drug_class": profile.get("drug_class", ""),
                "approval_status": profile.get("approval_status", ""),
                "route_of_administration": profile.get("route_of_administration", ""),
                "clinical_uses": list_to_text(profile.get("clinical_uses", [])),
                "aging_rationale": profile.get("aging_rationale", ""),
                "safety_notes": profile.get("safety_notes", ""),
                "linked_biomarkers": list_to_text(linked_biomarkers),
                "evidence_summary": profile.get("evidence_summary", ""),
                "evidence_count": len(evidence_articles),
            }
        )
        json_rows.append(profile)

    return pd.DataFrame(rows), json_rows


def build_biomarker_profiles(df_biomarkers, df_relations, biomarker_articles):
    rows = []
    json_rows = []

    if len(df_biomarkers) == 0:
        return pd.DataFrame(), []

    for _, row in tqdm(df_biomarkers.iterrows(), total=len(df_biomarkers), desc="Biomarker profiles"):
        biomarker = row["canonical_name"]

        subset_rel = (
            df_relations[df_relations["biomarker"] == biomarker]
            if len(df_relations) > 0 else pd.DataFrame()
        )
        linked_interventions = sorted(set(subset_rel["intervention"].tolist())) if len(subset_rel) > 0 else []
        evidence_articles = dedupe_articles_by_pmid(biomarker_articles.get(biomarker, []))

        profile = build_biomarker_profile(biomarker, linked_interventions, evidence_articles)
        profile["biomarker_id"] = row["biomarker_id"]
        profile["linked_interventions"] = linked_interventions
        profile["evidence_count"] = len(evidence_articles)

        rows.append(
            {
                "biomarker_id": row["biomarker_id"],
                "biomarker": profile.get("biomarker", biomarker),
                "gene_symbol": profile.get("gene_symbol", ""),
                "protein_full_name": profile.get("protein_full_name", ""),
                "sample_type": profile.get("sample_type", ""),
                "biomarker_class": profile.get("biomarker_class", ""),
                "biological_function": profile.get("biological_function", ""),
                "aging_relevance": profile.get("aging_relevance", ""),
                "surrogate_endpoint_notes": profile.get("surrogate_endpoint_notes", ""),
                "assay_notes": profile.get("assay_notes", ""),
                "confounder_notes": profile.get("confounder_notes", ""),
                "linked_interventions": list_to_text(linked_interventions),
                "evidence_summary": profile.get("evidence_summary", ""),
                "evidence_count": len(evidence_articles),
            }
        )
        json_rows.append(profile)

    return pd.DataFrame(rows), json_rows


def confidence_score(series):
    mapping = {"low": 0.33, "medium": 0.66, "high": 1.0}
    vals = [mapping.get(str(x).lower(), 0.0) for x in series if str(x).strip()]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def consistency_score(directions):
    vals = [str(x).lower() for x in directions if str(x).lower() not in {"", "unknown", "mixed"}]
    if not vals:
        return 0.0
    cnt = Counter(vals)
    top = cnt.most_common(1)[0][1]
    return top / len(vals)


def build_surrogate_candidates(df_relations):
    if len(df_relations) == 0:
        return pd.DataFrame()

    rows = []

    grouped = df_relations.groupby(["intervention_id", "biomarker_id", "intervention", "biomarker"], dropna=False)

    for (intervention_id, biomarker_id, intervention, biomarker), g in grouped:
        n_papers = len(set(g["pmid"].tolist()))
        n_human = int(g["human_flag"].fillna(False).sum())
        n_rct = int((g["study_type"] == "rct").sum())
        n_clinical = int(g["study_type"].isin(["rct", "clinical_trial"]).sum())
        outcome_nonempty = int(g["outcome_context"].fillna("").astype(str).str.len().gt(0).sum())
        plasma_serum_hits = int(g["sample_type"].isin(["plasma", "serum", "circulating"]).sum())

        direction_consistency = consistency_score(g["effect_direction"].tolist())
        avg_conf = confidence_score(g["confidence"].tolist())
        human_evidence_score = min(1.0, (n_human + n_rct + n_clinical) / max(1, n_papers * 2))
        outcome_link_score = min(1.0, outcome_nonempty / max(1, len(g)))
        plasma_specificity_score = min(1.0, plasma_serum_hits / max(1, len(g)))

        overall = round(
            (
                0.25 * direction_consistency +
                0.20 * avg_conf +
                0.20 * human_evidence_score +
                0.20 * outcome_link_score +
                0.15 * plasma_specificity_score
            ),
            4,
        )

        rows.append(
            {
                "intervention_id": intervention_id,
                "biomarker_id": biomarker_id,
                "intervention": intervention,
                "biomarker": biomarker,
                "n_relation_rows": len(g),
                "n_papers": n_papers,
                "n_human_studies": n_human,
                "n_rct_rows": n_rct,
                "n_clinical_rows": n_clinical,
                "direction_consistency_score": round(direction_consistency, 4),
                "average_confidence_score": round(avg_conf, 4),
                "human_evidence_score": round(human_evidence_score, 4),
                "outcome_link_score": round(outcome_link_score, 4),
                "plasma_specificity_score": round(plasma_specificity_score, 4),
                "overall_surrogate_score": overall,
                "dominant_effect_direction": Counter(
                    [str(x).lower() for x in g["effect_direction"].tolist() if str(x).strip()]
                ).most_common(1)[0][0] if len(g) > 0 else "unknown",
                "example_outcome_context": next(
                    (x for x in g["outcome_context"].fillna("").tolist() if str(x).strip()),
                    "",
                ),
                "example_evidence_sentence": next(
                    (x for x in g["evidence_sentence"].fillna("").tolist() if str(x).strip()),
                    "",
                ),
            }
        )

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(
            by=["overall_surrogate_score", "n_papers", "n_human_studies"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    return df


def main():
    print("Collecting corpus...")
    (
        df_papers,
        df_intervention_mentions,
        df_biomarker_mentions,
        df_relations,
        intervention_articles,
        biomarker_articles,
    ) = collect_corpus(
        SEED_SEARCH_TERMS,
        papers_per_query=PAPERS_PER_QUERY,
        max_articles=MAX_ARTICLES,
    )

    print(f"Collected papers: {len(df_papers)}")
    print(f"Intervention mentions: {len(df_intervention_mentions)}")
    print(f"Biomarker mentions: {len(df_biomarker_mentions)}")
    print(f"Raw relations: {len(df_relations)}")

    df_papers.to_csv("papers.csv", index=False)

    if len(df_intervention_mentions) == 0:
        pd.DataFrame(columns=[
            "paper_id", "pmid", "intervention", "intervention_type", "title", "journal", "year"
        ]).to_csv("intervention_mentions.csv", index=False)
    else:
        df_intervention_mentions.to_csv("intervention_mentions.csv", index=False)

    if len(df_biomarker_mentions) == 0:
        pd.DataFrame(columns=[
            "paper_id", "pmid", "biomarker", "sample_type", "title", "journal", "year"
        ]).to_csv("biomarker_mentions.csv", index=False)
    else:
        df_biomarker_mentions.to_csv("biomarker_mentions.csv", index=False)

    if len(df_relations) == 0:
        print("No intervention-biomarker relations found.")

        pd.DataFrame().to_csv("interventions.csv", index=False)
        pd.DataFrame().to_csv("biomarkers.csv", index=False)
        pd.DataFrame().to_csv("intervention_biomarker_relations.csv", index=False)
        pd.DataFrame().to_csv("intervention_profiles.csv", index=False)
        pd.DataFrame().to_csv("biomarker_profiles.csv", index=False)
        pd.DataFrame().to_csv("surrogate_candidates.csv", index=False)

        with open("intervention_profiles.json", "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

        with open("biomarker_profiles.json", "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

        return

    print("Building entity tables...")
    (
        df_interventions,
        df_biomarkers,
        df_relations,
        intervention_id_map,
        biomarker_id_map,
    ) = build_entity_tables(
        df_intervention_mentions,
        df_biomarker_mentions,
        df_relations,
    )

    if len(df_relations) > 0:
        df_relations = df_relations.drop_duplicates(
            subset=[
                "pmid",
                "intervention",
                "biomarker",
                "effect_direction",
                "evidence_sentence",
            ]
        ).reset_index(drop=True)

    df_interventions.to_csv("interventions.csv", index=False)
    df_biomarkers.to_csv("biomarkers.csv", index=False)
    df_relations.to_csv("intervention_biomarker_relations.csv", index=False)

    print("Building intervention profiles...")
    df_intervention_profiles, intervention_profiles_json = build_intervention_profiles(
        df_interventions,
        df_relations,
        intervention_articles,
    )

    print("Building biomarker profiles...")
    df_biomarker_profiles, biomarker_profiles_json = build_biomarker_profiles(
        df_biomarkers,
        df_relations,
        biomarker_articles,
    )

    df_intervention_profiles.to_csv("intervention_profiles.csv", index=False)
    df_biomarker_profiles.to_csv("biomarker_profiles.csv", index=False)

    with open("intervention_profiles.json", "w", encoding="utf-8") as f:
        json.dump(intervention_profiles_json, f, ensure_ascii=False, indent=2)

    with open("biomarker_profiles.json", "w", encoding="utf-8") as f:
        json.dump(biomarker_profiles_json, f, ensure_ascii=False, indent=2)

    print("Scoring surrogate candidates...")
    df_surrogate_candidates = build_surrogate_candidates(df_relations)
    df_surrogate_candidates.to_csv("surrogate_candidates.csv", index=False)

    print(f"Saved {len(df_papers)} rows to papers.csv")
    print(f"Saved {len(df_intervention_mentions)} rows to intervention_mentions.csv")
    print(f"Saved {len(df_biomarker_mentions)} rows to biomarker_mentions.csv")
    print(f"Saved {len(df_interventions)} rows to interventions.csv")
    print(f"Saved {len(df_biomarkers)} rows to biomarkers.csv")
    print(f"Saved {len(df_relations)} rows to intervention_biomarker_relations.csv")
    print(f"Saved {len(df_intervention_profiles)} rows to intervention_profiles.csv")
    print(f"Saved {len(df_biomarker_profiles)} rows to biomarker_profiles.csv")
    print(f"Saved {len(df_surrogate_candidates)} rows to surrogate_candidates.csv")
    print("Saved JSON profiles to intervention_profiles.json and biomarker_profiles.json")


if __name__ == "__main__":
    main()