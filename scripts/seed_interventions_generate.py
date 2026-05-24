import requests
import pandas as pd
import json
import time
import re
import os
from pathlib import Path
from collections import defaultdict
from Bio import Entrez, Medline

API_KEY = os.getenv("LITELLM_API_KEY", "") # read your API key from the LITELLM_API_KEY environment variable
BASE_URL = os.getenv("LITELLM_BASE_URL", "") # set your base url via the LITELLM_BASE_URL environment variable
MODEL = os.getenv("LITELLM_MODEL", "openai/gpt-5.4")  # LLM model name, e.g., "openai/gpt-4" or "openai/gpt-5.4"
EMAIL = os.getenv("ENTREZ_EMAIL", "") # set your email via the ENTREZ_EMAIL environment variable

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATA_DIR / "seed_interventions.csv"

# ============================================================
# CORE GEROSCIENCE QUERIES
# ============================================================

SEARCH_QUERIES = [
    "geroscience review",
    "aging intervention",
    "longevity intervention",
    "hallmarks of aging therapeutics",
    "senolytics aging",
    "mTOR aging",
    "AMPK aging",
    "NAD+ aging",
    "GLP-1 aging",
    "SGLT2 aging",
    "immune aging intervention",
    "healthy aging clinical trial",
    "geroprotector",
    "aging biomarker intervention"
]

# ============================================================
# MANUAL HIGH-CONFIDENCE SEED
# (critical backbone interventions)
# ============================================================

MANUAL_CORE = {
    "metformin": {
        "mechanism": "AMPK activation and glucose homeostasis",
        "match_terms": "metformin",
        "priority": "high"
    },
    "rapamycin": {
        "mechanism": "mTOR inhibition and autophagy",
        "match_terms": "rapamycin|sirolimus",
        "priority": "high"
    },
    "acarbose": {
        "mechanism": "postprandial glucose modulation",
        "match_terms": "acarbose",
        "priority": "high"
    },
    "semaglutide": {
        "mechanism": "GLP-1 receptor agonism and satiety signaling",
        "match_terms": "semaglutide",
        "priority": "high"
    },
    "liraglutide": {
        "mechanism": "GLP-1 receptor agonism and metabolic regulation",
        "match_terms": "liraglutide",
        "priority": "medium"
    },
    "empagliflozin": {
        "mechanism": "SGLT2 inhibition and glucosuria-driven metabolic unloading",
        "match_terms": "empagliflozin",
        "priority": "medium"
    },
    "dasatinib + quercetin": {
        "mechanism": "senolytic combination targeting senescent cell survival pathways",
        "match_terms": "dasatinib|quercetin",
        "priority": "high"
    },
    "fisetin": {
        "mechanism": "senolytic flavonoid with anti-inflammatory effects",
        "match_terms": "fisetin",
        "priority": "medium"
    },
    "nicotinamide riboside": {
        "mechanism": "NAD+ precursor supporting mitochondrial energetics",
        "match_terms": "nicotinamide riboside",
        "priority": "high"
    },
    "nmn": {
        "mechanism": "NAD+ precursor supporting redox balance",
        "match_terms": "nmn|nicotinamide mononucleotide",
        "priority": "high"
    },
    "spermidine": {
        "mechanism": "autophagy induction and cellular maintenance",
        "match_terms": "spermidine",
        "priority": "medium"
    },
    "resveratrol": {
        "mechanism": "stress-response signaling and metabolic modulation",
        "match_terms": "resveratrol",
        "priority": "medium"
    }
}

# ============================================================
# MECHANISM CLUSTERS
# ============================================================

MECHANISM_MAP = {
    "mTOR": [
        "rapamycin", "sirolimus", "everolimus", "torin"
    ],

    "AMPK": [
        "metformin", "berberine", "biguanide"
    ],

    "GLP1": [
        "semaglutide", "liraglutide", "tirzepatide"
    ],

    "SGLT2": [
        "empagliflozin", "dapagliflozin", "canagliflozin"
    ],

    "senolytic": [
        "dasatinib", "quercetin", "fisetin", "navitoclax"
    ],

    "NAD+": [
        "nmn", "nicotinamide riboside", "nr"
    ],

    "autophagy": [
        "spermidine", "trehalose"
    ],

    "mitochondrial": [
        "urolithin"
    ],

    "anti-inflammatory": [
        "colchicine", "canakinumab"
    ]
}

# ============================================================
# EXCLUSION FILTER
# ============================================================

BLACKLIST = {
    "vitamin",
    "exercise",
    "diet",
    "fasting",
    "protein",
    "fatty acid",
    "pfas",
    "pfos",
    "polymer",
    "nanoparticle",
    "sunscreen",
    "cosmetic",
    "peptide-1",
    "oligopeptide"
}

# ============================================================
# PUBMED
# ============================================================

def search_pubmed(query, n=25):

    h = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=n
    )

    res = Entrez.read(h)
    h.close()

    return res["IdList"]


def fetch_article(pmid):

    h = Entrez.efetch(
        db="pubmed",
        id=pmid,
        rettype="medline",
        retmode="text"
    )

    records = list(Medline.parse(h))
    h.close()

    if not records:
        return None

    r = records[0]

    return {
        "title": r.get("TI", ""),
        "abstract": r.get("AB", "")
    }

# ============================================================
# LLM
# ============================================================

def ask_llm(prompt):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {
            "type": "json_object"
        }
    }

    r = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=180
    )

    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"]

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_name(name):

    name = name.lower().strip()

    name = re.sub(r"\s+", " ", name)

    aliases = {
        "sirolimus": "rapamycin",
        "nr": "nicotinamide riboside",
        "nicotinamide mononucleotide": "nmn"
    }

    return aliases.get(name, name)

# ============================================================
# MECHANISM ASSIGNMENT
# ============================================================

def assign_mechanism(name):

    n = name.lower()

    for mech, terms in MECHANISM_MAP.items():

        for t in terms:

            if t in n:
                return mech

    return "other"

# ============================================================
# PRIORITY SCORING
# ============================================================

def priority_from_score(score):

    if score >= 8:
        return "high"

    if score >= 4:
        return "medium"

    return "low"

# ============================================================
# PUBMED QUERY GENERATOR
# ============================================================

def build_pubmed_query(name):

    aliases = {
        "rapamycin": [
            "rapamycin",
            "sirolimus"
        ],

        "nmn": [
            "NMN",
            "nicotinamide mononucleotide"
        ],

        "nicotinamide riboside": [
            "nicotinamide riboside"
        ]
    }

    terms = aliases.get(name, [name])

    term_block = " OR ".join(
        [f"{t}[Title/Abstract]" for t in terms]
    )

    query = (
        f"({term_block}) AND "
        f"(aging OR longevity OR healthspan "
        f"OR biomarker OR clinical trial)"
    )

    return query

# ============================================================
# MATCH TERMS
# ============================================================

def build_match_terms(name):

    aliases = {
        "rapamycin": "rapamycin|sirolimus",
        "nmn": "nmn|nicotinamide mononucleotide",
        "dasatinib + quercetin": "dasatinib|quercetin"
    }

    return aliases.get(name, name)

# ============================================================
# LLM EXTRACTION
# ============================================================

def extract_candidates(text):

    prompt = f"""
Extract ONLY biologically meaningful geroscience interventions.

INCLUDE:
- FDA-approved drugs
- clinical-stage compounds
- validated geroprotectors
- senolytics
- metabolic interventions
- NAD+ therapeutics
- mTOR / AMPK / GLP1 / SGLT2 agents

EXCLUDE:
- foods
- vitamins
- cosmetics
- polymers
- PFAS
- generic chemicals
- exercise
- diets
- vague intervention classes

Return JSON:

{{
  "interventions": [
    {{
      "name": "",
      "mechanism_hint": "",
      "confidence": 0.0
    }}
  ]
}}

TEXT:
{text}
"""

    try:
        raw = ask_llm(prompt)
        return json.loads(raw)

    except Exception as e:

        print("LLM ERROR:", e)

        return {
            "interventions": []
        }

# ============================================================
# MAIN PIPELINE
# ============================================================

all_interventions = defaultdict(lambda: {
    "mentions": 0,
    "mechanism_hints": [],
    "score": 0
})

# ----------------------------
# add manual core
# ----------------------------

for name, meta in MANUAL_CORE.items():

    all_interventions[name]["mentions"] += 10
    all_interventions[name]["score"] += 10
    all_interventions[name]["mechanism_hints"].append(
        meta["mechanism"]
    )

# ----------------------------
# literature mining
# ----------------------------

for query in SEARCH_QUERIES:

    print(f"\nQUERY: {query}")

    pmids = search_pubmed(query, n=20)

    for pmid in pmids:

        art = fetch_article(pmid)

        if not art:
            continue

        text = (
            art["title"] + "\n" +
            (art["abstract"] or "")
        )

        extracted = extract_candidates(text)

        for item in extracted.get("interventions", []):

            name = normalize_name(
                item.get("name", "")
            )

            if not name:
                continue

            if any(b in name for b in BLACKLIST):
                continue

            conf = float(
                item.get("confidence", 0.5)
            )

            if conf < 0.55:
                continue

            hint = item.get(
                "mechanism_hint",
                ""
            )

            all_interventions[name]["mentions"] += 1
            all_interventions[name]["score"] += conf
            all_interventions[name]["mechanism_hints"].append(
                hint
            )

        time.sleep(0.4)

# ============================================================
# BUILD FINAL DATAFRAME
# ============================================================

rows = []

for name, data in all_interventions.items():

    mentions = data["mentions"]
    score = data["score"]

    if mentions < 2:
        continue

    mechanism_cluster = assign_mechanism(name)

    mechanism_hint = ""

    if data["mechanism_hints"]:
        mechanism_hint = max(
            data["mechanism_hints"],
            key=len
        )

    if mechanism_cluster != "other":
        mechanism_hint = (
            f"{mechanism_cluster}: "
            f"{mechanism_hint}"
        )

    row = {
        "intervention": name.title(),
        "mechanism_hint": mechanism_hint,
        "pubmed_query": build_pubmed_query(name),
        "match_terms": build_match_terms(name),
        "priority": priority_from_score(score)
    }

    rows.append(row)

# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(rows)

df = df.sort_values(
    by=["priority", "intervention"],
    ascending=[True, True]
)

df = df.drop_duplicates(
    subset=["intervention"]
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:", OUTPUT_FILE)
print("Total interventions:", len(df))
