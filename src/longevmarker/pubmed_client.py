from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


@dataclass
class Article:
    pmid: str
    title: str
    abstract: str
    journal: str
    year: str
    publication_types: list[str]
    doi: str = ""

    def asdict(self) -> dict[str, object]:
        return asdict(self)


class PubMedClient:
    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
        tool: str = "LongevMarkerAI",
        timeout: int = 30,
    ) -> None:
        self.email = email or os.environ.get("NCBI_EMAIL", "")
        self.api_key = api_key or os.environ.get("NCBI_API_KEY", "")
        self.tool = tool
        self.timeout = timeout
        self._last_request = 0.0
        self._min_interval = 0.12 if self.api_key else 0.35

    def search(
        self,
        query: str,
        retmax: int = 25,
        mindate: str | None = None,
        maxdate: str | None = None,
    ) -> list[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(retmax),
            "retmode": "xml",
            "sort": "relevance",
        }
        if mindate and maxdate:
            params.update({"datetype": "pdat", "mindate": mindate, "maxdate": maxdate})
        payload = self._request("esearch.fcgi", params)
        root = ET.fromstring(payload)
        return [node.text.strip() for node in root.findall("./IdList/Id") if node.text]

    def fetch_details(self, pmids: Iterable[str], batch_size: int = 50) -> list[Article]:
        pmid_list = [pmid for pmid in pmids if pmid]
        articles: list[Article] = []
        for start in range(0, len(pmid_list), batch_size):
            chunk = pmid_list[start : start + batch_size]
            if not chunk:
                continue
            params = {
                "db": "pubmed",
                "id": ",".join(chunk),
                "rettype": "abstract",
                "retmode": "xml",
            }
            payload = self._request("efetch.fcgi", params)
            articles.extend(parse_pubmed_xml(payload))
        return articles

    def dump_articles(self, articles: Iterable[Article], output_path: str | Path) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for article in articles:
                handle.write(json.dumps(article.asdict(), ensure_ascii=False) + "\n")

    def _request(self, endpoint: str, params: dict[str, str]) -> str:
        self._throttle()
        payload = dict(params)
        payload["tool"] = self.tool
        if self.email:
            payload["email"] = self.email
        if self.api_key:
            payload["api_key"] = self.api_key
        url = BASE_URL + endpoint + "?" + urlencode(payload)
        request = Request(url, headers={"User-Agent": f"{self.tool}/0.1"})
        with urlopen(request, timeout=self.timeout) as response:
            return response.read().decode("utf-8")

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()


def parse_pubmed_xml(payload: str) -> list[Article]:
    root = ET.fromstring(payload)
    articles: list[Article] = []
    for node in root.findall(".//PubmedArticle"):
        citation = node.find("MedlineCitation")
        article_node = citation.find("Article") if citation is not None else None
        if article_node is None:
            continue

        pmid = _get_text(citation.find("PMID"))
        title = _get_text(article_node.find("ArticleTitle"))
        abstract_parts = []
        for abstract_text in article_node.findall("./Abstract/AbstractText"):
            label = abstract_text.attrib.get("Label", "").strip()
            text = _get_text(abstract_text)
            abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(part for part in abstract_parts if part)
        journal = _get_text(article_node.find("./Journal/Title"))
        year = _extract_year(article_node)
        publication_types = [
            _get_text(item) for item in article_node.findall("./PublicationTypeList/PublicationType") if _get_text(item)
        ]
        doi = ""
        for article_id in node.findall("./PubmedData/ArticleIdList/ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                doi = _get_text(article_id)
                break

        articles.append(
            Article(
                pmid=pmid,
                title=title,
                abstract=abstract,
                journal=journal,
                year=year,
                publication_types=publication_types,
                doi=doi,
            )
        )
    return articles


def _extract_year(article_node: ET.Element) -> str:
    year = _get_text(article_node.find("./Journal/JournalIssue/PubDate/Year"))
    if year:
        return year
    medline_date = _get_text(article_node.find("./Journal/JournalIssue/PubDate/MedlineDate"))
    return medline_date[:4] if medline_date else ""


def _get_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()
