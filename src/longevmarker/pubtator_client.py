from __future__ import annotations

import json
import time
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/"


class PubTatorClient:
    def __init__(self, tool: str = "LongevMarkerAI", timeout: int = 30) -> None:
        self.tool = tool
        self.timeout = timeout
        self._last_request = 0.0
        self._min_interval = 0.35

    def fetch_annotations(self, pmids: Iterable[str]) -> dict[str, list[str]]:
        pmid_list = [pmid for pmid in pmids if pmid]
        if not pmid_list:
            return {}
        params = {"pmids": ",".join(pmid_list)}
        payload = self._request("publications/export/biocjson", params)
        return parse_pubtator_annotations_json(payload)

    def _request(self, endpoint: str, params: dict[str, str]) -> str:
        self._throttle()
        url = BASE_URL + endpoint + "?" + urlencode(params)
        request = Request(url, headers={"User-Agent": f"{self.tool}/0.1"})
        with urlopen(request, timeout=self.timeout) as response:
            return response.read().decode("utf-8")

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()


def parse_pubtator_annotations_json(payload: str) -> dict[str, list[str]]:
    data = json.loads(payload)
    if isinstance(data, dict) and "documents" in data:
        documents = data.get("documents", [])
    elif isinstance(data, list):
        documents = data
    else:
        documents = []

    results: dict[str, list[str]] = {}
    for document in documents:
        pmid = str(document.get("id", "")).strip()
        terms: set[str] = set()
        for passage in document.get("passages", []):
            for annotation in passage.get("annotations", []):
                text = str(annotation.get("text", "")).strip()
                if text:
                    terms.add(text)
        if pmid:
            results[pmid] = sorted(terms)
    return results
