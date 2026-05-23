from pathlib import Path
import unittest

from longevmarker.biomarker_matcher import BiomarkerMatcher, load_lexicon
from longevmarker.dataset_builder import (
    RELATION_TYPE,
    EvidenceRow,
    SeedIntervention,
    aggregate_evidence_rows,
    extract_evidence_rows,
    make_pair_id,
)
from longevmarker.pubmed_client import parse_pubmed_xml

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests' / 'fixtures'
DATA_DIR = ROOT / 'data'


class DatasetBuilderTests(unittest.TestCase):
    def test_parse_pubmed_xml_extracts_articles(self) -> None:
        payload = (FIXTURES / 'pubmed_sample.xml').read_text(encoding='utf-8')
        articles = parse_pubmed_xml(payload)
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0].pmid, '100001')
        self.assertIn('HbA1c', articles[0].title)

    def test_extract_evidence_rows_keeps_intervention_surrogate_pair(self) -> None:
        payload = (FIXTURES / 'pubmed_sample.xml').read_text(encoding='utf-8')
        articles = parse_pubmed_xml(payload)
        matcher = BiomarkerMatcher(load_lexicon(DATA_DIR / 'biomarker_lexicon.csv'))
        seed = SeedIntervention(
            intervention='Metformin',
            mechanism_hint='AMPK activation and glucose homeostasis',
            pubmed_query='metformin aging biomarker',
            match_terms=('metformin',),
            priority='high',
        )

        evidence_rows = extract_evidence_rows(seed, [articles[0]], matcher)

        self.assertTrue(evidence_rows)
        self.assertTrue(all(row.intervention == 'Metformin' for row in evidence_rows))
        self.assertTrue(any(row.surrogate == 'HbA1c' for row in evidence_rows))
        self.assertTrue(all(row.relation_type == RELATION_TYPE for row in evidence_rows))

    def test_aggregate_evidence_rows_groups_by_pair(self) -> None:
        rows = [
            EvidenceRow(
                intervention='Metformin',
                mechanism_hint='AMPK activation',
                surrogate='HbA1c',
                surrogate_category='metabolic',
                pathway='glucose metabolism',
                hallmark='deregulated nutrient sensing',
                relation_type=RELATION_TYPE,
                pmid='1',
                title='Metformin improves HbA1c',
                journal='A',
                year='2022',
                study_type='clinical_trial',
                article_score=0.8,
                surrogate_signal=0.9,
                sample_snippet='Metformin improves HbA1c.',
                doi='x',
            ),
            EvidenceRow(
                intervention='Metformin',
                mechanism_hint='AMPK activation',
                surrogate='HbA1c',
                surrogate_category='metabolic',
                pathway='glucose metabolism',
                hallmark='deregulated nutrient sensing',
                relation_type=RELATION_TYPE,
                pmid='2',
                title='Metformin lowers HbA1c in aging cohort',
                journal='B',
                year='2021',
                study_type='randomized_trial',
                article_score=0.9,
                surrogate_signal=0.95,
                sample_snippet='Metformin lowers HbA1c.',
                doi='y',
            ),
        ]

        aggregated = aggregate_evidence_rows(rows)

        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]['intervention'], 'Metformin')
        self.assertEqual(aggregated[0]['surrogate'], 'HbA1c')
        self.assertEqual(aggregated[0]['pair_id'], make_pair_id('Metformin', 'HbA1c'))
        self.assertEqual(aggregated[0]['article_count'], 2)


if __name__ == '__main__':
    unittest.main()
