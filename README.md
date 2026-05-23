# LongevMarker AI

Drug-only biomarker recommendation MVP for longevity interventions, with a literature-backed curated dataset and embedding-ready benchmark package.

## Core rule

The central unit is an explicit `intervention -> surrogate` pair.

## Current scope

- Only chemical interventions and drug-like compounds
- Curated `intervention -> surrogate` runtime table
- Molecular inputs for intervention-side encoders
- Text inputs for biomarker and pair-side encoders
- Retrieval benchmark package for comparing molecular encoders

## Main datasets

### Runtime curated table
- `data/processed/curated_biomarkers.csv`
- 70 curated `intervention -> surrogate` rows
- 12 chemical interventions

### Chemical registry
- `data/processed/compound_registry.csv`
- `data/processed/compound_components.csv`
- PubChem-backed SMILES and component-level decomposition

### Text embedding tables
- `data/embedding/intervention_texts.csv`
- `data/embedding/surrogate_texts.csv`
- `data/embedding/biomarker_semantic_texts.csv`
- `data/embedding/pair_texts.csv`
- `data/embedding/retrieval_corpus.csv`
- `data/embedding/queries.csv`
- `data/embedding/query_qrels.csv`
- `data/embedding/contrastive_triplets.csv`
- `data/embedding/pair_examples.csv`

### Molecular benchmark tables
- `data/embedding/molecular_interventions.csv`
- `data/embedding/molecular_pair_qrels.csv`
- `data/embedding/molecular_surrogate_qrels.csv`
- `data/embedding/molecular_biomarker_semantic_qrels.csv`
- `data/embedding/molecule_to_pair_triplets.csv`
- `data/embedding/molecule_to_surrogate_triplets.csv`
- `data/embedding/molecule_to_biomarker_semantic_triplets.csv`
- `data/embedding/encoder_candidates.csv`
- `data/embedding/model_comparison_runs.csv`
- `data/embedding/molecular_model_registry.csv`
- `data/embedding/text_model_registry.csv`

## Model strategy

### Intervention side
Use molecular encoders on chemical structures.

- `ChemBERT` on `canonical_smiles` or `connectivity_smiles`
- `MolFormer` on `canonical_smiles` or `connectivity_smiles`
- `SELFormer` on `selfies_sequence`

### Biomarker side
Use text encoders on biomarker semantics.

The recommended right tower is `biomarker_semantic_texts.csv`.

Input text should include:
- biomarker name
- category
- pathways
- hallmarks
- linked interventions
- aggregated mechanism-of-action contexts
- normalized mechanism themes
- high-signal recommendation reasons

### Retrieval tasks
1. `molecule -> biomarker_semantic_text`
2. `molecule -> pair_text`
3. `molecule -> surrogate_text`

## Recommended benchmark models

### Molecular encoders
- `seyonec/ChemBERTa-zinc-base-v1`
- `ibm-research/MoLFormer-XL-both-10pct`
- `HUBioDataLab/SELFormer`

### Text encoders
- `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- `dmis-lab/biobert-base-cased-v1.2`
- a biomedical sentence-transformer variant if available in your environment

## Setup

### Full environment from requirements
```bash
cd /home/cinnamonness/longevmarker-ai
python3 -m pip install --user -r requirements.txt
```

### Recommended runtime install for the first baseline
Use CPU-only PyTorch plus `transformers 4.x`.

```bash
cd /home/cinnamonness/longevmarker-ai
python3 -m pip install --user --extra-index-url https://download.pytorch.org/whl/cpu torch 'transformers>=4.52,<5'
```

## Data preparation workflow

### 1. Rebuild the literature dataset
```bash
python3 scripts/build_literature_dataset.py --retmax 20 --use-pubtator
```

### 2. Rebuild embedding-ready text tables
```bash
python3 scripts/prepare_embedding_materials.py
```

### 3. Fetch the compound registry from PubChem
```bash
python3 scripts/fetch_compound_registry.py
```

### 4. Build molecular benchmark tables
```bash
python3 scripts/prepare_molecular_embedding_materials.py
```

### 5. Populate SELFIES for SELFormer
```bash
python3 scripts/populate_selfies_sequences.py
```

## Embedding extraction

### Important architecture note
Raw molecule embeddings and raw biomarker-text embeddings do not live in the same space.

That means the minimal working pipeline is:
1. embed molecular queries
2. embed biomarker or pair texts
3. learn an alignment layer on the train split
4. run retrieval on the aligned query embeddings

### Molecule embeddings
Example for ChemBERTa:
```bash
python3 scripts/embed_molecules.py   --input-csv data/embedding/molecular_interventions.csv   --id-column query_id   --input-column canonical_smiles   --model-name seyonec/ChemBERTa-zinc-base-v1   --batch-size 4   --output-prefix outputs/chemberta_queries
```

Example for MolFormer:
```bash
python3 scripts/embed_molecules.py   --input-csv data/embedding/molecular_interventions.csv   --id-column query_id   --input-column canonical_smiles   --model-name ibm-research/MoLFormer-XL-both-10pct   --trust-remote-code   --batch-size 4   --output-prefix outputs/molformer_queries
```

Example for SELFormer:
```bash
python3 scripts/embed_molecules.py   --input-csv data/embedding/molecular_interventions.csv   --id-column query_id   --input-column selfies_sequence   --model-name HUBioDataLab/SELFormer   --batch-size 4   --output-prefix outputs/selformer_queries
```

### Text embeddings
Recommended right tower for chemical-to-biomarker retrieval with BioBERT:
```bash
python3 scripts/embed_texts.py   --input-csv data/embedding/biomarker_semantic_texts.csv   --id-column surrogate_id   --text-column embedding_text   --model-name dmis-lab/biobert-base-cased-v1.2   --batch-size 4   --output-prefix outputs/biobert_biomarker_semantic_texts
```

Example for pair texts with BioBERT:
```bash
python3 scripts/embed_texts.py   --input-csv data/embedding/pair_texts.csv   --id-column pair_id   --text-column embedding_text   --model-name dmis-lab/biobert-base-cased-v1.2   --batch-size 4   --output-prefix outputs/biobert_pair_texts
```

Example for surrogate texts with PubMedBERT:
```bash
python3 scripts/embed_texts.py   --input-csv data/embedding/surrogate_texts.csv   --id-column surrogate_id   --text-column embedding_text   --model-name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext   --batch-size 4   --output-prefix outputs/pubmedbert_surrogate_texts
```

### Alignment layer
Ridge alignment from molecular embedding space into the text embedding space.

Recommended chemical-to-biomarker semantic alignment:
```bash
python3 scripts/align_embeddings.py   --query-prefix outputs/molformer_queries   --corpus-prefix outputs/biobert_biomarker_semantic_texts   --qrels-csv data/embedding/molecular_biomarker_semantic_qrels.csv   --output-prefix outputs/molformer_to_biobert_biomarker_semantic_aligned   --train-split train   --eval-splits train,val,test   --regularization 1.0   --top-k 10
```

### Output files
Each embedding script writes:

- `<output-prefix>.embeddings.npy`
- `<output-prefix>.metadata.csv`

The alignment step also writes:

- `<output-prefix>.alignment.npz`
- `<output-prefix>.metrics.csv`

## Retrieval benchmark

### Molecule -> biomarker semantic retrieval from aligned queries
```bash
python3 scripts/run_retrieval_benchmark.py   --query-prefix outputs/molformer_to_biobert_biomarker_semantic_aligned   --corpus-prefix outputs/biobert_biomarker_semantic_texts   --qrels-csv data/embedding/molecular_biomarker_semantic_qrels.csv   --only-split test   --output-dir outputs/benchmarks/molformer_biobert_biomarker_semantic_test
```

### Molecule -> pair retrieval from aligned queries
```bash
python3 scripts/run_retrieval_benchmark.py   --query-prefix outputs/chemberta_to_biobert_pair_aligned   --corpus-prefix outputs/biobert_pair_texts   --qrels-csv data/embedding/molecular_pair_qrels.csv   --only-split test   --output-dir outputs/benchmarks/chemberta_biobert_pair_test
```

### Molecule -> surrogate retrieval from aligned queries
```bash
python3 scripts/run_retrieval_benchmark.py   --query-prefix outputs/selformer_queries   --corpus-prefix outputs/pubmedbert_surrogate_texts   --qrels-csv data/embedding/molecular_surrogate_qrels.csv   --output-dir outputs/benchmarks/selformer_surrogate
```

### Benchmark outputs
Each retrieval run writes:

- `metrics.csv`
- `rankings.csv`

## Baseline runs

### Recommended chemical -> biomarker semantic baseline
This is now the primary right-tower setup for biomarker retrieval.

Artifacts:
- `outputs/biomarker_semantic_run/biobert_biomarker_semantic_texts.*`
- `outputs/biomarker_semantic_run/molformer_to_biobert_biomarker_semantic_aligned.*`
- `outputs/biomarker_semantic_run/benchmarks/molformer_biobert_biomarker_semantic_train/`
- `outputs/biomarker_semantic_run/benchmarks/molformer_biobert_biomarker_semantic_val/`
- `outputs/biomarker_semantic_run/benchmarks/molformer_biobert_biomarker_semantic_test/`
- `outputs/biomarker_semantic_run/run_summary.csv`
- `outputs/model_comparison/biomarker_semantic_runs.csv`

Configuration:
- query model: `ibm-research/MoLFormer-XL-both-10pct`
- query input: `canonical_smiles`
- corpus model: `dmis-lab/biobert-base-cased-v1.2`
- corpus table: `biomarker_semantic_texts.csv`
- alignment: ridge, `regularization = 1.0`

Metrics:
- train `MRR@10 = 0.854167`, `Recall@5 = 1.0`, `nDCG@10 = 0.676466`
- val `MRR@10 = 0.333333`, `Recall@5 = 1.0`, `nDCG@10 = 0.281185`
- test `MRR@10 = 0.75`, `Recall@5 = 1.0`, `nDCG@10 = 0.628184`

Interpretation:
- adding `hallmarks + pathways + aggregated MoA contexts` on the biomarker side changes the task materially for the better
- this right tower generalizes much better than the earlier `pair_texts` and `surrogate_texts` baselines
- the semantic biomarker profile should be treated as the default retrieval corpus for compound-to-biomarker recommendation

### Legacy pair-retrieval baselines
These remain useful as auxiliary baselines and reranking corpora.

#### ChemBERTa + BioBERT + ridge
- test `MRR@10 = 0.25`, `Recall@5 = 0.5`, `nDCG@10 = 0.148748`

#### MolFormer + BioBERT + ridge
- test `MRR@10 = 0.25`, `Recall@5 = 0.5`, `nDCG@10 = 0.151241`

#### SELFormer + BioBERT + ridge
- test `MRR@10 = 0.25`, `Recall@5 = 0.5`, `nDCG@10 = 0.146887`

### Comparison files
- `outputs/model_comparison/pair_benchmark_comparison.csv`
- `outputs/model_comparison/biomarker_semantic_runs.csv`

## Evaluation metrics

Compare runs by:
- `MRR@10`
- `Recall@5`
- `nDCG@10`

## Recommended order

1. Build or refresh the curated and molecular datasets.
2. Populate `SELFIES` and verify `selfies_conversion_status == ok`.
3. Generate molecular embeddings for one encoder family.
4. Generate `biomarker_semantic_texts.csv` embeddings with a biomedical text encoder.
5. Train the ridge alignment on the `train` split against `molecular_biomarker_semantic_qrels.csv`.
6. Benchmark `train`, `val`, and `test` separately.
7. Use `pair_texts.csv` as a secondary corpus for explanation or reranking, not as the primary biomarker retrieval target.
8. Only after that, add projection heads or contrastive fine-tuning.

## Notes

- `ChemBERT` and `MolFormer` are configured as SMILES-based branches.
- `SELFormer` is configured as a SELFIES-based branch.
- `molecular_model_registry.csv` and `text_model_registry.csv` define the recommended model matrix.
- Combination interventions should start with mean pooling over component embeddings.
- The first strong cross-modal biomarker baseline uses frozen encoders plus a learned linear alignment.
- `biomarker_semantic_texts.csv` is the preferred right-tower corpus for compound-to-biomarker retrieval.
- `pair_texts.csv` is still useful for explanation, evidence display, and reranking.
- The text side can remain frozen for the first baseline; then you can add projection heads and contrastive fine-tuning.

## Source references

- PubChem PUG REST API: https://pubchem.ncbi.nlm.nih.gov/rest/pug/
- MolFormer model card: https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct
- ChemBERTa model card: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1
- SELFormer model card: https://huggingface.co/HUBioDataLab/SELFormer
