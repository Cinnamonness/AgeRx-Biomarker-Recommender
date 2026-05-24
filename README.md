# LongevMarker AI

LongevMarker AI is an MVP recommender system for **ranking surrogate biomarkers for chemical longevity interventions**.

The project goal is simple:
- take a chemical intervention
- use its molecular structure
- compare it against biomarker semantic descriptions
- return a ranked biomarker logit vector

## Context

Longevity studies are slow and expensive.
A practical MVP is not to prove causality, but to recommend **which biomarkers are worth tracking** for a given intervention.

This repo now does that with a **chemical-structure to biomarker-text Twin Tower model**.

## What was not changed

The following parts were left untouched:
- all tables in `data/`
- all table-building and preprocessing logic
- the existing processing scripts that generate training tables

The source-of-truth table remains:
- `data/processed/curated_biomarkers.csv`

## What the active model does

Input:
- a chemical intervention represented by molecular structure
- usually taken from `data/embedding/molecular_interventions.csv`
- or provided manually as a structure string for inference

Output:
- a **vector of biomarker logits** over the full biomarker candidate set
- the same vector **ranked descending by score**
- top-k biomarker recommendations

So yes, the model output is:
- one logit per biomarker candidate
- then a ranked biomarker list built from those logits

## Which tables the model uses

### Main semantic source
- `data/processed/curated_biomarkers.csv`

### Existing generated embedding tables
The model consumes these already-prepared files and does not modify them:
- `data/embedding/molecular_interventions.csv`
- `data/embedding/biomarker_semantic_texts.csv`
- `data/embedding/molecular_biomarker_semantic_qrels.csv`
- `data/embedding/molecule_to_biomarker_semantic_triplets.csv`

### Optional simpler biomarker table
- `data/embedding/surrogate_texts.csv`

## Architecture

The active model is a **Twin Tower recommender**.

### Left tower: intervention chemistry
The left tower consumes molecular structure for the intervention.

Supported chemical encoders:
- `ChemBERTa`
- `MolFormer`
- `SELFormer`

Representations:
- `ChemBERTa`: `canonical_smiles`
- `MolFormer`: `canonical_smiles`
- `SELFormer`: `selfies_sequence`

The left side is responsible for encoding the intervention as chemistry, not as free text.

### Right tower: biomarker semantics
The right tower consumes biomarker text from:
- `biomarker_semantic_texts.csv`

Each candidate text already contains rich semantic context such as:
- biomarker name
- biomarker category
- pathways
- hallmarks
- linked interventions
- mechanism themes
- short evidence-oriented description

The right side is the final recommendation space.
At inference time the model scores the molecule embedding against all biomarker candidate embeddings.

### Backbone strategy
The model uses:
- a frozen pretrained chemical encoder on the left
- a TF-IDF text encoder on the right
- trainable projection heads on both sides

This keeps the MVP realistic:
- no table changes
- real molecular encoders are used
- training stays lightweight enough for a hackathon-scale dataset

### Projection heads
After base features are computed:
- molecular features go through a trainable query projection tower
- biomarker text features go through a trainable candidate projection tower

Each tower applies:
- linear layer
- layer norm
- GELU
- dropout
- final linear layer
- L2 normalization

### Scoring
The final score is a temperature-scaled dot product between:
- projected intervention embedding
- projected biomarker embedding

For one intervention query, the model returns:
- `logits shape = [number_of_biomarker_candidates]`

For the current saved MVP run:
- `logits shape = [19]`

Then logits are sorted descending to produce the final ranking.

## How supervision works

The model is trained directly on the intervention-surrogate relation that already exists in the generated molecular tables.

### 1. Multi-label BCE over biomarker candidates
From:
- `molecular_biomarker_semantic_qrels.csv`

For each intervention query:
- relevant biomarkers are positives
- the rest are negatives
- relevance values weight the positive supervision

This is the main loss that teaches the model to output a biomarker logit vector.

### 2. Hard-negative triplet loss
From:
- `molecule_to_biomarker_semantic_triplets.csv`

For each anchor intervention:
- one correct biomarker semantic candidate is positive
- one hard negative biomarker semantic candidate is negative

This sharpens ranking around confusing biomarker candidates.

### Final training objective
The current model uses:
- weighted BCE loss over the full biomarker candidate set
- triplet margin loss over hard negatives

So this is a **chemical-to-biomarker retrieval model with ranking-aware supervision**.

## Chemical encoder comparison

Different chemical encoders were actually tested in the new Twin Tower.

Benchmark file:
- `outputs/encoder_benchmark/encoder_comparison.csv`

Compared encoders:
- `ChemBERTa`
- `MolFormer`
- `SELFormer`

Benchmark results on the current dataset:

- `ChemBERTa`: test `MRR@10 = 0.75`, `Recall@5 = 1.0`, `nDCG@10 = 0.596107`
- `MolFormer`: test `MRR@10 = 0.75`, `Recall@5 = 1.0`, `nDCG@10 = 0.648394`
- `SELFormer`: test `MRR@10 = 1.0`, `Recall@5 = 1.0`, `nDCG@10 = 0.646759`

Current active MVP run is based on:
- `SELFormer`

Why it was selected:
- best `test MRR@10`
- full retrieval worked end-to-end on the current dataset

## Active code

### Main model module
- `src/longevmarker/twin_tower.py`

This file contains:
- chemical encoder loading
- TF-IDF biomarker text encoding
- Twin Tower model
- training loop
- evaluation
- prediction

### Benchmark module
- `src/longevmarker/twin_tower_benchmark.py`

### CLI entry points
- `scripts/train_twin_tower.py`
- `scripts/evaluate_twin_tower.py`
- `scripts/predict_biomarkers.py`
- `scripts/benchmark_chemical_encoders.py`

## Processing code that remains untouched

These stay in the repo as-is:
- `scripts/build_literature_dataset.py`
- `scripts/prepare_embedding_materials.py`
- `scripts/fetch_compound_registry.py`
- `scripts/prepare_molecular_embedding_materials.py`
- `scripts/populate_selfies_sequences.py`
- `src/longevmarker/dataset_builder.py`
- `src/longevmarker/embedding_data.py`
- `src/longevmarker/compound_registry.py`
- `src/longevmarker/molecular_embedding_data.py`
- `src/longevmarker/selfies_data.py`
- related PubMed / PubTator / biomarker matcher code

## How to run

### 1. Make sure the existing embedding tables already exist
If needed, regenerate them with the unchanged processing scripts:

```bash
cd /longevmarker-ai
python3 scripts/prepare_embedding_materials.py
python3 scripts/prepare_molecular_embedding_materials.py
python3 scripts/populate_selfies_sequences.py
```

### 2. Train one Twin Tower run
Example: active recommended run with `SELFormer`

```bash
cd /longevmarker-ai
PYTHONPATH=src python3 scripts/train_twin_tower.py \
  --molecule-encoder selformer \
  --output-dir outputs/twin_tower_mvp \
  --device cpu
```

Example: train with `MolFormer`

```bash
cd /longevmarker-ai
PYTHONPATH=src python3 scripts/train_twin_tower.py \
  --molecule-encoder molformer \
  --output-dir outputs/twin_tower_molformer \
  --device cpu
```

Example: train with `ChemBERTa`

```bash
cd /longevmarker-ai
PYTHONPATH=src python3 scripts/train_twin_tower.py \
  --molecule-encoder chemberta \
  --output-dir outputs/twin_tower_chemberta \
  --device cpu
```

### 3. Benchmark all three chemical encoders

```bash
cd /longevmarker-ai
PYTHONPATH=src python3 scripts/benchmark_chemical_encoders.py \
  --output-dir outputs/encoder_benchmark \
  --device cpu
```

### 4. Evaluate the active model

```bash
cd /longevmarker-ai
PYTHONPATH=src python3 scripts/evaluate_twin_tower.py \
  --model-dir outputs/twin_tower_mvp \
  --device cpu
```

### 5. Predict biomarkers for an existing intervention in the saved query table

```bash
cd /longevmarker-ai
PYTHONPATH=src python3 scripts/predict_biomarkers.py \
  --model-dir outputs/twin_tower_mvp \
  --intervention Metformin \
  --top-k 10 \
  --device cpu
```

### 6. Predict biomarkers for a custom structure
If you pass `--structure`, the structure must match the active encoder representation.

Examples:
- for `ChemBERTa` or `MolFormer`: pass SMILES
- for `SELFormer`: pass SELFIES

```bash
cd /longevmarker-ai
PYTHONPATH=src python3 scripts/predict_biomarkers.py \
  --model-dir outputs/twin_tower_mvp \
  --structure "[C][N][Branch1][C][C][C][=Branch1][C][=N][N][=C][Branch1][C][N][N]" \
  --top-k 10 \
  --device cpu
```

## What prediction returns

The prediction script computes:
- one logit per biomarker candidate
- then ranks all candidates descending by logit

Printed output includes:
- rank
- biomarker name
- logit
- biomarker category
- pathways
- hallmarks

Example current active run for `Metformin` top-5:
- `fasting insulin`
- `IL6`
- `CRP`
- `GDF15`
- `triglycerides`

## Saved model artifacts

The active model run in `outputs/twin_tower_mvp` contains:
- `model.pt`
- `config.json`
- `text_vectorizer.pkl`
- `query_base_features.npy`
- `candidate_base_features.npy`
- `query_rows.csv`
- `biomarker_candidates.csv`
- `training_history.csv`
- `metrics.csv`
- `rankings_train.csv`
- `rankings_val.csv`
- `rankings_test.csv`
- `eval_metrics.csv`
- `eval_rankings_train.csv`
- `eval_rankings_val.csv`
- `eval_rankings_test.csv`

## Current active MVP run

The cleaned repo now keeps:
- one active model run in `outputs/twin_tower_mvp`
- one benchmark directory in `outputs/encoder_benchmark`

Current active run metrics (`SELFormer`):
- train `MRR@10 = 0.84375`, `Recall@5 = 1.0`, `nDCG@10 = 0.680612`
- val `MRR@10 = 1.0`, `Recall@5 = 1.0`, `nDCG@10 = 0.571253`
- test `MRR@10 = 1.0`, `Recall@5 = 1.0`, `nDCG@10 = 0.646759`

