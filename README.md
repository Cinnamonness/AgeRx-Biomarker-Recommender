# LongevMarker AI

LongevMarker AI is an MVP Twin Tower recommender for **ranking surrogate biomarkers for chemical longevity interventions**.

The active model does three things:
- reads the intervention-surrogate supervision from `data/processed/curated_biomarkers.csv`
- encodes the intervention through a pretrained **chemical structure encoder**
- scores all biomarker candidates and returns a **ranked logit vector**

This repo keeps the table-building and preprocessing layer unchanged. The work here is only the **model layer**, benchmarking, saved checkpoints, and documentation.

## Problem

Longevity trials are long and expensive.
A practical MVP is not to prove causality or surrogate validity from scratch, but to answer a simpler question:

**given a chemical intervention, which surrogate biomarkers are the best candidates to track?**

That is the task the model solves.

## Data Flow

The model now builds its runtime training/evaluation objects directly from existing processed tables.

### 1. Main supervision table
- `data/processed/curated_biomarkers.csv`

This is the main table.
Each row is one `intervention -> surrogate` link with:
- `mechanism_hint`
- `surrogate_category`
- `pathway`
- `hallmark`
- `evidence_score`
- `surrogate_confidence`
- `literature_support`

Current scale:
- `158` rows
- `42` unique interventions
- `41` unique biomarkers/surrogates

### 2. Structure source
- `data/processed/compound_registry.csv`
- `data/processed/compound_components.csv`
- `data/seed_interventions.csv`

These are used only to resolve chemical inputs for the left tower.
If an intervention is not available as a direct exact molecule in the registry, the model uses one of these fallbacks without changing any table:
- exact registry molecule
- component molecule
- prototype average for class-level interventions
- supplemental local structure mapping inside the model layer
- final text fallback only if no structure proxy exists

### 3. Runtime objects built by the model
At train time the model constructs:
- query rows for interventions
- candidate rows for biomarkers
- qrels-like relevance rows
- triplets for hard-negative training
- fold assignments for cross-validation

These are saved in:
- `outputs/twin_tower_mvp/runtime_queries.csv`
- `outputs/twin_tower_mvp/runtime_biomarker_candidates.csv`
- `outputs/twin_tower_mvp/runtime_relations.csv`
- `outputs/twin_tower_mvp/runtime_triplets.csv`
- `outputs/twin_tower_mvp/fold_assignments.csv`
- `outputs/twin_tower_mvp/structure_resolution.csv`

## Model Architecture

The model is a **Twin Tower recommender**.

### Left tower: intervention chemistry
Input:
- molecular structure for the intervention
- usually SMILES
- for `SELFormer`, SELFIES

Supported encoders:
- `ChemBERTa`
- `MolFormer`
- `SELFormer`

The left tower uses a pretrained chemical encoder and **unfreezes the last transformer layer**.
Everything else remains frozen.

### Right tower: biomarker semantic text
Each biomarker candidate is turned into a semantic profile containing:
- biomarker name
- category
- pathways
- hallmarks
- linked interventions
- mechanism hints
- evidence aggregates
- short literature-derived context

This text is encoded with a lightweight TF-IDF representation.

### Projection heads
After base features are computed:
- the chemical representation goes through a trainable `query_tower`
- the biomarker text representation goes through a trainable `candidate_tower`

Each tower is:
- `Linear`
- `LayerNorm`
- `GELU`
- `Dropout`
- `Linear`
- `L2 normalize`

### Scoring
The final score is a temperature-scaled dot product:
- `score(intervention, biomarker) = q_emb dot b_emb / temperature`

For one intervention query the output is:
- one logit per biomarker candidate
- here: a vector of length `41`

Then logits are sorted descending.
The user-facing prediction is **top-3 biomarkers**.

## What `relevance` Means

`Relevance` is an evidence-weighted proxy label built from:
- `evidence_score`
- `surrogate_confidence`
- `literature_support`

This is used only as ranking supervision.
It should be interpreted as:
- stronger or weaker curated support for the intervention-surrogate association
- not as a clinical effect size

## Training Objective

The model uses two supervision signals.

### 1. Multi-label BCE over biomarker candidates
For each intervention:
- true linked biomarkers are positives
- other biomarkers are negatives
- positive pairs are weighted by the continuous proxy strength described above

This is the main loss that teaches the model to emit a biomarker logit vector.

### 2. Triplet loss
Hard negatives are generated from biomarkers with overlapping:
- category
- hallmark

This improves ranking among confusing candidates.

## Cross-Validation

The current active setup uses **3-fold cross-validation** over interventions.

For each fold:
- one fold is `test`
- the next fold is `val`
- the remaining fold(s) are `train`

Saved CV results:
- `outputs/twin_tower_mvp/cross_validation_folds.csv`
- `outputs/twin_tower_mvp/cross_validation_summary.csv`

## Chemical Encoder Benchmark

Benchmark summary:
- `outputs/encoder_benchmark/encoder_comparison.csv`

Current rigorous run on the updated dataset:

- `MolFormer` with `max_epochs=30`, `patience=5`: `test MRR@10 mean = 0.686035`, `Recall@5 mean = 0.833333`, `nDCG@10 mean = 0.560440`

Reference short-run comparisons from the earlier exploratory pass:
- `ChemBERTa`: `test MRR@10 mean = 0.618944`, `Recall@5 mean = 0.833333`, `nDCG@10 mean = 0.490962`
- `SELFormer`: `test MRR@10 mean = 0.414664`, `Recall@5 mean = 0.809524`, `nDCG@10 mean = 0.390678`

Active winner:
- `MolFormer`

Why `MolFormer` is active now:
- best currently available long-run result
- improved over the earlier quick benchmark
- selected from completed `30`-epoch CV folds by validation quality

## Active Production Checkpoint

Active output directory:
- `outputs/twin_tower_mvp`

Selected deployment checkpoint:
- `outputs/twin_tower_mvp/final_model`

Deployment metadata:
- `outputs/twin_tower_mvp/deployment_summary.json`

Selection rule:
- highest `val_mrr@10` among completed `30`-epoch MolFormer CV folds

Selected fold:
- `fold_1`

Configured training regime:
- max epochs: `30`
- early stopping patience: `5`
- actual epochs by fold: `11`, `16`, `18`

Checkpoint metrics of the deployed fold:
- train `MRR@10 = 0.952381`, `Recall@5 = 1.0`, `nDCG@10 = 0.82723`
- val `MRR@10 = 0.725`, `Recall@5 = 0.928571`, `nDCG@10 = 0.57252`
- test `MRR@10 = 0.61763`, `Recall@5 = 0.714286`, `nDCG@10 = 0.509728`

Cross-validated mean metrics for the active encoder family:
- test `MRR@10 = 0.686035 ± 0.053085`
- test `Recall@5 = 0.833333 ± 0.089087`
- test `nDCG@10 = 0.560440 ± 0.067057`

## What The Model Returns

The prediction script returns:
- `candidate_count`
- the full hidden classifier space size
- `class_to_biomarker` mapping
- the ranked top-3 biomarkers with logits

So yes, the model output is:
- a **vector of biomarker logits**
- then a **ranked top-3 recommendation list**

Example current prediction for `Metformin` from the active MolFormer checkpoint:
- `NAD+ NADH ratio`
- `SASP panel`
- `fasting glucose`

Saved dictionaries:
- `outputs/twin_tower_mvp/final_model/class_to_biomarker.json`
- `outputs/twin_tower_mvp/final_model/biomarker_to_class.json`

Saved weights:
- `outputs/twin_tower_mvp/final_model/model.pt`
- `outputs/twin_tower_mvp/final_model/query_tower_weights.pt`
- `outputs/twin_tower_mvp/final_model/candidate_tower_weights.pt`

## How To Run

### Train one model with cross-validation and save the deployment checkpoint

Example: active recommended path with `MolFormer`

```bash
cd longevmarker-ai
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python3 scripts/train_twin_tower.py \
  --output-dir outputs/twin_tower_mvp \
  --molecule-encoder molformer \
  --device cpu \
  --epochs 30 \
  --early-stopping-patience 5 \
  --num-folds 3 \
  --predict-top-k 3 \
  --unfreeze-last-n-layers 1
```

### Benchmark all three chemical encoders

```bash
cd longevmarker-ai
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python3 scripts/benchmark_chemical_encoders.py \
  --output-dir outputs/encoder_benchmark \
  --device cpu \
  --epochs 30 \
  --early-stopping-patience 5 \
  --num-folds 3 \
  --predict-top-k 3 \
  --unfreeze-last-n-layers 1
```

### Evaluate the active deployment checkpoint

```bash
cd longevmarker-ai
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python3 scripts/evaluate_twin_tower.py \
  --model-dir outputs/twin_tower_mvp/final_model \
  --device cpu
```

### Predict top-3 biomarkers for a known intervention

```bash
cd longevmarker-ai
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python3 scripts/predict_biomarkers.py \
  --model-dir outputs/twin_tower_mvp/final_model \
  --intervention Metformin \
  --top-k 3 \
  --device cpu
```

### Predict for a custom structure

```bash
cd longevmarker-ai
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python3 scripts/predict_biomarkers.py \
  --model-dir outputs/twin_tower_mvp/final_model \
  --structure "CN(C)C(=N)N=C(N)N" \
  --query-text "Intervention profile: custom AMPK-like compound." \
  --top-k 3 \
  --device cpu
```