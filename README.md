Metabolite predictor (GAT + Transformer)

This folder contains a minimal end-to-end pipeline for learning a precursor->metabolite mapping:

- A Graph Attention Network (GAT) encodes the precursor molecule graph.
- A Transformer decoder generates metabolite targets autoregressively, using SELFIES by default to improve structural validity.
- A transformation head predicts reaction class.
- The enzyme head is disabled by default because enzyme labels in this dataset are often multi-valued.


Project structure

- cid_lookup.py
  - CID->SMILES via PubChem (PubChemPy) with a local cache file (smiles_cache.json)
- preprocess_dataset.py
  - Reads a raw CSV, resolves SMILES from MetXBioDB_substances.csv using Predecessor_CID/Successor_CID, merges selected fields from metxbiodb.csv, writes processed CSV + label maps + train/val/test splits
- data_utils.py
  - SMILES tokenizer, SMILES->PyG graph conversion, PyG Dataset
- model.py
  - GAT encoder + Transformer decoder + multi-task heads
- train.py
  - Training loop + saves model weights and metadata
- inference.py
  - Loads saved model + samples metabolite SMILES from a precursor SMILES


Dataset format (raw)

The preprocessing script expects a CSV with at least these columns:

- Predecessor_CID
- Successor_CID
- Transformation
- Enzyme

Optional extra columns (kept if present): names, biosystem, citations, evidence, etc.

In this workspace:
- the transformation file is [MetXBioDB_Transformations.csv](../MetXBioDB_Transformations.csv)
- the local CID→SMILES table is [MetXBioDB_substances.csv](../MetXBioDB_substances.csv)
- the supplementary reaction dataset is [metxbiodb.csv](../metxbiodb.csv)


Setup

1. Create a Python environment (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r Project-1\requirements.txt
```

Notes:
- RDKit installation can be platform-dependent. If pip installation fails, install RDKit via conda-forge.
- torch-geometric (PyG) often requires installing matching wheels for your PyTorch/CUDA version.


Google Colab + MLflow

This project is set up to run in Google Colab with GPU acceleration and optional MLflow tracking.

Quick path:
- switch Colab runtime to GPU
- clone the GitHub repo
- `pip install -r requirements.txt`
- run `train.py` with `--device cuda --use_mlflow`

Full Colab instructions are in [COLAB.md](COLAB.md).
The ready-to-run notebook is [colab/Metabolite_GPrediction_Colab.ipynb](colab/Metabolite_GPrediction_Colab.ipynb).


Preprocess

This step resolves CIDs to SMILES from the local substances file, merges supplemental metadata from metxbiodb.csv, and writes:
- processed_metabolism_data.csv
- transform_map.json
- enzyme_map.json
- train.csv / val.csv / test.csv

The processed CSV also includes `meta_*` columns such as reference, InChI, InChIKey, reaction type, and biotransformation type when available.
Rows missing `Transformation` or `Enzyme` are dropped, RDKit-invalid SMILES are filtered out, exact duplicate `(Parent_SMILES, Metabolite_SMILES, Transformation, Enzyme)` rows are removed, and the split is performed by `Parent_SMILES` so the same precursor does not appear across train/val/test.

Example:

```powershell
python Project-1\preprocess_dataset.py --input MetXBioDB_Transformations.csv --substances MetXBioDB_substances.csv --metadata metxbiodb.csv --out_dir Project-1
```

Use `--pubchem_fallback` only if some CIDs are missing from the local substances table.


Train

Train on the split CSV (default: train.csv) and save:
- trained_model.pt
- trained_model.best.pt (best validation-loss checkpoint when validation data is available)
- trained_model.metadata.json (includes class counts + maps)

By default, the current training script writes those files into `artifacts/<run_name>/`, where `run_name` is a timestamp such as `run_20260408_153000`. This avoids Colab runs overwriting each other. You can override that with `--output_dir` and `--run_name`, or still pass explicit artifact paths if needed.

By default, training uses SMILES generation + transformation prediction only. The enzyme head is off unless you pass `--use_enzyme_head`.
When sibling [Project-1/val.csv](Project-1/val.csv) and [Project-1/test.csv](Project-1/test.csv) files are present, training evaluates validation loss and transformation accuracy each epoch, saves the best validation-loss checkpoint, and reports final test metrics at the end.
The decoder target representation defaults to SELFIES (`--representation selfies`) because it is substantially more robust than raw SMILES for sequence generation.

Example:

```powershell
python Project-1\train.py --data Project-1\train.csv --epochs 20 --batch_size 16 --lr 1e-4 --output_dir Project-1\artifacts --run_name local_smoketest
```


Inference

This generates a metabolite SMILES from a precursor SMILES string:

```powershell
python Project-1\inference.py --model Project-1\trained_model.pt --metadata Project-1\trained_model.metadata.json
```

Then enter a precursor SMILES when prompted. Use `--top_k 5 --beam_width 5` to return multiple ranked metabolite candidates instead of a single greedy output. Use `--no_repeat_ngram_size 3` to reduce repetitive degenerate strings during decoding.

Inference now also reports the highest-scoring site-of-metabolism-like atoms from the model's reaction-center head. These are heuristic atom-level scores derived from the auxiliary reaction-center task, not curated SoM labels.

For scripted inference, pass the precursor directly and optionally save a combined JSON payload:

```powershell
python Project-1\inference.py --model Project-1\artifacts\local_smoketest\trained_model.best.pt --metadata Project-1\artifacts\local_smoketest\trained_model.metadata.json --precursor "CCO" --top_k 5 --beam_width 5 --no_repeat_ngram_size 3 --som_top_k 5 --json_out Project-1\artifacts\local_smoketest\inference.json
```

You can also render an SVG of the precursor with the top predicted SoM-like atoms highlighted:

```powershell
python Project-1\inference.py --model Project-1\artifacts\local_smoketest\trained_model.best.pt --metadata Project-1\artifacts\local_smoketest\trained_model.metadata.json --precursor "CCO" --som_top_k 5 --som_svg_out Project-1\artifacts\local_smoketest\som.svg
```


Generation evaluation

To evaluate structure generation on a split using canonical SMILES exact-match metrics:

```powershell
python Project-1\evaluate_generation.py --data Project-1\test.csv --model Project-1\artifacts\local_smoketest\trained_model.best.pt --metadata Project-1\artifacts\local_smoketest\trained_model.metadata.json --top_k 5 --beam_width 5 --use_mlflow
```

This reports top-1 validity, top-k validity, top-1 exact canonical match rate, top-k exact canonical match rate, and average number of unique valid candidates per row. With `--use_mlflow`, those metrics are also logged to MLflow and the optional JSON output is stored as an artifact.


Common issues

- Local preprocessing is fast because SMILES are resolved from MetXBioDB_substances.csv.
- PubChem fallback is optional and only needed if the local substances file is incomplete.
- CUDA mismatch errors: use --device cpu for training/inference, or ensure PyTorch detects CUDA.
- If your dataset has more classes than the default heads: training writes metadata and inference loads it, so keep the metadata file alongside the model.
