import argparse
import json
import os

import pandas as pd
from tqdm.auto import tqdm
from inference import (
    _resolve_device,
    beam_search_candidates,
    canonicalize_smiles,
    load_model,
)

try:
    import mlflow
except ImportError:
    mlflow = None


def main():
    parser = argparse.ArgumentParser(description="Evaluate metabolite generation on a CSV split")
    parser.add_argument("--data", default="test.csv", help="CSV split with Parent_SMILES and Metabolite_SMILES columns")
    parser.add_argument("--model", default="trained_model.pt", help="Path to model weights")
    parser.add_argument("--metadata", default="trained_model.metadata.json", help="Training metadata JSON")
    parser.add_argument("--top_k", type=int, default=5, help="Number of candidate metabolites to generate")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for decoding")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for quick evaluation")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", default=None, help="Optional JSON path to save evaluation metrics")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--use_mlflow", action="store_true", help="Log evaluation params, metrics, and artifacts to MLflow")
    parser.add_argument("--mlflow_experiment", default="metabolite-generation-eval", help="MLflow experiment name")
    parser.add_argument("--mlflow_run_name", default=None, help="Optional MLflow run name")
    parser.add_argument("--mlflow_tracking_uri", default=None, help="MLflow tracking URI, e.g. file:/content/mlruns in Colab")
    args = parser.parse_args()

    if args.use_mlflow:
        if mlflow is None:
            raise ImportError("mlflow is not installed. Install requirements.txt before using --use_mlflow.")
        if args.out is None:
            model_dir = os.path.dirname(os.path.abspath(args.model)) or "."
            args.out = os.path.join(model_dir, "generation_eval.json")
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        if args.mlflow_experiment:
            mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=args.mlflow_run_name)
        mlflow.log_params(
            {
                "data": args.data,
                "model": args.model,
                "metadata": args.metadata,
                "top_k": args.top_k,
                "beam_width": args.beam_width,
                "limit": args.limit,
                "device": args.device,
            }
        )

    df = pd.read_csv(args.data)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    required = {"Parent_SMILES", "Metabolite_SMILES"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Evaluation CSV is missing required columns: {sorted(missing)}")

    device = _resolve_device(args.device)
    model, tokenizer, _ = load_model(args.model, args.metadata, device)

    total_rows = 0
    valid_top1 = 0
    valid_any = 0
    exact_top1 = 0
    exact_topk = 0
    unique_candidate_counts = []

    row_iter = tqdm(df.iterrows(), total=len(df), desc="Generation eval", disable=args.no_progress)
    for _, row in row_iter:
        expected = canonicalize_smiles(str(row["Metabolite_SMILES"]).strip())
        precursor = str(row["Parent_SMILES"]).strip()
        if not expected or not precursor:
            continue

        candidates, _ = beam_search_candidates(
            model,
            tokenizer,
            precursor,
            device,
            num_candidates=args.top_k,
            beam_width=args.beam_width,
        )
        if not candidates:
            continue

        total_rows += 1
        candidate_canonicals = [candidate["canonical_smiles"] for candidate in candidates if candidate["canonical_smiles"]]
        unique_candidate_counts.append(len(candidate_canonicals))

        if candidates[0]["is_valid"]:
            valid_top1 += 1
        if candidate_canonicals:
            valid_any += 1
        if candidates[0]["canonical_smiles"] == expected:
            exact_top1 += 1
        if expected in candidate_canonicals:
            exact_topk += 1

        row_iter.set_postfix(valid_top1=valid_top1, exact_top1=exact_top1, evaluated=total_rows)

    metrics = {
        "rows_evaluated": total_rows,
        "top1_valid_rate": valid_top1 / total_rows if total_rows else 0.0,
        "topk_any_valid_rate": valid_any / total_rows if total_rows else 0.0,
        "top1_exact_canonical_match_rate": exact_top1 / total_rows if total_rows else 0.0,
        "topk_exact_canonical_match_rate": exact_topk / total_rows if total_rows else 0.0,
        "avg_valid_unique_candidates": sum(unique_candidate_counts) / len(unique_candidate_counts) if unique_candidate_counts else 0.0,
        "top_k": args.top_k,
        "beam_width": args.beam_width,
    }

    print(json.dumps(metrics, indent=2))

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)

    if args.use_mlflow:
        mlflow.log_metrics(metrics)
        if args.out and os.path.exists(args.out):
            mlflow.log_artifact(args.out)
        mlflow.end_run()


if __name__ == "__main__":
    main()