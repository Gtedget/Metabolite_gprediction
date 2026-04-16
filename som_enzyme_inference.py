import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from data_utils import normalize_transformation_family
from inference import (
    _resolve_device,
    load_model,
    predict_sites_of_metabolism,
    predict_top_transformations,
)


def _invert_label_map(label_map: Dict[str, Any]) -> Dict[int, str]:
    id_to_label: Dict[int, str] = {}
    for label, idx in (label_map or {}).items():
        try:
            id_to_label[int(idx)] = str(label)
        except Exception:
            continue
    return id_to_label


def select_most_reliable_som(
    som_candidates: List[Dict[str, Any]],
    threshold: float,
) -> Optional[Dict[str, Any]]:
    if not som_candidates:
        return None

    ranked = sorted(som_candidates, key=lambda item: float(item.get("probability", 0.0)), reverse=True)
    above = [item for item in ranked if float(item.get("probability", 0.0)) >= threshold]
    selected = above[0] if above else ranked[0]

    top1 = float(ranked[0].get("probability", 0.0))
    top2 = float(ranked[1].get("probability", 0.0)) if len(ranked) > 1 else 0.0
    margin = top1 - top2

    return {
        "selected": selected,
        "selection_rule": "highest_probability_above_threshold_else_top1",
        "top1_probability": top1,
        "top2_probability": top2,
        "top1_minus_top2_margin": margin,
    }


@torch.no_grad()
def predict_top_enzymes_from_head(
    model,
    metadata: Dict[str, Any],
    graph,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if getattr(model, "enzyme_head", None) is None:
        return []

    sos_id = 1  # matches SmilesTokenizer special tokens (<pad>=0,<sos>=1,<eos>=2)
    _, _, _, _, pred_enzyme = model(
        graph,
        torch.tensor([[sos_id]], dtype=torch.long, device=graph.x.device),
    )
    if pred_enzyme is None:
        return []

    enzyme_map = metadata.get("enzyme_map", {})
    id_to_enzyme = _invert_label_map(enzyme_map)

    probs = torch.softmax(pred_enzyme[0], dim=0)
    values, indices = torch.topk(probs, k=min(int(top_k), int(probs.shape[0])))
    out: List[Dict[str, Any]] = []
    for value, idx in zip(values.tolist(), indices.tolist()):
        out.append(
            {
                "enzyme_id": int(idx),
                "enzyme": id_to_enzyme.get(int(idx), str(int(idx))),
                "probability": float(value),
            }
        )
    return out


def _build_p_enzyme_given_family(train_csv: str) -> Dict[str, Dict[str, float]]:
    import pandas as pd

    df = pd.read_csv(train_csv)
    needed = {"Transformation", "Enzyme"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Fallback enzyme layer requires columns: {sorted(needed)} (missing {sorted(missing)})")

    df = df.dropna(subset=["Transformation", "Enzyme"]).copy()
    df["_family"] = df["Transformation"].apply(normalize_transformation_family)

    counts = df.groupby(["_family", "Enzyme"]).size().rename("count").reset_index()
    totals = df.groupby(["_family"]).size().rename("total").reset_index()
    merged = counts.merge(totals, on="_family", how="left")
    merged["prob"] = merged["count"] / merged["total"].clip(lower=1)

    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for _, row in merged.iterrows():
        family = str(row["_family"])
        enzyme = str(row["Enzyme"])
        out[family][enzyme] = float(row["prob"])

    return dict(out)


def predict_top_enzymes_fallback_from_transformations(
    train_csv: str,
    top_transformations: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if not top_transformations:
        return []

    p_enzyme_given_family = _build_p_enzyme_given_family(train_csv)

    scores: Dict[str, float] = defaultdict(float)
    for item in top_transformations:
        family = str(item.get("coarse_family") or normalize_transformation_family(item.get("transformation")))
        weight = float(item.get("probability", 0.0))
        for enzyme, p in p_enzyme_given_family.get(family, {}).items():
            scores[enzyme] += weight * float(p)

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_k))]
    total = sum(score for _, score in ranked) or 1.0

    return [
        {
            "enzyme": enzyme,
            "score": float(score),
            "normalized_score": float(score / total),
            "method": "fallback_p(enzyme|transformation_family) * p(transformation_family)",
        }
        for enzyme, score in ranked
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Predict multiple SoM-like atoms and rank likely enzymes without requiring metabolite generation"
    )
    parser.add_argument("--model", default="trained_model.pt", help="Path to model weights")
    parser.add_argument(
        "--metadata",
        default="trained_model.metadata.json",
        help="Path to metadata JSON saved during training",
    )
    parser.add_argument("--precursor", required=True, help="Precursor SMILES")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--som_top_k", type=int, default=8, help="Number of SoM-like atoms to report")
    parser.add_argument("--som_threshold", type=float, default=0.5, help="Threshold for flagging a predicted SoM")

    parser.add_argument("--transform_top_k", type=int, default=5, help="Number of top transformations to use")

    parser.add_argument("--enzyme_top_k", type=int, default=5, help="Number of top enzymes to report")
    parser.add_argument(
        "--fallback_train_csv",
        default=None,
        help="Optional CSV (with Transformation and Enzyme columns) to enable a fallback enzyme layer when the model has no enzyme head",
    )

    parser.add_argument("--json_out", default=None, help="Optional JSON path to save results")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    model, _, metadata = load_model(args.model, args.metadata, device)

    som_candidates, graph = predict_sites_of_metabolism(
        model,
        args.precursor,
        device,
        top_k=args.som_top_k,
        threshold=args.som_threshold,
    )
    som_selection = select_most_reliable_som(som_candidates, threshold=args.som_threshold)

    top_transformations = predict_top_transformations(model, metadata=metadata, graph=graph, top_k=args.transform_top_k)

    enzymes: List[Dict[str, Any]] = []
    enzyme_method = None

    enzymes_from_head = predict_top_enzymes_from_head(model, metadata, graph, top_k=args.enzyme_top_k)
    if enzymes_from_head:
        enzymes = enzymes_from_head
        enzyme_method = "model_enzyme_head"
    elif args.fallback_train_csv:
        enzymes = predict_top_enzymes_fallback_from_transformations(
            args.fallback_train_csv,
            top_transformations,
            top_k=args.enzyme_top_k,
        )
        enzyme_method = "fallback_from_transformations"

    payload = {
        "precursor": args.precursor,
        "sites_of_metabolism": som_candidates,
        "som_threshold": float(args.som_threshold),
        "som_selection": som_selection,
        "transformations": top_transformations,
        "enzymes": enzymes,
        "enzyme_method": enzyme_method,
    }

    print(json.dumps(payload, indent=2))

    if args.json_out:
        out_dir = os.path.dirname(os.path.abspath(args.json_out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
