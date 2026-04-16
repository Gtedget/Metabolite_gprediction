import argparse
import json

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from cid_lookup import cid_to_smiles
from data_utils import MetaboliteDataset
from inference import _resolve_device, load_model
from train import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Fast checkpoint evaluation (teacher-forcing loss + transform accuracy; no beam search)"
    )
    parser.add_argument("--data", default="test.csv", help="CSV split with Parent_SMILES/Metabolite_SMILES")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    model, tokenizer, metadata = load_model(args.model, args.metadata, device)

    df = pd.read_csv(args.data)

    transform_map = metadata.get("transform_map", {})
    enzyme_map = metadata.get("enzyme_map", {})
    coarse_transform_map = metadata.get("coarse_transform_map", {})

    dataset = MetaboliteDataset(
        df=df,
        smiles_lookup_fn=cid_to_smiles,
        tokenizer=tokenizer,
        transform_map=transform_map,
        enzyme_map=enzyme_map,
        coarse_transform_map=coarse_transform_map,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    sequence_criterion = nn.CrossEntropyLoss(ignore_index=0)
    transform_criterion = nn.CrossEntropyLoss()
    coarse_transform_criterion = nn.CrossEntropyLoss()
    reaction_center_criterion = nn.BCEWithLogitsLoss()
    enzyme_criterion = nn.CrossEntropyLoss()

    metrics = evaluate(
        model,
        loader,
        sequence_criterion,
        transform_criterion,
        coarse_transform_criterion,
        reaction_center_criterion,
        enzyme_criterion,
        device,
        transform_loss_weight=float(metadata.get("transform_loss_weight", 0.5)),
        coarse_transform_loss_weight=float(metadata.get("coarse_transform_loss_weight", 0.2)),
        reaction_center_loss_weight=float(metadata.get("reaction_center_loss_weight", 0.2)),
        enzyme_loss_weight=float(metadata.get("enzyme_loss_weight", 0.3)),
        use_enzyme_head=bool(metadata.get("use_enzyme_head", False)),
        progress_desc="Eval",
        show_progress=not args.no_progress,
    )

    payload = {
        "split": args.data,
        "rows": int(len(df)),
        **metrics,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
