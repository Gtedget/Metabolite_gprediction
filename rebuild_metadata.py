import argparse
import json
import os

import pandas as pd

from data_utils import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from train import (
    build_coarse_transform_map,
    build_tokenizer,
    load_label_map,
    resolve_default_path,
)


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild a training metadata JSON (tokenizer + label maps) for an existing checkpoint"
    )
    parser.add_argument("--data", default="train.csv", help="Training CSV path")
    parser.add_argument("--val_data", default=None, help="Optional validation CSV path")
    parser.add_argument("--test_data", default=None, help="Optional test CSV path")
    parser.add_argument("--transform_map", default=None, help="Transformation label map JSON")
    parser.add_argument("--enzyme_map", default=None, help="Enzyme label map JSON")
    parser.add_argument("--representation", choices=["smiles", "selfies"], default="selfies")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument(
        "--use_enzyme_head",
        action="store_true",
        help="Set if the checkpoint was trained with the enzyme head enabled",
    )
    parser.add_argument("--out", required=True, help="Path to write metadata JSON")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    val_df = pd.read_csv(args.val_data) if args.val_data and os.path.exists(args.val_data) else None
    test_df = pd.read_csv(args.test_data) if args.test_data and os.path.exists(args.test_data) else None

    if args.transform_map is None:
        args.transform_map = resolve_default_path(args.data, "transform_map.json")
    if args.enzyme_map is None:
        args.enzyme_map = resolve_default_path(args.data, "enzyme_map.json")

    if "Transformation" not in df.columns or "Enzyme" not in df.columns:
        raise ValueError("Training CSV must include 'Transformation' and 'Enzyme' columns")

    tokenizer = build_tokenizer(args, df, val_df=val_df, test_df=test_df)
    transform_map = load_label_map(args.transform_map, "Transformation", df)
    enzyme_map = load_label_map(args.enzyme_map, "Enzyme", df)
    coarse_transform_map = build_coarse_transform_map(df, val_df=val_df, test_df=test_df)

    metadata = {
        "representation": args.representation,
        "tokenizer": tokenizer.to_config(),
        "num_transform_classes": len(transform_map),
        "num_coarse_transform_classes": len(coarse_transform_map),
        "num_enzyme_classes": len(enzyme_map) if args.use_enzyme_head else 0,
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "bond_feature_dim": BOND_FEATURE_DIM,
        "use_enzyme_head": bool(args.use_enzyme_head),
        "coarse_transform_map": coarse_transform_map,
        "transform_map": transform_map,
        "enzyme_map": enzyme_map,
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(args.out)


if __name__ == "__main__":
    main()
