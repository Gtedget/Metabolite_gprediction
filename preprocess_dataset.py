# preprocess_dataset.py
import pandas as pd
import json
import argparse
import os
from cid_lookup import cid_to_smiles
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def read_csv_with_fallback(path):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def normalize_cid(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() == "NULL":
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def build_local_smiles_lookup(substances_path):
    df = read_csv_with_fallback(substances_path)
    required = {"PubChem_CID", "SMILES"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Substances file is missing required columns: {sorted(missing)}"
        )

    cid_to_smiles_map = {}
    for _, row in df.iterrows():
        cid = normalize_cid(row["PubChem_CID"])
        smiles = row["SMILES"] if "SMILES" in row else None
        if cid and pd.notna(smiles) and str(smiles).strip():
            cid_to_smiles_map[cid] = str(smiles).strip()
    return cid_to_smiles_map


def build_metadata_table(metadata_path):
    df = read_csv_with_fallback(metadata_path)
    if "biotid" not in df.columns:
        raise ValueError("Metadata file must contain a 'biotid' column")

    keep_columns = [
        "biotid",
        "substrate_name",
        "substrate_cid",
        "substrate_inchikey",
        "substrate_inchi",
        "enzyme",
        "reaction_type",
        "biotransformation_type",
        "biosystem",
        "prod_name",
        "prod_cid",
        "prod_inchikey",
        "prod_inchi",
        "reference",
    ]
    available_columns = [column for column in keep_columns if column in df.columns]
    metadata_df = df[available_columns].copy()
    metadata_df["biotid"] = metadata_df["biotid"].astype(str).str.strip()

    duplicate_count = int(metadata_df["biotid"].duplicated().sum())
    if duplicate_count:
        metadata_df = metadata_df.drop_duplicates(subset=["biotid"], keep="first")

    rename_map = {column: f"meta_{column}" for column in metadata_df.columns if column != "biotid"}
    metadata_df = metadata_df.rename(columns=rename_map)
    return metadata_df, duplicate_count


def split_by_group(df, group_column, seed, train_frac=0.8, val_frac=0.1):
    group_values = (
        df[group_column]
        .dropna()
        .drop_duplicates()
        .sample(frac=1.0, random_state=seed)
        .tolist()
    )

    train_end = int(len(group_values) * train_frac)
    val_end = train_end + int(len(group_values) * val_frac)

    train_groups = set(group_values[:train_end])
    val_groups = set(group_values[train_end:val_end])
    test_groups = set(group_values[val_end:])

    train = df[df[group_column].isin(train_groups)]
    val = df[df[group_column].isin(val_groups)]
    test = df[df[group_column].isin(test_groups)]
    return train, val, test


def resolve_smiles(cid, local_map, use_pubchem_fallback):
    normalized_cid = normalize_cid(cid)
    if normalized_cid is None:
        return None
    if normalized_cid in local_map:
        return local_map[normalized_cid]
    if use_pubchem_fallback:
        return cid_to_smiles(normalized_cid)
    return None


def is_valid_smiles(smiles):
    if pd.isna(smiles):
        return False
    text = str(smiles).strip()
    if not text:
        return False
    return Chem.MolFromSmiles(text) is not None

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw metabolism CSV (CID→SMILES, maps, splits).")
    parser.add_argument("--input", default="MetXBioDB_Transformations.csv", help="Input CSV path")
    parser.add_argument(
        "--substances",
        default="MetXBioDB_substances.csv",
        help="Local substances CSV containing PubChem_CID and SMILES columns",
    )
    parser.add_argument(
        "--metadata",
        default="metxbiodb.csv",
        help="Supplementary metadata CSV keyed by biotid",
    )
    parser.add_argument("--out_dir", default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for splits")
    parser.add_argument(
        "--pubchem_fallback",
        action="store_true",
        help="If a CID is missing from the local substances file, query PubChem as a fallback",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = read_csv_with_fallback(args.input)
    local_smiles_map = build_local_smiles_lookup(args.substances)
    metadata_df, metadata_duplicate_count = build_metadata_table(args.metadata)

    print(f"→ Loaded {len(local_smiles_map)} local CID→SMILES mappings")
    print(f"→ Loaded {len(metadata_df)} metadata rows from {args.metadata}")
    if metadata_duplicate_count:
        print(f"→ Dropped {metadata_duplicate_count} duplicate metadata rows by biotid")
    print("→ Resolving SMILES...")

    parent_smiles = []
    metabolite_smiles = []

    for i, row in df.iterrows():
        p = resolve_smiles(row["Predecessor_CID"], local_smiles_map, args.pubchem_fallback)
        m = resolve_smiles(row["Successor_CID"], local_smiles_map, args.pubchem_fallback)
        parent_smiles.append(p)
        metabolite_smiles.append(m)

    df["Parent_SMILES"] = parent_smiles
    df["Metabolite_SMILES"] = metabolite_smiles

    # Drop rows missing SMILES
    df = df.dropna(subset=["Parent_SMILES", "Metabolite_SMILES"])

    df["Source_ID"] = df["Source_ID"].astype(str).str.strip()
    df = df.merge(metadata_df, how="left", left_on="Source_ID", right_on="biotid")
    if "biotid" in df.columns:
        df = df.drop(columns=["biotid"])

    before_label_drop = len(df)
    df = df.dropna(subset=["Transformation", "Enzyme"])
    dropped_missing_labels = before_label_drop - len(df)

    valid_parent = df["Parent_SMILES"].map(is_valid_smiles)
    valid_metabolite = df["Metabolite_SMILES"].map(is_valid_smiles)
    invalid_structure_rows = int((~(valid_parent & valid_metabolite)).sum())
    df = df[valid_parent & valid_metabolite].copy()

    duplicate_subset = ["Parent_SMILES", "Metabolite_SMILES", "Transformation", "Enzyme"]
    duplicate_rows = int(df.duplicated(subset=duplicate_subset).sum())
    if duplicate_rows:
        df = df.drop_duplicates(subset=duplicate_subset, keep="first").copy()

    print(f"→ Valid rows: {len(df)}")
    if dropped_missing_labels:
        print(f"→ Dropped {dropped_missing_labels} rows missing Transformation/Enzyme labels")
    if invalid_structure_rows:
        print(f"→ Dropped {invalid_structure_rows} rows with RDKit-invalid SMILES")
    if duplicate_rows:
        print(f"→ Dropped {duplicate_rows} exact duplicate transformation rows")

    # Create maps
    transform_map = {t: i for i, t in enumerate(df["Transformation"].unique())}
    enzyme_map = {e: i for i, e in enumerate(df["Enzyme"].unique())}

    # Save maps
    with open(os.path.join(args.out_dir, "transform_map.json"), "w") as f:
        json.dump(transform_map, f, indent=2)

    with open(os.path.join(args.out_dir, "enzyme_map.json"), "w") as f:
        json.dump(enzyme_map, f, indent=2)

    # Save processed file
    df.to_csv(os.path.join(args.out_dir, "processed_metabolism_data.csv"), index=False)

    # Train/val/test split by precursor to avoid leakage across the same parent molecule.
    train, val, test = split_by_group(df, group_column="Parent_SMILES", seed=args.seed)

    train.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)

    print(f"→ Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

    print("→ Preprocessing completed.")


if __name__ == "__main__":
    main()