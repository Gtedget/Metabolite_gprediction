# data_utils.py
import torch
from torch_geometric.data import Dataset, Data
from rdkit import Chem
from rdkit.Chem import rdFMCS
import pandas as pd
import selfies as sf
import re


ATOM_FEATURE_DIM = 9
BOND_FEATURE_DIM = 6


def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def atom_to_feature_vector(atom):
    hybridization = atom.GetHybridization()
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 4.0,
        atom.GetFormalCharge(),
        1.0 if atom.GetIsAromatic() else 0.0,
        atom.GetTotalNumHs(includeNeighbors=True) / 4.0,
        1.0 if atom.IsInRing() else 0.0,
        1.0 if hybridization == Chem.rdchem.HybridizationType.SP else 0.0,
        1.0 if hybridization == Chem.rdchem.HybridizationType.SP2 else 0.0,
        1.0 if hybridization == Chem.rdchem.HybridizationType.SP3 else 0.0,
    ]


def bond_to_feature_vector(bond):
    bond_type = bond.GetBondType()
    return [
        1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0,
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0,
    ]


def normalize_transformation_family(transformation):
    if pd.isna(transformation):
        return "unknown"
    text = str(transformation).strip()
    if not text:
        return "unknown"
    family = text.split("/")[0].strip().lower()
    family = family.split(";")[0].strip()
    family = re.sub(r"\s*\(pattern\d+\)", "", family)
    family = re.sub(r"\s*\([^)]*\)", "", family)
    family = re.sub(r"\s+of\s+.*$", "", family)
    family = re.sub(r"\s+", " ", family).strip()
    return family or "unknown"


def infer_reaction_center_targets(parent_smiles, metabolite_smiles):
    parent_mol = Chem.MolFromSmiles(str(parent_smiles)) if pd.notna(parent_smiles) else None
    metabolite_mol = Chem.MolFromSmiles(str(metabolite_smiles)) if pd.notna(metabolite_smiles) else None
    if parent_mol is None or metabolite_mol is None:
        return None

    num_parent_atoms = parent_mol.GetNumAtoms()
    targets = torch.zeros(num_parent_atoms, dtype=torch.float)
    if num_parent_atoms == 0:
        return targets

    try:
        mcs = rdFMCS.FindMCS(
            [parent_mol, metabolite_mol],
            timeout=2,
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
            matchValences=False,
        )
    except Exception:
        return targets

    if not mcs.smartsString:
        return torch.ones(num_parent_atoms, dtype=torch.float)

    core = Chem.MolFromSmarts(mcs.smartsString)
    if core is None:
        return targets

    parent_match = parent_mol.GetSubstructMatch(core)
    metabolite_match = metabolite_mol.GetSubstructMatch(core)
    if not parent_match or not metabolite_match:
        return torch.ones(num_parent_atoms, dtype=torch.float)

    parent_to_metabolite = {int(p_idx): int(m_idx) for p_idx, m_idx in zip(parent_match, metabolite_match)}
    matched_parent_atoms = set(parent_to_metabolite.keys())

    for atom_idx in range(num_parent_atoms):
        if atom_idx not in matched_parent_atoms:
            targets[atom_idx] = 1.0

    for parent_idx, metabolite_idx in parent_to_metabolite.items():
        parent_atom = parent_mol.GetAtomWithIdx(parent_idx)
        metabolite_atom = metabolite_mol.GetAtomWithIdx(metabolite_idx)
        if parent_atom.GetFormalCharge() != metabolite_atom.GetFormalCharge():
            targets[parent_idx] = 1.0
            continue
        if parent_atom.GetTotalNumHs(includeNeighbors=True) != metabolite_atom.GetTotalNumHs(includeNeighbors=True):
            targets[parent_idx] = 1.0
            continue

        for neighbor in parent_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in parent_to_metabolite:
                targets[parent_idx] = 1.0
                break

            parent_bond = parent_mol.GetBondBetweenAtoms(parent_idx, neighbor_idx)
            mapped_neighbor = parent_to_metabolite[neighbor_idx]
            metabolite_bond = metabolite_mol.GetBondBetweenAtoms(metabolite_idx, mapped_neighbor)
            if metabolite_bond is None or parent_bond.GetBondType() != metabolite_bond.GetBondType():
                targets[parent_idx] = 1.0
                break

    if torch.count_nonzero(targets) == 0:
        return targets

    expanded_targets = targets.clone()
    changed_atoms = torch.nonzero(targets).view(-1).tolist()
    for atom_idx in changed_atoms:
        atom = parent_mol.GetAtomWithIdx(int(atom_idx))
        for neighbor in atom.GetNeighbors():
            expanded_targets[neighbor.GetIdx()] = 1.0

    return expanded_targets

class SmilesTokenizer:
    """Tokenizer for sequence generation with SMILES or SELFIES targets."""
    def __init__(self, vocab=None, representation="selfies", max_len=128):
        self.representation = representation
        self.max_len = max_len
        self.special = ["<pad>", "<sos>", "<eos>"]
        if vocab is None:
            if representation == "smiles":
                vocab = sorted(list(set(list(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "abcdefghijklmnopqrstuvwxyz"
                    "0123456789#+-=()[]/@\\%."
                ))))
            else:
                vocab = []
        vocab = [token for token in vocab if token not in self.special]
        self.vocab = self.special + list(vocab)
        self.token2id = {c: i for i, c in enumerate(self.vocab)}
        self.id2token = {i: c for i, c in enumerate(self.vocab)}

    @classmethod
    def from_smiles_list(cls, smiles_list, representation="selfies", max_len=128):
        vocab = set()
        tokenizer = cls(vocab=[], representation=representation, max_len=max_len)
        for smiles in smiles_list:
            for token in tokenizer.tokenize(smiles):
                vocab.add(token)
        return cls(vocab=sorted(vocab), representation=representation, max_len=max_len)

    @classmethod
    def from_config(cls, config):
        return cls(
            vocab=config.get("vocab", []),
            representation=config.get("representation", "selfies"),
            max_len=int(config.get("max_len", 128)),
        )

    def to_config(self):
        return {
            "representation": self.representation,
            "max_len": self.max_len,
            "vocab": self.vocab[len(self.special):],
        }

    def tokenize(self, smiles):
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            return []
        if self.representation == "selfies":
            selfies_str = sf.encoder(canonical)
            return list(sf.split_selfies(selfies_str))
        return list(canonical)

    def encode(self, smiles, max_len=None):
        max_len = max_len or self.max_len
        tokens = ["<sos>"] + self.tokenize(smiles)[:max_len-2] + ["<eos>"]
        ids = [self.token2id[t] for t in tokens if t in self.token2id]
        return torch.tensor(ids + [0]*(max_len-len(ids)), dtype=torch.long)

    def decode(self, ids):
        tokens = [self.id2token[i] for i in ids]
        sequence_tokens = [t for t in tokens if t not in ["<pad>", "<sos>", "<eos>"]]
        if self.representation == "selfies":
            if not sequence_tokens:
                return ""
            try:
                smiles = sf.decoder("".join(sequence_tokens))
            except Exception:
                return ""
            canonical = canonicalize_smiles(smiles)
            return canonical if canonical is not None else smiles
        return "".join(sequence_tokens)


def smiles_to_graph(smiles, atom_feature_dim=ATOM_FEATURE_DIM, bond_feature_dim=BOND_FEATURE_DIM):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feats = []
    for atom in mol.GetAtoms():
        if atom_feature_dim == 1:
            atom_feats.append([atom.GetAtomicNum()])
        else:
            atom_feats.append(atom_to_feature_vector(atom))

    # Edges (bidirectional)
    edge_index = [[], []]
    edge_attr = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        bond_features = bond_to_feature_vector(bond)
        edge_index[0] += [a, b]
        edge_index[1] += [b, a]
        edge_attr += [bond_features, bond_features]

    x = torch.tensor(atom_feats, dtype=torch.float)
    if edge_attr:
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, bond_feature_dim), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr_tensor)


class MetaboliteDataset(Dataset):
    """
    Dataset that returns:
    - Graph of precursor molecule
    - Tokenized metabolite SMILES
    - Transformation label
    - Enzyme label
    """
    def __init__(self, df, smiles_lookup_fn, tokenizer, transform_map, enzyme_map, coarse_transform_map=None):
        super().__init__()
        self.df = df
        self.lookup = smiles_lookup_fn
        self.tokenizer = tokenizer
        self.transform_map = transform_map
        self.enzyme_map = enzyme_map
        self.coarse_transform_map = coarse_transform_map or {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Prefer precomputed SMILES columns (from preprocessing), otherwise fall back to CID lookup.
        parent_smiles = None
        metabolite_smiles = None

        if "Parent_SMILES" in row and pd.notna(row["Parent_SMILES"]):
            parent_smiles = row["Parent_SMILES"]
        if "Metabolite_SMILES" in row and pd.notna(row["Metabolite_SMILES"]):
            metabolite_smiles = row["Metabolite_SMILES"]

        if parent_smiles is None:
            parent_smiles = self.lookup(row["Predecessor_CID"])
        if metabolite_smiles is None:
            metabolite_smiles = self.lookup(row["Successor_CID"])

        graph = smiles_to_graph(parent_smiles)
        if graph is None:
            raise ValueError(f"Invalid precursor SMILES at idx={idx}: {parent_smiles!r}")

        reaction_center_target = infer_reaction_center_targets(parent_smiles, metabolite_smiles)
        if reaction_center_target is None:
            reaction_center_target = torch.zeros(graph.num_nodes, dtype=torch.float)
        graph.reaction_center_target = reaction_center_target

        metabolite_tokens = self.tokenizer.encode(metabolite_smiles)

        y_transform = torch.tensor(self.transform_map[row["Transformation"]])
        coarse_family = normalize_transformation_family(row["Transformation"])
        y_coarse_transform = torch.tensor(self.coarse_transform_map.get(coarse_family, 0))
        y_enzyme = torch.tensor(self.enzyme_map[row["Enzyme"]])

        return graph, metabolite_tokens, y_transform, y_coarse_transform, y_enzyme