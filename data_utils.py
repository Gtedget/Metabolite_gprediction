# data_utils.py
import torch
from torch_geometric.data import Dataset, Data
from rdkit import Chem
import pandas as pd
import selfies as sf


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


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features (basic: atomic number)
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append([atom.GetAtomicNum()])

    # Edges (bidirectional)
    edge_index = [[], []]
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        edge_index[0] += [a, b]
        edge_index[1] += [b, a]

    x = torch.tensor(atom_feats, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


class MetaboliteDataset(Dataset):
    """
    Dataset that returns:
    - Graph of precursor molecule
    - Tokenized metabolite SMILES
    - Transformation label
    - Enzyme label
    """
    def __init__(self, df, smiles_lookup_fn, tokenizer, transform_map, enzyme_map):
        super().__init__()
        self.df = df
        self.lookup = smiles_lookup_fn
        self.tokenizer = tokenizer
        self.transform_map = transform_map
        self.enzyme_map = enzyme_map

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

        # Tokenize metabolite SMILES
        metabolite_tokens = self.tokenizer.encode(metabolite_smiles)

        # Get labels
        y_transform = torch.tensor(self.transform_map[row["Transformation"]])
        y_enzyme = torch.tensor(self.enzyme_map[row["Enzyme"]])

        return graph, metabolite_tokens, y_transform, y_enzyme