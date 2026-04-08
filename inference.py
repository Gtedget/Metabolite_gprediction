# inference.py
import torch
from model import MetaboliteGenerator
from data_utils import SmilesTokenizer, canonicalize_smiles, smiles_to_graph
import json
import argparse
import os
from typing import Optional
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

MAX_LEN = 128

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model(model_path: str, metadata_path: Optional[str], device: torch.device):
    tokenizer = None

    num_transform_classes = 20
    num_enzyme_classes = 50
    use_enzyme_head = False
    metadata = {}

    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        num_transform_classes = int(metadata.get("num_transform_classes", num_transform_classes))
        num_enzyme_classes = int(metadata.get("num_enzyme_classes", num_enzyme_classes))
        use_enzyme_head = bool(metadata.get("use_enzyme_head", use_enzyme_head))
        tokenizer_config = metadata.get("tokenizer")
        if tokenizer_config:
            tokenizer = SmilesTokenizer.from_config(tokenizer_config)

    if tokenizer is None:
        tokenizer = SmilesTokenizer(representation=metadata.get("representation", "smiles"))

    model = MetaboliteGenerator(
        vocab_size=len(tokenizer.vocab),
        num_transform_classes=num_transform_classes,
        num_enzyme_classes=num_enzyme_classes,
        use_enzyme_head=use_enzyme_head,
    )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer, metadata


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _prepare_graph(precursor_smiles: str, device: torch.device):
    graph = smiles_to_graph(precursor_smiles)
    if graph is None:
        raise ValueError("Invalid precursor SMILES")

    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    return graph.to(device)


def predict_top_transformations(model, metadata, graph, top_k=3):
    sos_id = 1
    _, pred_transform, _ = model(
        graph,
        torch.tensor([[sos_id]], dtype=torch.long, device=graph.x.device),
    )
    transform_map = metadata.get("transform_map", {})
    id_to_transform = {int(value): key for key, value in transform_map.items()}
    if not id_to_transform:
        return []

    top_values, top_indices = torch.topk(torch.softmax(pred_transform[0], dim=0), k=min(top_k, pred_transform.shape[-1]))
    return [
        {
            "transformation": id_to_transform.get(int(index), str(int(index))),
            "probability": float(value),
        }
        for value, index in zip(top_values.tolist(), top_indices.tolist())
    ]


@torch.no_grad()
def beam_search_candidates(
    model,
    tokenizer,
    precursor_smiles,
    device: torch.device,
    num_candidates=5,
    beam_width=5,
    max_len=MAX_LEN,
):
    graph = _prepare_graph(precursor_smiles, device)
    sos_id = tokenizer.token2id["<sos>"]
    eos_id = tokenizer.token2id["<eos>"]

    beams = [(0.0, [sos_id], False)]
    finished = []

    max_len = max_len or tokenizer.max_len
    for _ in range(max_len):
        next_beams = []
        active_found = False

        for score, tokens, done in beams:
            if done:
                finished.append((score, tokens, done))
                next_beams.append((score, tokens, done))
                continue

            active_found = True
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            logits, _, _ = model(graph, x)
            next_token_logits = logits[0, -1]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            top_log_probs, top_ids = torch.topk(log_probs, k=min(beam_width, log_probs.shape[-1]))

            for log_prob, token_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                updated_tokens = tokens + [int(token_id)]
                token_done = int(token_id) == eos_id
                next_beams.append((score + float(log_prob), updated_tokens, token_done))

        if not active_found:
            break

        next_beams.sort(key=lambda item: item[0] / max(1, len(item[1])), reverse=True)
        beams = next_beams[:beam_width]

    finished.extend([beam for beam in beams if beam[2]])
    if not finished:
        finished = beams

    ranked_candidates = []
    seen = set()
    for score, tokens, _ in sorted(finished, key=lambda item: item[0] / max(1, len(item[1])), reverse=True):
        decoded = tokenizer.decode(tokens)
        canonical = canonicalize_smiles(decoded)
        candidate_key = canonical if canonical is not None else decoded
        if not decoded or candidate_key in seen:
            continue
        seen.add(candidate_key)
        ranked_candidates.append(
            {
                "smiles": decoded,
                "canonical_smiles": canonical,
                "score": float(score / max(1, len(tokens))),
                "is_valid": canonical is not None,
            }
        )
        if len(ranked_candidates) >= num_candidates:
            break

    return ranked_candidates, graph

@torch.no_grad()
def sample_tokens(model, tokenizer, precursor_smiles, device: torch.device):
    candidates, _ = beam_search_candidates(
        model,
        tokenizer,
        precursor_smiles,
        device,
        num_candidates=1,
        beam_width=1,
        max_len=tokenizer.max_len,
    )
    return candidates[0]["smiles"] if candidates else ""

def main():
    parser = argparse.ArgumentParser(description="Run metabolite SMILES generation inference")
    parser.add_argument("--model", default="trained_model.pt", help="Path to model weights")
    parser.add_argument(
        "--metadata",
        default="trained_model.metadata.json",
        help="Path to metadata JSON saved during training",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of candidate metabolites to return")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for decoding")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    model, tokenizer, metadata = load_model(args.model, args.metadata, device)

    precursor = input("Enter precursor SMILES: ").strip()
    candidates, graph = beam_search_candidates(
        model,
        tokenizer,
        precursor,
        device,
        num_candidates=args.top_k,
        beam_width=args.beam_width,
    )
    top_transformations = predict_top_transformations(model, metadata, graph, top_k=3)

    print("\nPredicted metabolite candidates:")
    for index, candidate in enumerate(candidates, start=1):
        validity = "valid" if candidate["is_valid"] else "invalid"
        print(f"{index}. {candidate['smiles']} ({validity}, score={candidate['score']:.4f})")

    if top_transformations:
        print("\nTop predicted transformations:")
        for item in top_transformations:
            print(f"- {item['transformation']} ({item['probability']:.4f})")


if __name__ == "__main__":
    main()