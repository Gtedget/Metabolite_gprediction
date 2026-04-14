# inference.py
import torch
from model import MetaboliteGenerator
from data_utils import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, SmilesTokenizer, canonicalize_smiles, normalize_transformation_family, smiles_to_graph
import json
import argparse
import math
import os
from typing import Optional
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

MAX_LEN = 128


def _creates_repeated_ngram(tokens, next_token_id, ngram_size):
    if ngram_size <= 1:
        return False
    if len(tokens) + 1 < ngram_size:
        return False

    candidate_ngram = tuple(tokens[-(ngram_size - 1):] + [int(next_token_id)])
    seen_ngrams = set()
    for start_idx in range(len(tokens) - ngram_size + 1):
        seen_ngrams.add(tuple(tokens[start_idx:start_idx + ngram_size]))
    return candidate_ngram in seen_ngrams

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model(model_path: str, metadata_path: Optional[str], device: torch.device):
    tokenizer = None

    num_transform_classes = 20
    num_coarse_transform_classes = 1
    num_enzyme_classes = 50
    use_enzyme_head = False
    metadata = {}

    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        num_transform_classes = int(metadata.get("num_transform_classes", num_transform_classes))
        num_coarse_transform_classes = int(metadata.get("num_coarse_transform_classes", num_coarse_transform_classes))
        num_enzyme_classes = int(metadata.get("num_enzyme_classes", num_enzyme_classes))
        use_enzyme_head = bool(metadata.get("use_enzyme_head", use_enzyme_head))
        tokenizer_config = metadata.get("tokenizer")
        if tokenizer_config:
            tokenizer = SmilesTokenizer.from_config(tokenizer_config)

    if tokenizer is None:
        tokenizer = SmilesTokenizer(representation=metadata.get("representation", "smiles"))

    state = torch.load(model_path, map_location=device)

    atom_feature_dim = int(metadata.get("atom_feature_dim", ATOM_FEATURE_DIM))
    encoder_weight = state.get("encoder.gat1.lin.weight")
    if encoder_weight is not None:
        atom_feature_dim = int(encoder_weight.shape[1])
    use_edge_features = "encoder.gat1.lin_edge.weight" in state
    bond_feature_dim = int(metadata.get("bond_feature_dim", BOND_FEATURE_DIM)) if use_edge_features else None

    model = MetaboliteGenerator(
        vocab_size=len(tokenizer.vocab),
        num_transform_classes=num_transform_classes,
        num_coarse_transform_classes=max(1, num_coarse_transform_classes),
        num_enzyme_classes=num_enzyme_classes,
        atom_feature_dim=atom_feature_dim,
        bond_feature_dim=bond_feature_dim,
        max_len=tokenizer.max_len,
        use_enzyme_head=use_enzyme_head,
    )

    missing_keys, _ = model.load_state_dict(state, strict=False)
    if "position_emb.weight" in missing_keys:
        with torch.no_grad():
            model.position_emb.weight.zero_()
    if "coarse_transform_emb.weight" in missing_keys:
        with torch.no_grad():
            model.coarse_transform_emb.weight.zero_()
    if "transform_emb.weight" in missing_keys:
        with torch.no_grad():
            model.transform_emb.weight.zero_()
    model.to(device)
    model.eval()
    return model, tokenizer, metadata


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _prepare_graph(model, precursor_smiles: str, device: torch.device):
    atom_feature_dim = int(model.encoder.gat1.lin.weight.shape[1])
    graph = smiles_to_graph(precursor_smiles, atom_feature_dim=atom_feature_dim)
    if graph is None:
        raise ValueError("Invalid precursor SMILES")

    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    return graph.to(device)


@torch.no_grad()
def predict_sites_of_metabolism(model, precursor_smiles, device: torch.device, top_k=5, threshold=0.5):
    graph = _prepare_graph(model, precursor_smiles, device)
    mol = Chem.MolFromSmiles(precursor_smiles)
    if mol is None:
        raise ValueError("Invalid precursor SMILES")

    sos_tokens = torch.tensor([[1]], dtype=torch.long, device=device)
    _, _, _, pred_reaction_center, _ = model(graph, sos_tokens)
    probabilities = torch.sigmoid(pred_reaction_center).detach().cpu().tolist()

    ranked_atoms = []
    for atom_idx, probability in enumerate(probabilities):
        atom = mol.GetAtomWithIdx(atom_idx)
        ranked_atoms.append(
            {
                "atom_index": int(atom_idx),
                "atom_symbol": atom.GetSymbol(),
                "probability": float(probability),
                "is_predicted_site": bool(probability >= threshold),
            }
        )

    ranked_atoms.sort(key=lambda item: item["probability"], reverse=True)
    return ranked_atoms[: max(1, min(top_k, len(ranked_atoms)))], graph


def render_sites_of_metabolism_svg(precursor_smiles, site_predictions, output_path, threshold=0.5):
    mol = Chem.MolFromSmiles(precursor_smiles)
    if mol is None:
        raise ValueError("Invalid precursor SMILES")

    display_mol = Chem.Mol(mol)
    for atom in display_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())

    highlighted_atoms = [item["atom_index"] for item in site_predictions]
    atom_colors = {}
    for item in site_predictions:
        atom_idx = item["atom_index"]
        probability = float(item["probability"])
        if probability >= threshold:
            atom_colors[atom_idx] = (0.90, 0.35, 0.15)
        else:
            atom_colors[atom_idx] = (0.98, 0.78, 0.22)

    drawer = rdMolDraw2D.MolDraw2DSVG(900, 600)
    options = drawer.drawOptions()
    options.addAtomIndices = True
    options.legendFontSize = 20
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        display_mol,
        highlightAtoms=highlighted_atoms,
        highlightAtomColors=atom_colors,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)

    return output_path


def predict_top_transformations(model, metadata, graph, top_k=3):
    sos_id = 1
    _, pred_transform, _, _, _ = model(
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
            "transformation_id": int(index),
            "transformation": id_to_transform.get(int(index), str(int(index))),
            "coarse_family": normalize_transformation_family(id_to_transform.get(int(index), str(int(index)))),
            "probability": float(value),
        }
        for value, index in zip(top_values.tolist(), top_indices.tolist())
    ]


@torch.no_grad()
def _beam_search_for_transform(
    model,
    tokenizer,
    graph,
    device,
    transform_label=None,
    coarse_transform_label=None,
    transform_log_prior=0.0,
    beam_width=5,
    max_len=MAX_LEN,
    no_repeat_ngram_size=3,
):
    sos_id = tokenizer.token2id["<sos>"]
    eos_id = tokenizer.token2id["<eos>"]
    pad_id = tokenizer.token2id["<pad>"]
    beams = [(float(transform_log_prior), [sos_id], False)]
    finished = []

    conditioned_transform = None
    conditioned_coarse_transform = None
    if transform_label is not None:
        conditioned_transform = torch.tensor([transform_label], dtype=torch.long, device=device)
    if coarse_transform_label is not None:
        conditioned_coarse_transform = torch.tensor([coarse_transform_label], dtype=torch.long, device=device)

    for _ in range(max_len or tokenizer.max_len):
        next_beams = []
        active_found = False

        for score, tokens, done in beams:
            if done:
                finished.append((score, tokens, done))
                next_beams.append((score, tokens, done))
                continue

            active_found = True
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            logits, _, _, _, _ = model(
                graph,
                x,
                transform_labels=conditioned_transform,
                coarse_transform_labels=conditioned_coarse_transform,
            )
            next_token_logits = logits[0, -1]
            next_token_logits[pad_id] = float("-inf")
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            top_log_probs, top_ids = torch.topk(log_probs, k=min(beam_width, log_probs.shape[-1]))
            fallback_expansions = []

            for log_prob, token_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                updated_tokens = tokens + [int(token_id)]
                token_done = int(token_id) == eos_id
                fallback_expansions.append((score + float(log_prob), updated_tokens, token_done))
                if _creates_repeated_ngram(tokens, token_id, no_repeat_ngram_size):
                    continue
                next_beams.append((score + float(log_prob), updated_tokens, token_done))

            if not any(expansion[1][:-1] == tokens for expansion in next_beams):
                next_beams.extend(fallback_expansions)

        if not active_found:
            break

        next_beams.sort(key=lambda item: item[0] / max(1, len(item[1])), reverse=True)
        beams = next_beams[:beam_width]

    finished.extend([beam for beam in beams if beam[2]])
    return finished if finished else beams


@torch.no_grad()
def beam_search_candidates(
    model,
    tokenizer,
    precursor_smiles,
    device: torch.device,
    metadata=None,
    num_candidates=5,
    beam_width=5,
    max_len=MAX_LEN,
    no_repeat_ngram_size=3,
):
    metadata = metadata or {}
    graph = _prepare_graph(model, precursor_smiles, device)

    max_len = max_len or tokenizer.max_len
    top_transformations = predict_top_transformations(model, metadata=metadata, graph=graph, top_k=max(1, min(num_candidates, beam_width)))
    finished = []
    coarse_transform_map = metadata.get("coarse_transform_map", {})

    if top_transformations:
        for item in top_transformations:
            transform_log_prior = math.log(max(item["probability"], 1e-12))
            coarse_transform_label = coarse_transform_map.get(item["coarse_family"])
            transform_finished = _beam_search_for_transform(
                model,
                tokenizer,
                graph,
                device,
                transform_label=item["transformation_id"],
                coarse_transform_label=coarse_transform_label,
                transform_log_prior=transform_log_prior,
                beam_width=beam_width,
                max_len=max_len,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            finished.extend((score, tokens, done, item["transformation"]) for score, tokens, done in transform_finished)
    else:
        transform_finished = _beam_search_for_transform(
            model,
            tokenizer,
            graph,
            device,
            transform_label=None,
            beam_width=beam_width,
            max_len=max_len,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        finished.extend((score, tokens, done, None) for score, tokens, done in transform_finished)

    ranked_candidates = []
    seen = set()
    for score, tokens, _, transformation_name in sorted(finished, key=lambda item: item[0] / max(1, len(item[1])), reverse=True):
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
                "transformation": transformation_name,
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
        metadata=None,
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
    parser.add_argument("--precursor", default=None, help="Optional precursor SMILES. If omitted, inference prompts interactively.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of candidate metabolites to return")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for decoding")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="Block repeated n-grams during decoding to reduce degenerate metabolite strings")
    parser.add_argument("--som_top_k", type=int, default=5, help="Number of highest-scoring SoM-like atoms to report")
    parser.add_argument("--som_threshold", type=float, default=0.5, help="Probability threshold used to flag a predicted site of metabolism")
    parser.add_argument("--json_out", default=None, help="Optional JSON path to save metabolite, transformation, and SoM predictions")
    parser.add_argument("--som_svg_out", default=None, help="Optional SVG path to render highlighted SoM-like atoms on the precursor molecule")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    model, tokenizer, metadata = load_model(args.model, args.metadata, device)

    precursor = args.precursor.strip() if args.precursor else input("Enter precursor SMILES: ").strip()
    candidates, graph = beam_search_candidates(
        model,
        tokenizer,
        precursor,
        device,
        metadata=metadata,
        num_candidates=args.top_k,
        beam_width=args.beam_width,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    top_transformations = predict_top_transformations(model, metadata, graph, top_k=3)
    top_sites_of_metabolism, _ = predict_sites_of_metabolism(
        model,
        precursor,
        device,
        top_k=args.som_top_k,
        threshold=args.som_threshold,
    )

    print("\nPredicted metabolite candidates:")
    for index, candidate in enumerate(candidates, start=1):
        validity = "valid" if candidate["is_valid"] else "invalid"
        print(f"{index}. {candidate['smiles']} ({validity}, score={candidate['score']:.4f})")

    if top_transformations:
        print("\nTop predicted transformations:")
        for item in top_transformations:
            print(f"- {item['transformation']} ({item['probability']:.4f})")

    if top_sites_of_metabolism:
        print("\nTop predicted sites of metabolism:")
        for item in top_sites_of_metabolism:
            flag = "predicted" if item["is_predicted_site"] else "below-threshold"
            print(
                f"- atom {item['atom_index']} [{item['atom_symbol']}] "
                f"prob={item['probability']:.4f} ({flag})"
            )

    if args.json_out:
        out_dir = os.path.dirname(os.path.abspath(args.json_out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "precursor": precursor,
            "metabolite_candidates": candidates,
            "transformations": top_transformations,
            "sites_of_metabolism": top_sites_of_metabolism,
            "som_threshold": args.som_threshold,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved predictions to {args.json_out}")

    if args.som_svg_out:
        render_sites_of_metabolism_svg(
            precursor,
            top_sites_of_metabolism,
            args.som_svg_out,
            threshold=args.som_threshold,
        )
        print(f"Saved SoM visualization to {args.som_svg_out}")


if __name__ == "__main__":
    main()