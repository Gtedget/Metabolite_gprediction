"""Train a GAT+Transformer model for metabolite SMILES generation and transformation prediction."""

from contextlib import nullcontext
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from data_utils import BOND_FEATURE_DIM, ATOM_FEATURE_DIM, MetaboliteDataset, SmilesTokenizer, normalize_transformation_family
from model import MetaboliteGenerator
import pandas as pd
import argparse
import json
import os
from datetime import datetime
from cid_lookup import cid_to_smiles
from tqdm.auto import tqdm

try:
    import mlflow
except ImportError:
    mlflow = None


def resolve_default_path(data_path, filename):
    candidate = os.path.join(os.path.dirname(os.path.abspath(data_path)), filename)
    return candidate if os.path.exists(candidate) else None


def resolve_best_model_out(model_out):
    root, ext = os.path.splitext(model_out)
    if not ext:
        ext = ".pt"
    return f"{root}.best{ext}"


def default_run_name():
    return datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")


def resolve_output_path(output_dir, run_name, provided_path, default_filename):
    if provided_path and provided_path != default_filename:
        return provided_path
    return os.path.join(output_dir, run_name, default_filename)


def log_message(message, log_file=None):
    print(message, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def setup_mlflow(args, train_df, val_df=None, test_df=None):
    if not args.use_mlflow:
        return False
    if mlflow is None:
        raise ImportError("mlflow is not installed. Install requirements.txt before using --use_mlflow.")

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    if args.mlflow_experiment:
        mlflow.set_experiment(args.mlflow_experiment)

    mlflow.start_run(run_name=args.mlflow_run_name)
    mlflow.log_params(
        {
            "data": args.data,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "representation": args.representation,
            "max_len": args.max_len,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "encoder_hidden_dim": args.encoder_hidden_dim,
            "encoder_out_dim": args.encoder_out_dim,
            "encoder_heads": args.encoder_heads,
            "decoder_heads": args.decoder_heads,
            "dropout": args.dropout,
            "run_name": args.run_name,
            "output_dir": args.output_dir,
            "device": args.device,
            "use_enzyme_head": args.use_enzyme_head,
            "oversample_strategy": args.oversample_strategy,
            "oversample_power": args.oversample_power,
            "scheduler": args.scheduler,
            "scheduler_patience": args.scheduler_patience,
            "scheduler_factor": args.scheduler_factor,
            "min_lr": args.min_lr,
            "early_stopping_patience": args.early_stopping_patience,
            "amp": args.amp,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "pin_memory": not args.disable_pin_memory,
            "train_rows": len(train_df),
            "val_rows": len(val_df) if val_df is not None else 0,
            "test_rows": len(test_df) if test_df is not None else 0,
        }
    )
    return True


def log_mlflow_metrics(metrics, step=None):
    if mlflow is None or mlflow.active_run() is None:
        return
    mlflow.log_metrics(metrics, step=step)


def log_mlflow_artifacts(paths):
    if mlflow is None or mlflow.active_run() is None:
        return
    for path in paths:
        if path and os.path.exists(path):
            mlflow.log_artifact(path)


def load_label_map(path, column_name, fallback_df):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            loaded = json.load(f)
        return {str(key): int(value) for key, value in loaded.items()}
    return {value: idx for idx, value in enumerate(fallback_df[column_name].dropna().unique())}


def build_tokenizer(args, train_df, val_df=None, test_df=None):
    tokenizer_data_path = resolve_default_path(args.data, "processed_metabolism_data.csv")
    if tokenizer_data_path and os.path.exists(tokenizer_data_path):
        tokenizer_df = pd.read_csv(tokenizer_data_path)
    else:
        frames = [train_df]
        if val_df is not None:
            frames.append(val_df)
        if test_df is not None:
            frames.append(test_df)
        tokenizer_df = pd.concat(frames, ignore_index=True)

    return SmilesTokenizer.from_smiles_list(
        tokenizer_df["Metabolite_SMILES"].dropna().tolist(),
        representation=args.representation,
        max_len=args.max_len,
    )


def build_class_weights(df, column_name, label_map):
    counts = df[column_name].value_counts()
    num_classes = max(1, len(label_map))
    total = max(1, len(df))
    weights = []
    for label, index in sorted(label_map.items(), key=lambda item: item[1]):
        count = int(counts.get(label, 0))
        if count <= 0:
            weights.append(1.0)
            continue
        weights.append(total / (num_classes * count))

    weights = torch.tensor(weights, dtype=torch.float)
    return weights / weights.mean()


def build_sample_weights(df, strategy, power=1.0):
    if strategy == "transform":
        labels = df["Transformation"].astype(str)
    elif strategy == "coarse_transform":
        labels = df["Transformation"].map(normalize_transformation_family)
    else:
        raise ValueError(f"Unsupported oversampling strategy: {strategy}")

    counts = labels.value_counts()
    sample_weights = labels.map(lambda value: 1.0 / max(float(counts[value]), 1.0))
    sample_weights = sample_weights.pow(power)
    return torch.tensor(sample_weights.to_numpy(), dtype=torch.double)


def build_coarse_transform_map(train_df, val_df=None, test_df=None):
    frames = [train_df]
    if val_df is not None:
        frames.append(val_df)
    if test_df is not None:
        frames.append(test_df)

    coarse_labels = []
    for frame in frames:
        coarse_labels.extend(normalize_transformation_family(value) for value in frame["Transformation"].dropna().tolist())

    coarse_labels = sorted(set(coarse_labels))
    return {label: idx for idx, label in enumerate(coarse_labels)}


def get_autocast_context(device, amp_enabled):
    if amp_enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_loader_kwargs(batch_size, num_workers, pin_memory, prefetch_factor, sampler=None, shuffle=False):
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = shuffle

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return loader_kwargs


def compute_batch_loss(
    model,
    batch,
    sequence_criterion,
    transform_criterion,
    coarse_transform_criterion,
    reaction_center_criterion,
    enzyme_criterion,
    device,
    transform_loss_weight=0.5,
    coarse_transform_loss_weight=0.2,
    reaction_center_loss_weight=0.2,
    enzyme_loss_weight=0.3,
    use_enzyme_head=False,
):
    graph, tgt_tokens, y_transform, y_coarse_transform, y_enzyme = batch

    graph = graph.to(device)
    tgt_tokens = tgt_tokens.to(device)
    y_transform = y_transform.to(device)
    y_coarse_transform = y_coarse_transform.to(device)
    if use_enzyme_head:
        y_enzyme = y_enzyme.to(device)

    non_pad_lengths = tgt_tokens.ne(0).sum(dim=1)
    effective_tgt_len = max(2, int(non_pad_lengths.max().item()))
    tgt_tokens = tgt_tokens[:, :effective_tgt_len]

    logits, pred_transform, pred_coarse_transform, pred_reaction_center, pred_enzyme = model(
        graph,
        tgt_tokens[:, :-1],
        transform_labels=y_transform,
        coarse_transform_labels=y_coarse_transform,
    )

    loss_smiles = sequence_criterion(
        logits.reshape(-1, logits.size(-1)),
        tgt_tokens[:, 1:].reshape(-1)
    )

    loss_transform = transform_criterion(pred_transform, y_transform)
    loss_coarse_transform = coarse_transform_criterion(pred_coarse_transform, y_coarse_transform)
    reaction_center_targets = getattr(graph, "reaction_center_target", None)
    if reaction_center_targets is None:
        reaction_center_targets = torch.zeros_like(pred_reaction_center)
    reaction_center_targets = reaction_center_targets.to(device).float()
    loss_reaction_center = reaction_center_criterion(pred_reaction_center, reaction_center_targets)

    loss = (
        loss_smiles
        + transform_loss_weight * loss_transform
        + coarse_transform_loss_weight * loss_coarse_transform
        + reaction_center_loss_weight * loss_reaction_center
    )
    if use_enzyme_head and pred_enzyme is not None:
        loss_enzyme = enzyme_criterion(pred_enzyme, y_enzyme)
        loss = loss + enzyme_loss_weight * loss_enzyme

    transform_accuracy = (pred_transform.argmax(dim=1) == y_transform).float().mean().item()

    return loss, transform_accuracy


# -----------------------------------------------------
# 1. Training step
# -----------------------------------------------------
def train_step(
    model,
    batch,
    optimizer,
    sequence_criterion,
    transform_criterion,
    coarse_transform_criterion,
    reaction_center_criterion,
    enzyme_criterion,
    device,
    transform_loss_weight=0.5,
    coarse_transform_loss_weight=0.2,
    reaction_center_loss_weight=0.2,
    enzyme_loss_weight=0.3,
    grad_clip=0.0,
    scaler=None,
    amp_enabled=False,
    use_enzyme_head=False,
):
    optimizer.zero_grad(set_to_none=True)

    with get_autocast_context(device, amp_enabled):
        loss, transform_accuracy = compute_batch_loss(
            model,
            batch,
            sequence_criterion,
            transform_criterion,
            coarse_transform_criterion,
            reaction_center_criterion,
            enzyme_criterion,
            device,
            transform_loss_weight=transform_loss_weight,
            coarse_transform_loss_weight=coarse_transform_loss_weight,
            reaction_center_loss_weight=reaction_center_loss_weight,
            enzyme_loss_weight=enzyme_loss_weight,
            use_enzyme_head=use_enzyme_head,
        )

    if scaler is not None and amp_enabled:
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return loss.item(), transform_accuracy


@torch.no_grad()
def evaluate(
    model,
    loader,
    sequence_criterion,
    transform_criterion,
    coarse_transform_criterion,
    reaction_center_criterion,
    enzyme_criterion,
    device,
    transform_loss_weight=0.5,
    coarse_transform_loss_weight=0.2,
    reaction_center_loss_weight=0.2,
    enzyme_loss_weight=0.3,
    amp_enabled=False,
    use_enzyme_head=False,
    progress_desc=None,
    show_progress=False,
):
    model.eval()
    losses = []
    transform_accuracies = []

    batch_iter = loader
    if show_progress:
        batch_iter = tqdm(loader, desc=progress_desc, leave=False)

    for batch in batch_iter:
        with get_autocast_context(device, amp_enabled):
            loss, transform_accuracy = compute_batch_loss(
                model,
                batch,
                sequence_criterion,
                transform_criterion,
                coarse_transform_criterion,
                reaction_center_criterion,
                enzyme_criterion,
                device,
                transform_loss_weight=transform_loss_weight,
                coarse_transform_loss_weight=coarse_transform_loss_weight,
                reaction_center_loss_weight=reaction_center_loss_weight,
                enzyme_loss_weight=enzyme_loss_weight,
                use_enzyme_head=use_enzyme_head,
            )
        losses.append(loss.item())
        transform_accuracies.append(transform_accuracy)

    model.train()
    return {
        "loss": sum(losses) / len(losses) if losses else float("nan"),
        "transform_accuracy": sum(transform_accuracies) / len(transform_accuracies) if transform_accuracies else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train metabolite predictor")
    parser.add_argument("--data", default="train.csv", help="Training CSV path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Optional gradient clipping norm; disabled when <= 0")
    parser.add_argument("--representation", choices=["smiles", "selfies"], default="selfies")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256, help="Transformer hidden size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of decoder layers")
    parser.add_argument("--encoder_hidden_dim", type=int, default=64, help="Hidden size of the first GAT layer")
    parser.add_argument("--encoder_out_dim", type=int, default=128, help="Output size of the GAT encoder")
    parser.add_argument("--encoder_heads", type=int, default=4, help="Number of GAT attention heads")
    parser.add_argument("--decoder_heads", type=int, default=8, help="Number of transformer decoder heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout applied in the encoder and decoder")
    parser.add_argument("--output_dir", default="artifacts", help="Base directory for run outputs when default artifact names are used")
    parser.add_argument("--run_name", default=None, help="Optional run name used for artifact subdirectories and MLflow")
    parser.add_argument("--log_file", default=None, help="Optional path to append training progress logs")
    parser.add_argument("--val_data", default=None, help="Validation CSV path. Defaults to a sibling val.csv when present")
    parser.add_argument("--test_data", default=None, help="Test CSV path. Defaults to a sibling test.csv when present")
    parser.add_argument("--transform_map", default=None, help="Transformation label map JSON. Defaults to a sibling transform_map.json when present")
    parser.add_argument("--enzyme_map", default=None, help="Enzyme label map JSON. Defaults to a sibling enzyme_map.json when present")
    parser.add_argument("--model_out", default="trained_model.pt", help="Path to save model weights")
    parser.add_argument("--best_model_out", default=None, help="Path to save the best validation-loss checkpoint. Defaults to <model_out>.best.pt")
    parser.add_argument(
        "--metadata_out",
        default="trained_model.metadata.json",
        help="Path to save metadata (class counts, maps)",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--use_mlflow", action="store_true", help="Log params, metrics, and artifacts to MLflow")
    parser.add_argument("--mlflow_experiment", default="metabolite-predictor", help="MLflow experiment name")
    parser.add_argument("--mlflow_run_name", default=None, help="Optional MLflow run name")
    parser.add_argument("--mlflow_tracking_uri", default=None, help="MLflow tracking URI, e.g. file:/content/mlruns in Colab")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--transform_loss_weight", type=float, default=0.5, help="Weight applied to the transformation classification loss")
    parser.add_argument("--coarse_transform_loss_weight", type=float, default=0.2, help="Weight applied to the coarse transformation-family classification loss")
    parser.add_argument("--reaction_center_loss_weight", type=float, default=0.2, help="Weight applied to the reaction-center prediction loss")
    parser.add_argument("--enzyme_loss_weight", type=float, default=0.3, help="Weight applied to the enzyme classification loss when enabled")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Optional label smoothing applied to sequence and classification cross-entropy losses")
    parser.add_argument("--balance_transform_classes", action="store_true", help="Use inverse-frequency class weights for the transformation loss")
    parser.add_argument(
        "--oversample_strategy",
        choices=["none", "transform", "coarse_transform"],
        default="none",
        help="Optional weighted oversampling of minority classes in the training loader",
    )
    parser.add_argument(
        "--oversample_power",
        type=float,
        default=1.0,
        help="Exponent applied to inverse-frequency sample weights; 0.5 is gentler than 1.0",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "plateau", "cosine"],
        default="none",
        help="Optional learning-rate scheduler",
    )
    parser.add_argument("--scheduler_patience", type=int, default=3, help="Patience for ReduceLROnPlateau")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="Factor for ReduceLROnPlateau")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum scheduler learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor used when num_workers > 0")
    parser.add_argument("--disable_pin_memory", action="store_true", help="Disable DataLoader pin_memory even on CUDA")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA for faster Colab GPU training")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop training after this many non-improving validation epochs; disabled when <= 0",
    )
    parser.add_argument(
        "--use_enzyme_head",
        action="store_true",
        help="Enable the enzyme classification head. Disabled by default because enzyme labels are often multi-valued.",
    )
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = default_run_name()
    if args.mlflow_run_name is None:
        args.mlflow_run_name = args.run_name

    args.model_out = resolve_output_path(args.output_dir, args.run_name, args.model_out, "trained_model.pt")
    if args.best_model_out is None:
        args.best_model_out = os.path.join(args.output_dir, args.run_name, "trained_model.best.pt")
    args.metadata_out = resolve_output_path(args.output_dir, args.run_name, args.metadata_out, "trained_model.metadata.json")
    if args.log_file is None:
        args.log_file = os.path.join(args.output_dir, args.run_name, "train.log")

    for path in [args.model_out, args.best_model_out, args.metadata_out, args.log_file]:
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if args.log_file:
        with open(args.log_file, "w", encoding="utf-8"):
            pass

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    pin_memory = bool(device.type == "cuda" and not args.disable_pin_memory)

    if args.val_data is None:
        args.val_data = resolve_default_path(args.data, "val.csv")
    if args.test_data is None:
        args.test_data = resolve_default_path(args.data, "test.csv")
    if args.transform_map is None:
        args.transform_map = resolve_default_path(args.data, "transform_map.json")
    if args.enzyme_map is None:
        args.enzyme_map = resolve_default_path(args.data, "enzyme_map.json")
    if args.best_model_out is None:
        args.best_model_out = resolve_best_model_out(args.model_out)

    df = pd.read_csv(args.data)
    val_df = pd.read_csv(args.val_data) if args.val_data and os.path.exists(args.val_data) else None
    test_df = pd.read_csv(args.test_data) if args.test_data and os.path.exists(args.test_data) else None
    mlflow_enabled = setup_mlflow(args, df, val_df=val_df, test_df=test_df)

    tokenizer = build_tokenizer(args, df, val_df=val_df, test_df=test_df)

    # Example maps: must be generated from your dataset
    if "Transformation" not in df.columns or "Enzyme" not in df.columns:
        raise ValueError("Training CSV must include 'Transformation' and 'Enzyme' columns")

    transform_map = load_label_map(args.transform_map, "Transformation", df)
    enzyme_map = load_label_map(args.enzyme_map, "Enzyme", df)
    coarse_transform_map = build_coarse_transform_map(df, val_df=val_df, test_df=test_df)

    dataset = MetaboliteDataset(
        df=df,
        smiles_lookup_fn=cid_to_smiles,
        tokenizer=tokenizer,
        transform_map=transform_map,
        enzyme_map=enzyme_map,
        coarse_transform_map=coarse_transform_map,
    )

    if args.oversample_strategy != "none":
        train_sample_weights = build_sample_weights(df, args.oversample_strategy, power=args.oversample_power)
        train_sampler = WeightedRandomSampler(
            weights=train_sample_weights,
            num_samples=len(train_sample_weights),
            replacement=True,
        )
        loader_kwargs = build_loader_kwargs(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor,
            sampler=train_sampler,
        )
    else:
        loader_kwargs = build_loader_kwargs(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor,
            shuffle=True,
        )
    loader = DataLoader(dataset, **loader_kwargs)
    val_loader = None
    if val_df is not None:
        val_dataset = MetaboliteDataset(
            df=val_df,
            smiles_lookup_fn=cid_to_smiles,
            tokenizer=tokenizer,
            transform_map=transform_map,
            enzyme_map=enzyme_map,
            coarse_transform_map=coarse_transform_map,
        )
        val_loader = DataLoader(
            val_dataset,
            **build_loader_kwargs(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                prefetch_factor=args.prefetch_factor,
                shuffle=False,
            ),
        )

    test_loader = None
    if test_df is not None:
        test_dataset = MetaboliteDataset(
            df=test_df,
            smiles_lookup_fn=cid_to_smiles,
            tokenizer=tokenizer,
            transform_map=transform_map,
            enzyme_map=enzyme_map,
            coarse_transform_map=coarse_transform_map,
        )
        test_loader = DataLoader(
            test_dataset,
            **build_loader_kwargs(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                prefetch_factor=args.prefetch_factor,
                shuffle=False,
            ),
        )

    model = MetaboliteGenerator(
        vocab_size=len(tokenizer.vocab),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_out_dim=args.encoder_out_dim,
        encoder_heads=args.encoder_heads,
        decoder_heads=args.decoder_heads,
        num_transform_classes=len(transform_map),
        num_coarse_transform_classes=len(coarse_transform_map),
        num_enzyme_classes=len(enzyme_map),
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        max_len=tokenizer.max_len,
        dropout=args.dropout,
        use_enzyme_head=args.use_enzyme_head,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_lr,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.min_lr,
        )
    sequence_criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing)
    transform_class_weights = None
    if args.balance_transform_classes:
        transform_class_weights = build_class_weights(df, "Transformation", transform_map).to(device)
    transform_criterion = nn.CrossEntropyLoss(weight=transform_class_weights, label_smoothing=args.label_smoothing)
    coarse_transform_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    reaction_center_criterion = nn.BCEWithLogitsLoss()
    enzyme_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_val_loss = None
    best_epoch = None
    epochs_without_improvement = 0
    show_progress = not args.no_progress

    for epoch in range(args.epochs):
        losses = []
        train_transform_accuracies = []
        batch_iter = loader
        if show_progress:
            batch_iter = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)

        for batch in batch_iter:
            loss, transform_accuracy = train_step(
                model,
                batch,
                optimizer,
                sequence_criterion,
                transform_criterion,
                coarse_transform_criterion,
                reaction_center_criterion,
                enzyme_criterion,
                device,
                transform_loss_weight=args.transform_loss_weight,
                coarse_transform_loss_weight=args.coarse_transform_loss_weight,
                reaction_center_loss_weight=args.reaction_center_loss_weight,
                enzyme_loss_weight=args.enzyme_loss_weight,
                grad_clip=args.grad_clip,
                scaler=scaler,
                amp_enabled=amp_enabled,
                use_enzyme_head=args.use_enzyme_head,
            )
            losses.append(loss)
            train_transform_accuracies.append(transform_accuracy)
            if show_progress:
                batch_iter.set_postfix(loss=f"{loss:.4f}", transform_acc=f"{transform_accuracy:.3f}")

        train_loss = sum(losses) / len(losses)
        train_transform_accuracy = sum(train_transform_accuracies) / len(train_transform_accuracies)

        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                sequence_criterion,
                transform_criterion,
                coarse_transform_criterion,
                reaction_center_criterion,
                enzyme_criterion,
                device,
                transform_loss_weight=args.transform_loss_weight,
                coarse_transform_loss_weight=args.coarse_transform_loss_weight,
                reaction_center_loss_weight=args.reaction_center_loss_weight,
                enzyme_loss_weight=args.enzyme_loss_weight,
                amp_enabled=amp_enabled,
                use_enzyme_head=args.use_enzyme_head,
                progress_desc="Validation",
                show_progress=show_progress,
            )
            log_message(
                f"Epoch {epoch}: train_loss={train_loss:.4f} "
                f"train_transform_acc={train_transform_accuracy:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_transform_acc={val_metrics['transform_accuracy']:.4f}",
                log_file=args.log_file,
            )
            if mlflow_enabled:
                log_mlflow_metrics(
                    {
                        "train_loss": train_loss,
                        "train_transform_acc": train_transform_accuracy,
                        "val_loss": val_metrics["loss"],
                        "val_transform_acc": val_metrics["transform_accuracy"],
                    },
                    step=epoch,
                )

            if best_val_loss is None or val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), args.best_model_out)
                log_message(f"Saved best checkpoint to: {args.best_model_out}", log_file=args.log_file)
                if mlflow_enabled:
                    log_mlflow_metrics({"best_val_loss": best_val_loss, "best_epoch": float(best_epoch)}, step=epoch)
            else:
                epochs_without_improvement += 1

            if scheduler is not None:
                if args.scheduler == "plateau":
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            log_message(f"Epoch {epoch}: lr={current_lr:.6g}", log_file=args.log_file)
            if mlflow_enabled:
                log_mlflow_metrics({"lr": current_lr}, step=epoch)

            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                log_message(
                    f"Early stopping triggered after {epochs_without_improvement} non-improving validation epochs.",
                    log_file=args.log_file,
                )
                break
        else:
            log_message(
                f"Epoch {epoch}: train_loss={train_loss:.4f} "
                f"train_transform_acc={train_transform_accuracy:.4f}",
                log_file=args.log_file,
            )
            if mlflow_enabled:
                log_mlflow_metrics(
                    {
                        "train_loss": train_loss,
                        "train_transform_acc": train_transform_accuracy,
                    },
                    step=epoch,
                )
            if scheduler is not None:
                if args.scheduler == "plateau":
                    scheduler.step(train_loss)
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                log_message(f"Epoch {epoch}: lr={current_lr:.6g}", log_file=args.log_file)
                if mlflow_enabled:
                    log_mlflow_metrics({"lr": current_lr}, step=epoch)

    torch.save(model.state_dict(), args.model_out)

    test_metrics = None
    if test_loader is not None:
        if val_loader is not None and os.path.exists(args.best_model_out):
            best_state = torch.load(args.best_model_out, map_location=device)
            model.load_state_dict(best_state)
        test_metrics = evaluate(
            model,
            test_loader,
            sequence_criterion,
            transform_criterion,
            coarse_transform_criterion,
            reaction_center_criterion,
            enzyme_criterion,
            device,
            transform_loss_weight=args.transform_loss_weight,
            coarse_transform_loss_weight=args.coarse_transform_loss_weight,
            reaction_center_loss_weight=args.reaction_center_loss_weight,
            enzyme_loss_weight=args.enzyme_loss_weight,
            amp_enabled=amp_enabled,
            use_enzyme_head=args.use_enzyme_head,
            progress_desc="Test",
            show_progress=show_progress,
        )
        log_message(
            f"Test: loss={test_metrics['loss']:.4f} "
            f"transform_acc={test_metrics['transform_accuracy']:.4f}",
            log_file=args.log_file,
        )
        if mlflow_enabled:
            log_mlflow_metrics(
                {
                    "test_loss": test_metrics["loss"],
                    "test_transform_acc": test_metrics["transform_accuracy"],
                }
            )

    with open(args.metadata_out, "w") as f:
        json.dump(
            {
                "representation": args.representation,
                "tokenizer": tokenizer.to_config(),
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "encoder_hidden_dim": args.encoder_hidden_dim,
                "encoder_out_dim": args.encoder_out_dim,
                "encoder_heads": args.encoder_heads,
                "decoder_heads": args.decoder_heads,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "num_transform_classes": len(transform_map),
                "num_coarse_transform_classes": len(coarse_transform_map),
                "num_enzyme_classes": len(enzyme_map) if args.use_enzyme_head else 0,
                "atom_feature_dim": ATOM_FEATURE_DIM,
                "bond_feature_dim": BOND_FEATURE_DIM,
                "use_enzyme_head": args.use_enzyme_head,
                "transform_loss_weight": args.transform_loss_weight,
                "coarse_transform_loss_weight": args.coarse_transform_loss_weight,
                "reaction_center_loss_weight": args.reaction_center_loss_weight,
                "label_smoothing": args.label_smoothing,
                "balance_transform_classes": args.balance_transform_classes,
                "oversample_strategy": args.oversample_strategy,
                "oversample_power": args.oversample_power,
                "scheduler": args.scheduler,
                "scheduler_patience": args.scheduler_patience,
                "scheduler_factor": args.scheduler_factor,
                "min_lr": args.min_lr,
                "early_stopping_patience": args.early_stopping_patience,
                "amp": args.amp,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "pin_memory": pin_memory,
                "best_model_out": args.best_model_out if val_loader is not None else None,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "test_metrics": test_metrics,
                "coarse_transform_map": coarse_transform_map,
                "transform_map": transform_map,
                "enzyme_map": enzyme_map,
            },
            f,
            indent=2,
        )
    log_message(f"Saved model to: {args.model_out}", log_file=args.log_file)
    log_message(f"Saved metadata to: {args.metadata_out}", log_file=args.log_file)
    if mlflow_enabled:
        log_mlflow_artifacts(
            [
                args.model_out,
                args.best_model_out,
                args.metadata_out,
                args.log_file,
            ]
        )
        mlflow.end_run()


if __name__ == "__main__":
    main()