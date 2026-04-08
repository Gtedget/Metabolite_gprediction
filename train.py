"""Train a GAT+Transformer model for metabolite SMILES generation and transformation prediction."""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from data_utils import MetaboliteDataset, SmilesTokenizer
from model import MetaboliteGenerator
import pandas as pd
import argparse
import json
import os
from cid_lookup import cid_to_smiles

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
            "representation": args.representation,
            "max_len": args.max_len,
            "device": args.device,
            "use_enzyme_head": args.use_enzyme_head,
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


def compute_batch_loss(model, batch, criterion_ce, device, use_enzyme_head=False):
    graph, tgt_tokens, y_transform, y_enzyme = batch

    graph = graph.to(device)
    tgt_tokens = tgt_tokens.to(device)
    y_transform = y_transform.to(device)
    if use_enzyme_head:
        y_enzyme = y_enzyme.to(device)

    logits, pred_transform, pred_enzyme = model(graph, tgt_tokens[:, :-1])

    # SMILES generation loss
    loss_smiles = criterion_ce(
        logits.reshape(-1, logits.size(-1)),
        tgt_tokens[:, 1:].reshape(-1)
    )

    # Multi-task loss
    loss_transform = criterion_ce(pred_transform, y_transform)

    loss = loss_smiles + 0.3 * loss_transform
    if use_enzyme_head and pred_enzyme is not None:
        loss_enzyme = criterion_ce(pred_enzyme, y_enzyme)
        loss = loss + 0.3 * loss_enzyme

    transform_accuracy = (pred_transform.argmax(dim=1) == y_transform).float().mean().item()

    return loss, transform_accuracy


# -----------------------------------------------------
# 1. Training step
# -----------------------------------------------------
def train_step(model, batch, optimizer, criterion_ce, device, use_enzyme_head=False):
    optimizer.zero_grad()

    loss, transform_accuracy = compute_batch_loss(
        model,
        batch,
        criterion_ce,
        device,
        use_enzyme_head=use_enzyme_head,
    )

    loss.backward()
    optimizer.step()

    return loss.item(), transform_accuracy


@torch.no_grad()
def evaluate(model, loader, criterion_ce, device, use_enzyme_head=False):
    model.eval()
    losses = []
    transform_accuracies = []

    for batch in loader:
        loss, transform_accuracy = compute_batch_loss(
            model,
            batch,
            criterion_ce,
            device,
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
    parser.add_argument("--representation", choices=["smiles", "selfies"], default="selfies")
    parser.add_argument("--max_len", type=int, default=128)
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
    parser.add_argument(
        "--use_enzyme_head",
        action="store_true",
        help="Enable the enzyme classification head. Disabled by default because enzyme labels are often multi-valued.",
    )
    args = parser.parse_args()

    if args.log_file:
        with open(args.log_file, "w", encoding="utf-8"):
            pass

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

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

    dataset = MetaboliteDataset(
        df=df,
        smiles_lookup_fn=cid_to_smiles,
        tokenizer=tokenizer,
        transform_map=transform_map,
        enzyme_map=enzyme_map
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = None
    if val_df is not None:
        val_dataset = MetaboliteDataset(
            df=val_df,
            smiles_lookup_fn=cid_to_smiles,
            tokenizer=tokenizer,
            transform_map=transform_map,
            enzyme_map=enzyme_map,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_loader = None
    if test_df is not None:
        test_dataset = MetaboliteDataset(
            df=test_df,
            smiles_lookup_fn=cid_to_smiles,
            tokenizer=tokenizer,
            transform_map=transform_map,
            enzyme_map=enzyme_map,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MetaboliteGenerator(
        vocab_size=len(tokenizer.vocab),
        num_transform_classes=len(transform_map),
        num_enzyme_classes=len(enzyme_map),
        use_enzyme_head=args.use_enzyme_head,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_ce = nn.CrossEntropyLoss()
    best_val_loss = None
    best_epoch = None

    for epoch in range(args.epochs):
        losses = []
        train_transform_accuracies = []
        for batch in loader:
            loss, transform_accuracy = train_step(
                model,
                batch,
                optimizer,
                criterion_ce,
                device,
                use_enzyme_head=args.use_enzyme_head,
            )
            losses.append(loss)
            train_transform_accuracies.append(transform_accuracy)

        train_loss = sum(losses) / len(losses)
        train_transform_accuracy = sum(train_transform_accuracies) / len(train_transform_accuracies)

        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                criterion_ce,
                device,
                use_enzyme_head=args.use_enzyme_head,
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
                torch.save(model.state_dict(), args.best_model_out)
                log_message(f"Saved best checkpoint to: {args.best_model_out}", log_file=args.log_file)
                if mlflow_enabled:
                    log_mlflow_metrics({"best_val_loss": best_val_loss, "best_epoch": float(best_epoch)}, step=epoch)
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

    torch.save(model.state_dict(), args.model_out)

    test_metrics = None
    if test_loader is not None:
        if val_loader is not None and os.path.exists(args.best_model_out):
            best_state = torch.load(args.best_model_out, map_location=device)
            model.load_state_dict(best_state)
        test_metrics = evaluate(
            model,
            test_loader,
            criterion_ce,
            device,
            use_enzyme_head=args.use_enzyme_head,
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
                "num_transform_classes": len(transform_map),
                "num_enzyme_classes": len(enzyme_map) if args.use_enzyme_head else 0,
                "use_enzyme_head": args.use_enzyme_head,
                "best_model_out": args.best_model_out if val_loader is not None else None,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "test_metrics": test_metrics,
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