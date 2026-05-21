from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

try:
    from .audio import AudioPipeline
    from .audio_model import ASTCLAPAudioClassifier
    from .config import DataConfig, LABELS, LABEL_TO_ID
    from .data import build_index, make_train_val_split
except ImportError:
    from audio import AudioPipeline
    from audio_model import ASTCLAPAudioClassifier
    from config import DataConfig, LABELS, LABEL_TO_ID
    from data import build_index, make_train_val_split


class AudioOnlyDataset(Dataset):
    def __init__(self, frame, config: DataConfig, train: bool = False, crop_mode: str | None = None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.crop_mode = crop_mode or (config.train_crop if train else config.eval_crop)
        self.audio = AudioPipeline(config.target_sample_rate, config.audio_window_sec)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        waveform = self.audio.load_processed_waveform(row.audio_path, self.crop_mode).squeeze(0)
        return {
            "waveform": waveform,
            "label": torch.tensor(row.label_id, dtype=torch.long),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Audio-only AST+CLAP training with debug diagnostics.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audio_debug"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-window-sec", type=float, default=0.8)
    parser.add_argument("--train-crop", choices=("random", "center", "energy"), default="energy")
    parser.add_argument("--eval-crop", choices=("center", "energy"), default="energy")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--ast-model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--clap-model-name", default="laion/clap-htsat-unfused")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig(
        data_root=args.data_root,
        target_sample_rate=args.target_sample_rate,
        audio_window_sec=args.audio_window_sec,
        train_crop=args.train_crop,
        eval_crop=args.eval_crop,
    )
    index = build_index(data_cfg.data_root, data_cfg.skip_missing_files)
    train_df, val_df = make_train_val_split(index, args.val_size, args.seed)

    print_index_debug(index, train_df, val_df)
    train_ds = AudioOnlyDataset(train_df, data_cfg, train=True)
    val_ds = AudioOnlyDataset(val_df, data_cfg, train=False)
    sampler = make_sampler(train_df) if args.weighted_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    inspect_first_batches(train_loader, "train")
    inspect_first_batches(val_loader, "val")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ASTCLAPAudioClassifier(
        num_classes=len(LABELS),
        sample_rate=data_cfg.target_sample_rate,
        ast_model_name=args.ast_model_name,
        clap_model_name=args.clap_model_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)
    print_model_debug(model)

    criterion = make_criterion(train_df, args.class_weights, device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_macro_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, args.use_amp)
        val_metrics = run_epoch(model, val_loader, criterion, None, device, args.use_amp)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print_epoch(epoch, train_metrics, val_metrics)
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "labels": LABELS,
                    "args": vars(args),
                    "best_val_metrics": val_metrics,
                },
                args.output_dir / "best_audio_debug_model.pt",
            )
        with (args.output_dir / "audio_debug_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_index_debug(index, train_df, val_df) -> None:
    print(f"valid_samples={len(index)} skipped_missing={index.attrs.get('skipped_missing_files', 0)}")
    print("all labels:")
    print(index["label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_string())
    print("train labels:")
    print(train_df["label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_string())
    print("val labels:")
    print(val_df["label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_string())


def make_sampler(train_df):
    counts = train_df["label_id"].value_counts().to_dict()
    weights = train_df["label_id"].map(lambda label_id: 1.0 / counts[label_id]).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def inspect_first_batches(loader, name: str, max_batches: int = 2) -> None:
    print(f"\n{name} batch sanity:")
    for batch_idx, batch in enumerate(loader):
        waveform = batch["waveform"]
        labels = batch["label"]
        rms = waveform.pow(2).mean(dim=1).sqrt()
        peak = waveform.abs().amax(dim=1)
        counts = torch.bincount(labels, minlength=len(LABELS)).tolist()
        print(
            f"batch={batch_idx} shape={tuple(waveform.shape)} "
            f"rms[min/mean/max]={rms.min().item():.6g}/{rms.mean().item():.6g}/{rms.max().item():.6g} "
            f"peak[min/mean/max]={peak.min().item():.6g}/{peak.mean().item():.6g}/{peak.max().item():.6g} "
            f"label_counts={dict(zip(LABELS, counts))}"
        )
        if batch_idx + 1 >= max_batches:
            break


def print_model_debug(model) -> None:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"\nmodel params total={total:,} trainable={trainable:,}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"trainable: {name} shape={tuple(param.shape)}")


def make_criterion(train_df, use_class_weights: bool, device: str):
    if not use_class_weights:
        return nn.CrossEntropyLoss()
    counts = train_df["label_id"].value_counts().reindex(range(len(LABELS))).fillna(0).to_numpy(dtype=np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    print(f"class_weights={dict(zip(LABELS, weights.round(4).tolist()))}")
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def run_epoch(model, loader, criterion, optimizer, device: str, use_amp: bool):
    is_train = optimizer is not None
    model.train(is_train)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device == "cuda")
    total_loss = 0.0
    total_items = 0
    logits_list = []
    labels_list = []
    grad_norms = []

    for batch in tqdm(loader, leave=False):
        waveform = batch["waveform"].to(device)
        labels = batch["label"].to(device)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp and device == "cuda"):
                logits = model(waveform)
                loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norms.append(grad_norm(model))
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())

    metrics = compute_metrics(logits_list, labels_list)
    metrics["loss"] = total_loss / max(total_items, 1)
    if grad_norms:
        metrics["grad_norm_mean"] = float(np.mean(grad_norms))
        metrics["grad_norm_max"] = float(np.max(grad_norms))
    return metrics


def grad_norm(model) -> float:
    total = 0.0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            total += float(param.grad.detach().pow(2).sum().cpu())
    return total ** 0.5


def compute_metrics(logits_list, labels_list):
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    preds = logits.argmax(dim=1)
    y_true = labels.numpy()
    y_pred = preds.numpy()
    out = {}
    try:
        from sklearn.metrics import classification_report, confusion_matrix, f1_score

        out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        out["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist()
        out["classification_report"] = classification_report(
            y_true,
            y_pred,
            target_names=LABELS,
            zero_division=0,
            output_dict=True,
        )
    except Exception:
        out["macro_f1"] = float("nan")
        out["weighted_f1"] = float("nan")
    out["accuracy"] = float((y_true == y_pred).mean())
    out["true_counts"] = count_ids(y_true)
    out["pred_counts"] = count_ids(y_pred)
    out["logit_mean"] = logits.float().mean(dim=0).tolist()
    out["logit_std"] = logits.float().std(dim=0).tolist()
    return out


def count_ids(values):
    counts = np.bincount(values, minlength=len(LABELS)).astype(int)
    return {label: int(counts[idx]) for idx, label in enumerate(LABELS)}


def print_epoch(epoch: int, train_metrics, val_metrics) -> None:
    print(
        f"\nepoch={epoch} "
        f"train_loss={train_metrics['loss']:.4f} train_macro_f1={train_metrics['macro_f1']:.4f} "
        f"val_loss={val_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
        f"val_weighted_f1={val_metrics['weighted_f1']:.4f} val_acc={val_metrics['accuracy']:.4f}"
    )
    if "grad_norm_mean" in train_metrics:
        print(
            f"grad_norm mean={train_metrics['grad_norm_mean']:.6f} "
            f"max={train_metrics['grad_norm_max']:.6f}"
        )
    print(f"train_pred_counts={train_metrics['pred_counts']}")
    print(f"val_pred_counts={val_metrics['pred_counts']}")
    print("val_confusion_matrix rows=true cols=pred labels=(ambient, leaf, trunk, twig)")
    for row in val_metrics.get("confusion_matrix", []):
        print(row)
    report = val_metrics.get("classification_report", {})
    for label in LABELS:
        if label in report:
            item = report[label]
            print(
                f"{label}: precision={item['precision']:.3f} "
                f"recall={item['recall']:.3f} f1={item['f1-score']:.3f} support={item['support']}"
            )


if __name__ == "__main__":
    main()
