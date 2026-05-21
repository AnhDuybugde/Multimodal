from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from .config import DataConfig, LABELS, ModelConfig, TrainConfig
    from .data import AudioVisualDataset, build_index, make_train_val_split
    from .metrics import binary_contact_metrics, collect_predictions, paper_classification_metrics
    from .paper_model import ASTCLAPAudioNet, PaperLikeFusionNet, ViTImageNet
except ImportError:
    from config import DataConfig, LABELS, ModelConfig, TrainConfig
    from data import AudioVisualDataset, build_index, make_train_val_split
    from metrics import binary_contact_metrics, collect_predictions, paper_classification_metrics
    from paper_model import ASTCLAPAudioNet, PaperLikeFusionNet, ViTImageNet


TASK_LABELS = {
    "multiclass": LABELS,
    "binary": ("non_contact", "contact"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cpu",
    use_amp=True,
    task="multiclass",
    paper_average="weighted",
):
    is_train = optimizer is not None
    model.train(is_train)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device == "cuda")
    total_loss = 0.0
    total_items = 0
    logits_list = []
    labels_list = []

    for batch in tqdm(loader, leave=False):
        waveform = batch["waveform"].to(device)
        audio = batch["audio"].to(device)
        image = batch["image"].to(device)
        label_key = "binary_label" if task == "binary" else "label"
        labels = batch[label_key].to(device)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp and device == "cuda"):
                logits = model(waveform=waveform, image=image, audio=audio)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())

    y_true, y_pred = collect_predictions(logits_list, labels_list)
    if task == "binary":
        metrics = binary_contact_metrics(y_true, y_pred, paper_average="binary")
        metrics["macro_f1"] = paper_classification_metrics(
            y_true,
            y_pred,
            labels=TASK_LABELS["binary"],
            paper_average="macro",
        )["macro_f1"]
        metrics["weighted_f1"] = paper_classification_metrics(
            y_true,
            y_pred,
            labels=TASK_LABELS["binary"],
            paper_average="weighted",
        )["weighted_f1"]
    else:
        metrics = paper_classification_metrics(
            y_true,
            y_pred,
            labels=LABELS,
            paper_average=paper_average,
        )
        binary = binary_contact_metrics(y_true, y_pred, paper_average="binary")
        metrics["binary_contact_f1"] = binary["binary_contact_f1"]
        metrics["binary_contact"] = binary
    metrics["loss"] = total_loss / total_items
    return metrics


def make_class_weight_tensor(train_df, device: str, task: str):
    if task == "binary":
        ids = (train_df["label"] != "ambient").astype(int)
        counts = ids.value_counts().reindex(range(2)).fillna(0).to_numpy(dtype=np.float32)
        labels = TASK_LABELS["binary"]
    else:
        counts = train_df["label_id"].value_counts().reindex(range(len(LABELS))).fillna(0).to_numpy(dtype=np.float32)
        labels = LABELS
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    print(f"Class weights: {dict(zip(labels, weights.round(4).tolist()))}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-window-sec", type=float, default=0.8)
    parser.add_argument("--train-crop", choices=("random", "center", "energy"), default="random")
    parser.add_argument("--eval-crop", choices=("center", "energy"), default="center")
    parser.add_argument("--ast-model-name", default=ModelConfig.ast_model_name)
    parser.add_argument("--clap-model-name", default=ModelConfig.clap_model_name)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--freeze-pretrained", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--task", choices=("multiclass", "binary"), default="multiclass")
    parser.add_argument("--mode", choices=("audio", "video", "fusion"), default="fusion")
    parser.add_argument("--split-strategy", choices=("domain", "random"), default="domain")
    parser.add_argument("--paper-average", choices=("weighted", "macro", "micro"), default="weighted")
    parser.add_argument("--ast-input-source", choices=("mel", "waveform"), default="mel")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        class_weights=not args.no_class_weights,
    )
    data_cfg = DataConfig(
        data_root=args.data_root,
        target_sample_rate=args.target_sample_rate,
        audio_window_sec=args.audio_window_sec,
        train_crop=args.train_crop,
        eval_crop=args.eval_crop,
    )
    model_cfg = ModelConfig(
        ast_model_name=args.ast_model_name,
        clap_model_name=args.clap_model_name,
        fusion_dim=args.fusion_dim,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        freeze_pretrained=args.freeze_pretrained,
        ast_input_source=args.ast_input_source,
    )
    set_seed(train_cfg.seed)

    index = build_index(data_cfg.data_root, skip_missing_files=data_cfg.skip_missing_files)
    print(f"Indexed {len(index)} samples. Skipped missing files: {index.attrs.get('skipped_missing_files', 0)}")
    print(index["label"].value_counts().reindex(LABELS).fillna(0).astype(int))

    if args.split_strategy == "domain":
        train_df = index[index["split_name"] == "audio_visual_dataset_default"].reset_index(drop=True)
        val_df = index[index["split_name"] == "audio_visual_dataset_robo_default"].reset_index(drop=True)
    else:
        train_df, val_df = make_train_val_split(index, train_cfg.val_size, train_cfg.seed)
    print(f"Split strategy: {args.split_strategy}")
    print(f"Task: {args.task}")
    print(f"Mode: {args.mode}")
    train_ds = AudioVisualDataset(train_df, data_cfg, train=True)
    val_ds = AudioVisualDataset(val_df, data_cfg, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "audio":
        model = ASTCLAPAudioNet(
            num_classes=len(TASK_LABELS[args.task]),
            sample_rate=data_cfg.target_sample_rate,
            ast_model_name=model_cfg.ast_model_name,
            clap_model_name=model_cfg.clap_model_name,
            fusion_dim=model_cfg.fusion_dim,
            fusion_heads=model_cfg.fusion_heads,
            fusion_layers=model_cfg.fusion_layers,
            fusion_dropout=model_cfg.fusion_dropout,
            freeze_pretrained=model_cfg.freeze_pretrained,
            ast_input_source=model_cfg.ast_input_source,
        )
    elif args.mode == "video":
        model = ViTImageNet(
            num_classes=len(TASK_LABELS[args.task]),
            fusion_dim=model_cfg.fusion_dim,
            fusion_dropout=model_cfg.fusion_dropout,
            freeze_pretrained=model_cfg.freeze_pretrained,
        )
    else:
        model = PaperLikeFusionNet(
            num_classes=len(TASK_LABELS[args.task]),
            sample_rate=data_cfg.target_sample_rate,
            ast_model_name=model_cfg.ast_model_name,
            clap_model_name=model_cfg.clap_model_name,
            fusion_dim=model_cfg.fusion_dim,
            fusion_heads=model_cfg.fusion_heads,
            fusion_layers=model_cfg.fusion_layers,
            fusion_dropout=model_cfg.fusion_dropout,
            freeze_pretrained=model_cfg.freeze_pretrained,
            ast_input_source=model_cfg.ast_input_source,
        )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=make_class_weight_tensor(train_df, device, args.task) if train_cfg.class_weights else None
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best_paper_f1 = -1.0
    best_metrics = None
    history = []

    for epoch in range(1, train_cfg.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train_cfg.use_amp,
            args.task,
            args.paper_average,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            train_cfg.use_amp,
            args.task,
            args.paper_average,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"train_paper_f1={train_metrics['paper_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f} "
            f"val_paper_f1={val_metrics['paper_f1']:.4f} "
            f"val_precision={val_metrics['paper_precision']:.4f} val_recall={val_metrics['paper_recall']:.4f}"
        )
        if args.task == "multiclass":
            print(f"val_binary_f1_from_multiclass={val_metrics['binary_contact_f1']:.4f}")
        if val_metrics["paper_f1"] > best_paper_f1:
            best_paper_f1 = val_metrics["paper_f1"]
            best_metrics = val_metrics
            torch.save(
                {
                    "model": model.state_dict(),
                    "labels": TASK_LABELS[args.task],
                    "mode": args.mode,
                    "task": args.task,
                    "split_strategy": args.split_strategy,
                    "paper_average": args.paper_average,
                    "data_config": data_cfg.__dict__,
                    "model_config": model_cfg.__dict__,
                    "val_metrics": val_metrics,
                },
                out_dir / f"best_{args.task}_{args.mode}_paper_like_model.pt",
            )

    print(f"Best validation paper-style F1: {best_paper_f1:.4f}")
    result = {
        "mode": args.mode,
        "task": args.task,
        "split_strategy": args.split_strategy,
        "paper_average": args.paper_average if args.task == "multiclass" else "binary",
        "best_paper_f1": best_paper_f1,
        "best_val_metrics": best_metrics,
        "weight_file": str(out_dir / f"best_{args.task}_{args.mode}_paper_like_model.pt"),
        "history": history,
    }
    result_path = out_dir / f"{args.task}_{args.mode}_paper_results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved result: {result_path}")
    if best_metrics is not None:
        print(
            "BEST "
            f"mode={args.mode} task={args.task} "
            f"F1={best_metrics.get('paper_f1', float('nan')):.4f} "
            f"Precision={best_metrics.get('paper_precision', float('nan')):.4f} "
            f"Recall={best_metrics.get('paper_recall', float('nan')):.4f} "
            f"macro_f1={best_metrics.get('macro_f1', float('nan')):.4f} "
            f"weighted_f1={best_metrics.get('weighted_f1', float('nan')):.4f} "
            f"acc={best_metrics.get('accuracy', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
