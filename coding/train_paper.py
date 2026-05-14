from __future__ import annotations

import argparse
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
    from .metrics import binary_contact_f1, collect_predictions, f1_scores
    from .paper_model import PaperLikeFusionNet
except ImportError:
    from config import DataConfig, LABELS, ModelConfig, TrainConfig
    from data import AudioVisualDataset, build_index, make_train_val_split
    from metrics import binary_contact_f1, collect_predictions, f1_scores
    from paper_model import PaperLikeFusionNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu", use_amp=True):
    is_train = optimizer is not None
    model.train(is_train)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device == "cuda")
    total_loss = 0.0
    total_items = 0
    logits_list = []
    labels_list = []

    for batch in tqdm(loader, leave=False):
        waveform = batch["waveform"].to(device)
        image = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp and device == "cuda"):
                logits = model(waveform=waveform, image=image)
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
    metrics = f1_scores(y_true, y_pred, num_classes=len(LABELS))
    metrics["binary_contact_f1"] = binary_contact_f1(y_true, y_pred)
    metrics["loss"] = total_loss / total_items
    metrics["accuracy"] = float((y_true == y_pred).mean())
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--target-sample-rate", type=int, default=22000)
    parser.add_argument("--audio-window-sec", type=float, default=0.8)
    parser.add_argument("--train-crop", choices=("random", "center", "energy"), default="random")
    parser.add_argument("--eval-crop", choices=("center", "energy"), default="center")
    parser.add_argument("--ast-model-name", default=ModelConfig.ast_model_name)
    parser.add_argument("--clap-model-name", default=ModelConfig.clap_model_name)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--freeze-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
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
    )
    set_seed(train_cfg.seed)

    index = build_index(data_cfg.data_root, skip_missing_files=data_cfg.skip_missing_files)
    print(f"Indexed {len(index)} samples. Skipped missing files: {index.attrs.get('skipped_missing_files', 0)}")
    print(index["label"].value_counts().reindex(LABELS).fillna(0).astype(int))

    train_df, val_df = make_train_val_split(index, train_cfg.val_size, train_cfg.seed)
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
    model = PaperLikeFusionNet(
        num_classes=len(LABELS),
        sample_rate=data_cfg.target_sample_rate,
        ast_model_name=model_cfg.ast_model_name,
        clap_model_name=model_cfg.clap_model_name,
        fusion_dim=model_cfg.fusion_dim,
        fusion_heads=model_cfg.fusion_heads,
        fusion_layers=model_cfg.fusion_layers,
        fusion_dropout=model_cfg.fusion_dropout,
        freeze_pretrained=model_cfg.freeze_pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    best_macro_f1 = 0.0

    for epoch in range(1, train_cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train_cfg.use_amp)
        val_metrics = run_epoch(model, val_loader, criterion, None, device, train_cfg.use_amp)
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"train_binary_f1={train_metrics['binary_contact_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f} "
            f"val_binary_f1={val_metrics['binary_contact_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "labels": LABELS,
                    "data_config": data_cfg.__dict__,
                    "model_config": model_cfg.__dict__,
                    "val_metrics": val_metrics,
                },
                out_dir / "best_paper_like_model.pt",
            )

    print(f"Best validation macro F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
