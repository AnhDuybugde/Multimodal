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
    from .config import DataConfig, TrainConfig, LABELS
    from .data import AudioVisualDataset, build_index, make_train_val_split
    from .models import AudioVisualFusionNet
except ImportError:
    from config import DataConfig, TrainConfig, LABELS
    from data import AudioVisualDataset, build_index, make_train_val_split
    from models import AudioVisualFusionNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer=None, device="cpu", use_amp=True):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_acc = 0.0
    total_items = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device == "cuda")

    for batch in tqdm(loader, leave=False):
        audio = batch["audio"].to(device)
        image = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp and device == "cuda"):
                logits = model(audio, image)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits.detach(), labels) * batch_size
        total_items += batch_size

    return total_loss / total_items, total_acc / total_items


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--target-sample-rate", type=int, default=22000)
    parser.add_argument("--audio-window-sec", type=float, default=0.8)
    parser.add_argument("--train-crop", choices=("random", "center", "energy"), default="random")
    parser.add_argument("--eval-crop", choices=("center", "energy"), default="center")
    parser.add_argument("--pretrained-image", action="store_true")
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
    set_seed(train_cfg.seed)

    index = build_index(data_cfg.data_root, skip_missing_files=data_cfg.skip_missing_files)
    skipped = index.attrs.get("skipped_missing_files", 0)
    print(f"Indexed {len(index)} samples. Skipped missing files: {skipped}")
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
    model = AudioVisualFusionNet(num_classes=len(LABELS), pretrained_image=args.pretrained_image).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    best_val_acc = 0.0
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train_cfg.use_amp
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, train_cfg.use_amp)
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "labels": LABELS,
                    "data_config": data_cfg.__dict__,
                },
                out_dir / "best_fusion_model.pt",
            )

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
