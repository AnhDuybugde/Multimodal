from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from .config import DataConfig, LABEL_TO_ID
    from .audio import AudioPipeline
except ImportError:
    from config import DataConfig, LABEL_TO_ID
    from audio import AudioPipeline


SPLITS = ("audio_visual_dataset_default", "audio_visual_dataset_robo_default")


def build_index(data_root: str | Path, skip_missing_files: bool = True) -> pd.DataFrame:
    data_root = Path(data_root)
    rows = []
    skipped = 0

    for split_name in SPLITS:
        split_dir = data_root / split_name
        csv_path = split_dir / "dataset.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        for item in df.to_dict("records"):
            audio_path = split_dir / item["audio_file"]
            image_path = split_dir / item["image_file"]
            exists = audio_path.exists() and image_path.exists()
            if skip_missing_files and not exists:
                skipped += 1
                continue
            rows.append(
                {
                    "split_name": split_name,
                    "audio_path": str(audio_path),
                    "image_path": str(image_path),
                    "label": item["category"],
                    "label_id": LABEL_TO_ID[item["category"]],
                    "files_exist": exists,
                }
            )

    index = pd.DataFrame(rows)
    index.attrs["skipped_missing_files"] = skipped
    return index


def make_train_val_split(df: pd.DataFrame, val_size: float = 0.2, seed: int = 42):
    try:
        from sklearn.model_selection import train_test_split

        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=seed,
            stratify=df["label_id"],
        )
    except Exception:
        train_df = df.sample(frac=1.0 - val_size, random_state=seed)
        val_df = df.drop(train_df.index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class AudioVisualDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        config: DataConfig,
        crop_mode: Optional[str] = None,
        train: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.train = train
        self.crop_mode = crop_mode or (config.train_crop if train else config.eval_crop)
        self.audio = AudioPipeline(
            target_sample_rate=config.target_sample_rate,
            window_sec=config.audio_window_sec,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            normalize_db=config.normalize_audio_db,
        )
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.RandomHorizontalFlip(p=0.5 if train else 0.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        mel = self.audio(row.audio_path, self.crop_mode)
        image = Image.open(row.image_path).convert("RGB")
        image = self.image_transform(image)
        label = torch.tensor(row.label_id, dtype=torch.long)
        return {"audio": mel, "image": image, "label": label}
