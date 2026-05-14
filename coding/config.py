from dataclasses import dataclass
from pathlib import Path


LABELS = ("ambient", "leaf", "trunk", "twig")
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


@dataclass
class DataConfig:
    data_root: Path
    target_sample_rate: int = 22000
    audio_window_sec: float = 0.8
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    image_size: int = 224
    skip_missing_files: bool = True
    train_crop: str = "random"
    eval_crop: str = "center"
    normalize_audio_db: bool = True


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    val_size: float = 0.2
    seed: int = 42
    use_amp: bool = True


@dataclass
class ModelConfig:
    ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    clap_model_name: str = "laion/clap-htsat-unfused"
    fusion_dim: int = 256
    fusion_heads: int = 4
    fusion_layers: int = 2
    fusion_dropout: float = 0.1
    freeze_pretrained: bool = False
