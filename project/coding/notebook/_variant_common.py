from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coding.audio import AudioPipeline
from coding.config import LABELS, DataConfig, ModelConfig, TrainConfig
from coding.data import AudioVisualDataset, build_index, make_train_val_split
from coding.metrics import binary_contact_metrics, collect_predictions, paper_classification_metrics
from coding.paper_model import ASTCLAPAudioNet, PaperLikeFusionNet, ViTImageNet


PAPER_AVERAGE = "weighted"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "variants"
DEFAULT_DATA_CANDIDATES = (
    PROJECT_ROOT / "dataset",
    PROJECT_ROOT.parent / "dataset",
    Path("/kaggle/input/datasets/anhduy54/visual-audio/raw_dataset"),
    Path("/kaggle/input/datasets/anhduy54/visual-audio/prepared_data"),
    Path("/kaggle/input/datasets/anhduy54/visual-audio"),
    Path("/kaggle/input/datasets/anhduy54/visual-audio/dataset"),
    Path("/kaggle/input/visual-audio/raw_dataset"),
    Path("/kaggle/input/visual-audio/prepared_data"),
    Path("/kaggle/input/visual-audio"),
    Path("/kaggle/input/visual-audio/dataset"),
    Path("/kaggle/input/datasets/nguynnguynhehe/audio-video-dataset/dataset"),
    Path("/kaggle/input/audio-video-dataset/dataset"),
    Path("/kaggle/input/contact-data/dataset"),
    Path("/kaggle/input/contact_data/dataset"),
)


@dataclass
class ExperimentContext:
    data_cfg: DataConfig
    train_cfg: TrainConfig
    model_cfg: ModelConfig
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    output_dir: Path
    device: str
    split_strategy: str = "domain"


def find_data_root(data_root: str | Path | None = None) -> Path:
    if data_root is not None:
        data_root = Path(data_root)
        if (data_root / "audio_visual_dataset_default" / "dataset.csv").exists():
            return data_root
        raw = data_root / "raw_dataset"
        if (raw / "audio_visual_dataset_default" / "dataset.csv").exists():
            return raw
        prepared = data_root / "prepared_data"
        if (prepared / "audio_visual_dataset_default" / "dataset.csv").exists():
            return prepared
        nested = data_root / "dataset"
        if (nested / "audio_visual_dataset_default" / "dataset.csv").exists():
            return nested
        return data_root
    for candidate in DEFAULT_DATA_CANDIDATES:
        if (candidate / "audio_visual_dataset_default" / "dataset.csv").exists():
            return candidate
    return DEFAULT_DATA_CANDIDATES[0]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_context(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    num_workers: int = 2,
    split_strategy: str = "domain",
    freeze_pretrained: bool = True,
    class_weights: bool = True,
    train_crop: str = "random",
    eval_crop: str = "energy",
    ast_input_source: str = "mel",
) -> ExperimentContext:
    data_root = find_data_root(data_root)
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig(data_root=data_root, train_crop=train_crop, eval_crop=eval_crop)
    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        class_weights=class_weights,
    )
    model_cfg = ModelConfig(freeze_pretrained=freeze_pretrained, ast_input_source=ast_input_source)
    set_seed(train_cfg.seed)

    index = build_index(data_cfg.data_root, skip_missing_files=data_cfg.skip_missing_files)
    if split_strategy == "domain":
        train_df = index[index["split_name"] == "audio_visual_dataset_default"].reset_index(drop=True)
        val_df = index[index["split_name"] == "audio_visual_dataset_robo_default"].reset_index(drop=True)
    else:
        train_df, val_df = make_train_val_split(index, train_cfg.val_size, train_cfg.seed)

    return ExperimentContext(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        train_df=train_df,
        val_df=val_df,
        output_dir=output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        split_strategy=split_strategy,
    )


def make_class_weight_tensor(frame: pd.DataFrame, ctx: ExperimentContext) -> torch.Tensor:
    counts = frame["label_id"].value_counts().reindex(range(len(LABELS))).fillna(0).to_numpy(dtype=np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=ctx.device)


def make_av_loaders(ctx: ExperimentContext):
    train_ds = AudioVisualDataset(ctx.train_df, ctx.data_cfg, train=True)
    val_ds = AudioVisualDataset(ctx.val_df, ctx.data_cfg, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=ctx.train_cfg.batch_size,
        shuffle=True,
        num_workers=ctx.train_cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=ctx.train_cfg.batch_size,
        shuffle=False,
        num_workers=ctx.train_cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def compute_multiclass_metrics(logits_list, labels_list) -> dict:
    y_true, y_pred = collect_predictions(logits_list, labels_list)
    metrics = paper_classification_metrics(y_true, y_pred, labels=LABELS, paper_average=PAPER_AVERAGE)
    binary = binary_contact_metrics(y_true, y_pred, paper_average="binary")
    metrics["binary_contact_f1"] = binary["binary_contact_f1"]
    metrics["binary_contact"] = binary
    return metrics


def run_epoch(model, loader, criterion, ctx: ExperimentContext, optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    scaler = torch.amp.GradScaler("cuda", enabled=ctx.train_cfg.use_amp and ctx.device == "cuda")
    total_loss = 0.0
    total_items = 0
    logits_list = []
    labels_list = []

    for batch in tqdm(loader, leave=False):
        waveform = batch["waveform"].to(ctx.device)
        audio = batch["audio"].to(ctx.device)
        image = batch["image"].to(ctx.device)
        labels = batch["label"].to(ctx.device)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=ctx.train_cfg.use_amp and ctx.device == "cuda"):
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

    metrics = compute_multiclass_metrics(logits_list, labels_list)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


class EarlyFusionConcatNet(PaperLikeFusionNet):
    """Feature-level early fusion by concatenating projected modality embeddings."""

    def __init__(self, ctx: ExperimentContext):
        cfg = ctx.model_cfg
        super().__init__(
            num_classes=len(LABELS),
            sample_rate=ctx.data_cfg.target_sample_rate,
            ast_model_name=cfg.ast_model_name,
            clap_model_name=cfg.clap_model_name,
            fusion_dim=cfg.fusion_dim,
            fusion_heads=cfg.fusion_heads,
            fusion_layers=cfg.fusion_layers,
            fusion_dropout=cfg.fusion_dropout,
            freeze_pretrained=cfg.freeze_pretrained,
            ast_input_source=cfg.ast_input_source,
        )
        self.concat_classifier = nn.Sequential(
            nn.LayerNorm(cfg.fusion_dim * 3),
            nn.Linear(cfg.fusion_dim * 3, cfg.fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(cfg.fusion_dim * 2, cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(cfg.fusion_dim, len(LABELS)),
        )

    def forward(self, waveform=None, image=None, audio=None):
        ast_emb = self.encode_ast_mel(audio) if self.ast_input_source == "mel" and audio is not None else self.encode_ast(waveform)
        clap_emb = self.encode_clap(waveform)
        image_emb = self.image_model(image)
        projected = [self.ast_proj(ast_emb), self.clap_proj(clap_emb), self.image_proj(image_emb)]
        return self.concat_classifier(torch.cat(projected, dim=-1))


class LateFusionLogitNet(nn.Module):
    """Logit-level late fusion with a learned weighted average of branch logits."""

    def __init__(self, ctx: ExperimentContext):
        super().__init__()
        cfg = ctx.model_cfg
        kwargs = dict(
            num_classes=len(LABELS),
            sample_rate=ctx.data_cfg.target_sample_rate,
            ast_model_name=cfg.ast_model_name,
            clap_model_name=cfg.clap_model_name,
            fusion_dim=cfg.fusion_dim,
            fusion_heads=cfg.fusion_heads,
            fusion_layers=cfg.fusion_layers,
            fusion_dropout=cfg.fusion_dropout,
            freeze_pretrained=cfg.freeze_pretrained,
            ast_input_source=cfg.ast_input_source,
        )
        self.audio_model = ASTCLAPAudioNet(**kwargs)
        self.image_model = ViTImageNet(
            num_classes=len(LABELS),
            fusion_dim=cfg.fusion_dim,
            fusion_dropout=cfg.fusion_dropout,
            freeze_pretrained=cfg.freeze_pretrained,
        )
        self.logit_weights = nn.Parameter(torch.zeros(2))

    def forward(self, waveform=None, image=None, audio=None):
        audio_logits = self.audio_model(waveform=waveform, image=image, audio=audio)
        image_logits = self.image_model(waveform=waveform, image=image, audio=audio)
        weights = torch.softmax(self.logit_weights, dim=0)
        return weights[0] * audio_logits + weights[1] * image_logits


def build_fusion_model(kind: str, ctx: ExperimentContext):
    cfg = ctx.model_cfg
    if kind == "early":
        return EarlyFusionConcatNet(ctx)
    if kind == "middle":
        return PaperLikeFusionNet(
            num_classes=len(LABELS),
            sample_rate=ctx.data_cfg.target_sample_rate,
            ast_model_name=cfg.ast_model_name,
            clap_model_name=cfg.clap_model_name,
            fusion_dim=cfg.fusion_dim,
            fusion_heads=cfg.fusion_heads,
            fusion_layers=cfg.fusion_layers,
            fusion_dropout=cfg.fusion_dropout,
            freeze_pretrained=cfg.freeze_pretrained,
            ast_input_source=cfg.ast_input_source,
        )
    if kind == "late":
        return LateFusionLogitNet(ctx)
    raise ValueError(f"Unknown fusion kind: {kind}")


def train_fusion_variant(
    kind: str,
    ctx: ExperimentContext,
    early_stopping_patience: int = 10,
    min_delta: float = 1e-4,
) -> dict:
    train_loader, val_loader = make_av_loaders(ctx)
    model = build_fusion_model(kind, ctx).to(ctx.device)
    criterion = nn.CrossEntropyLoss(
        weight=make_class_weight_tensor(ctx.train_df, ctx) if ctx.train_cfg.class_weights else None
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=ctx.train_cfg.lr,
        weight_decay=ctx.train_cfg.weight_decay,
    )
    best_f1 = -1.0
    best_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, ctx.train_cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, ctx, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, ctx)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"fusion_{kind} epoch={epoch} "
            f"train_f1={train_metrics['paper_f1']:.4f} "
            f"val_f1={val_metrics['paper_f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["paper_f1"] > best_f1 + min_delta:
            best_f1 = val_metrics["paper_f1"]
            best_metrics = val_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "mode": f"fusion_{kind}",
                    "model": model.state_dict(),
                    "labels": LABELS,
                    "paper_average": PAPER_AVERAGE,
                    "data_config": ctx.data_cfg.__dict__,
                    "model_config": ctx.model_cfg.__dict__,
                    "best_val_metrics": val_metrics,
                },
                ctx.output_dir / f"best_fusion_{kind}_model.pt",
            )
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(
                    f"fusion_{kind} early stopping at epoch={epoch}; "
                    f"best_epoch={best_epoch} best_val_f1={best_f1:.4f}"
                )
                break

    result = {
        "mode": f"fusion_{kind}",
        "paper_average": PAPER_AVERAGE,
        "best_paper_f1": best_f1,
        "best_epoch": best_epoch,
        "early_stopping_patience": early_stopping_patience,
        "min_delta": min_delta,
        "best_val_metrics": best_metrics,
        "weight_file": str(ctx.output_dir / f"best_fusion_{kind}_model.pt"),
        "history": history,
    }
    result_path = ctx.output_dir / f"fusion_{kind}_results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


FEATURE_CACHE: dict[tuple, np.ndarray] = {}


def sample_stratified_frame(frame: pd.DataFrame, max_samples: int, seed: int = 42) -> pd.DataFrame:
    if not max_samples or len(frame) <= max_samples:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    counts = frame["label"].value_counts().reindex(LABELS).fillna(0).astype(int)
    total = int(counts.sum())
    parts = []
    for label in LABELS:
        rows = frame[frame["label"] == label]
        if rows.empty:
            continue
        quota = max(1, int(round(max_samples * len(rows) / total)))
        take = min(quota, len(rows))
        choices = rng.choice(len(rows), size=take, replace=False)
        parts.append(rows.iloc[choices])
    return pd.concat(parts, ignore_index=True) if parts else frame.head(0)


def resize_matrix(mat: torch.Tensor, height: int = 32, width: int = 32) -> torch.Tensor:
    tensor = mat.float().unsqueeze(0).unsqueeze(0)
    out = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return out.squeeze(0).squeeze(0)


def dct_matrix(n_mels: int, n_mfcc: int, device) -> torch.Tensor:
    n = torch.arange(n_mels, device=device).float()
    k = torch.arange(n_mfcc, device=device).float().unsqueeze(1)
    basis = torch.cos(math.pi / n_mels * (n + 0.5) * k)
    basis[0] *= math.sqrt(1.0 / n_mels)
    if n_mfcc > 1:
        basis[1:] *= math.sqrt(2.0 / n_mels)
    return basis


def mfcc_map_from_mel(mel: torch.Tensor, n_mfcc: int = 40) -> torch.Tensor:
    mel_db = mel.squeeze(0)
    if mel_db.min().item() >= 0.0 and mel_db.max().item() <= 1.0:
        mel_db = mel_db * 80.0 - 80.0
    return dct_matrix(mel_db.shape[0], n_mfcc, mel_db.device) @ mel_db


def fft_vector_from_waveform(waveform: torch.Tensor, bins: int = 512) -> torch.Tensor:
    values = waveform.squeeze(0).float()
    mag = torch.log1p(torch.fft.rfft(values).abs()).view(1, 1, -1)
    return F.interpolate(mag, size=bins, mode="linear", align_corners=False).view(-1)


def make_audio_pipeline(ctx: ExperimentContext) -> AudioPipeline:
    return AudioPipeline(
        ctx.data_cfg.target_sample_rate,
        ctx.data_cfg.audio_window_sec,
        ctx.data_cfg.n_mels,
        ctx.data_cfg.n_fft,
        ctx.data_cfg.hop_length,
        ctx.data_cfg.normalize_audio_db,
        ctx.data_cfg.spectral_gate,
        ctx.data_cfg.spectral_gate_noise_percentile,
        ctx.data_cfg.spectral_gate_strength,
    )


def audio_feature_vector(
    path: str,
    feature_name: str,
    crop_mode: str,
    ctx: ExperimentContext,
    pipe: AudioPipeline | None = None,
) -> torch.Tensor:
    pipe = pipe or make_audio_pipeline(ctx)
    waveform = pipe.load_processed_waveform(path, crop_mode)
    if feature_name == "fft":
        return fft_vector_from_waveform(waveform)

    if feature_name in ("mfcc", "mfcc_fft", "psla_logmel"):
        mel = pipe.waveform_to_mel(waveform)
        if feature_name == "psla_logmel":
            return resize_matrix(mel.squeeze(0), 64, 64).reshape(-1)
        mfcc = resize_matrix(mfcc_map_from_mel(mel), 40, 32).reshape(-1)
        if feature_name == "mfcc":
            return mfcc
        return torch.cat([mfcc, fft_vector_from_waveform(waveform)], dim=0)

    if feature_name == "stft":
        window = torch.hann_window(ctx.data_cfg.n_fft, device=waveform.device)
        spec = torch.stft(
            waveform.float(),
            n_fft=ctx.data_cfg.n_fft,
            hop_length=ctx.data_cfg.hop_length,
            win_length=ctx.data_cfg.n_fft,
            window=window,
            return_complex=True,
        ).abs().squeeze(0)
        return resize_matrix(torch.log1p(spec), 64, 32).reshape(-1)

    raise ValueError(f"Unknown audio feature: {feature_name}")


def audio_feature_map(
    path: str,
    feature_name: str,
    crop_mode: str,
    ctx: ExperimentContext,
    pipe: AudioPipeline | None = None,
) -> torch.Tensor:
    pipe = pipe or make_audio_pipeline(ctx)
    waveform = pipe.load_processed_waveform(path, crop_mode)
    if feature_name == "fft":
        return fft_vector_from_waveform(waveform).view(1, 32, 16)

    if feature_name in ("mfcc", "mfcc_fft", "psla_logmel"):
        mel = pipe.waveform_to_mel(waveform)
        if feature_name == "psla_logmel":
            return resize_matrix(mel.squeeze(0), 64, 64).unsqueeze(0)
        mfcc = resize_matrix(mfcc_map_from_mel(mel), 40, 32)
        if feature_name == "mfcc":
            return mfcc.unsqueeze(0)
        fft = fft_vector_from_waveform(waveform).view(16, 32)
        return torch.cat([mfcc, fft], dim=0).unsqueeze(0)

    if feature_name == "stft":
        window = torch.hann_window(ctx.data_cfg.n_fft, device=waveform.device)
        spec = torch.stft(
            waveform.float(),
            n_fft=ctx.data_cfg.n_fft,
            hop_length=ctx.data_cfg.hop_length,
            win_length=ctx.data_cfg.n_fft,
            window=window,
            return_complex=True,
        ).abs().squeeze(0)
        return resize_matrix(torch.log1p(spec), 64, 32).unsqueeze(0)

    raise ValueError(f"Unknown audio feature: {feature_name}")


def feature_map_shape(feature_name: str) -> tuple[int, int, int]:
    if feature_name == "fft":
        return (1, 32, 16)
    if feature_name == "mfcc":
        return (1, 40, 32)
    if feature_name == "mfcc_fft":
        return (1, 56, 32)
    if feature_name == "psla_logmel":
        return (1, 64, 64)
    if feature_name == "stft":
        return (1, 64, 32)
    raise ValueError(f"Unknown audio feature: {feature_name}")


def build_audio_feature_matrix(frame: pd.DataFrame, feature_name: str, crop_mode: str, ctx: ExperimentContext) -> np.ndarray:
    key = (feature_name, crop_mode, tuple(frame["audio_path"].tolist()))
    if key in FEATURE_CACHE:
        return FEATURE_CACHE[key]
    pipe = make_audio_pipeline(ctx)
    vectors = []
    for row in tqdm(frame.to_dict("records"), desc=f"{feature_name}/{crop_mode}", leave=False):
        vectors.append(audio_feature_vector(row["audio_path"], feature_name, crop_mode, ctx, pipe).detach().cpu().numpy())
    x = np.asarray(vectors, dtype=np.float32)
    FEATURE_CACHE[key] = x
    return x


def audio_frames(ctx: ExperimentContext, max_samples: int = 0):
    return (
        sample_stratified_frame(ctx.train_df, max_samples, ctx.train_cfg.seed),
        sample_stratified_frame(ctx.val_df, max_samples, ctx.train_cfg.seed),
    )


def run_audio_ml_feature(feature_name: str, ctx: ExperimentContext, max_samples: int = 0) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    train_part, val_part = audio_frames(ctx, max_samples)
    x_train = build_audio_feature_matrix(train_part, feature_name, ctx.data_cfg.train_crop, ctx)
    x_val = build_audio_feature_matrix(val_part, feature_name, ctx.data_cfg.eval_crop, ctx)
    y_train = train_part["label_id"].to_numpy()
    y_val = val_part["label_id"].to_numpy()
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "linear_svm": make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", max_iter=5000)),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            random_state=ctx.train_cfg.seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }
    rows = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        p, r, f1, _ = precision_recall_fscore_support(y_val, pred, average=PAPER_AVERAGE, zero_division=0)
        macro_f1 = precision_recall_fscore_support(y_val, pred, average="macro", zero_division=0)[2]
        rows.append(
            {
                "family": "ml",
                "feature": feature_name,
                "model": model_name,
                "paper_f1": float(f1),
                "paper_precision": float(p),
                "paper_recall": float(r),
                "macro_f1": float(macro_f1),
                "accuracy": float(accuracy_score(y_val, pred)),
            }
        )
        print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(ctx.output_dir / f"audio_ml_{feature_name}_results.csv", index=False)
    return out


class AudioFeatureMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 4, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class AudioFeatureCNN(nn.Module):
    def __init__(self, feature_shape: tuple[int, int, int], num_classes: int = 4, dropout: float = 0.25):
        super().__init__()
        channels, _, _ = feature_shape
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class AudioFeatureCRNN(nn.Module):
    def __init__(
        self,
        feature_shape: tuple[int, int, int],
        num_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        channels, _, _ = feature_shape
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d((2, 1)),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *feature_shape)
            _, conv_channels, conv_freq, _ = self.conv(dummy).shape
        self.rnn = nn.GRU(
            input_size=conv_channels * conv_freq,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2).flatten(2)
        out, _ = self.rnn(x)
        return self.classifier(out.mean(dim=1))


class AudioFeatureTransformer(nn.Module):
    def __init__(
        self,
        feature_shape: tuple[int, int, int],
        num_classes: int = 4,
        embed_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        channels, _, _ = feature_shape
        self.patch = nn.Conv2d(channels, embed_dim, kernel_size=(4, 4), stride=(4, 4))
        with torch.no_grad():
            dummy = torch.zeros(1, *feature_shape)
            tokens = self.patch(dummy).flatten(2).transpose(1, 2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, tokens.shape[1] + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed[:, : tokens.size(1) + 1]
        return self.classifier(self.encoder(tokens)[:, 0])


def build_audio_deep_model(model_kind: str, feature_name: str, input_dim: int) -> nn.Module:
    if model_kind == "mlp":
        return AudioFeatureMLP(input_dim, len(LABELS))
    shape = feature_map_shape(feature_name)
    if model_kind == "cnn":
        return AudioFeatureCNN(shape, len(LABELS))
    if model_kind == "crnn":
        return AudioFeatureCRNN(shape, len(LABELS))
    if model_kind == "transformer":
        return AudioFeatureTransformer(shape, len(LABELS))
    raise ValueError(f"Unknown deep model kind: {model_kind}")


def reshape_deep_features(x: torch.Tensor, feature_name: str, model_kind: str) -> torch.Tensor:
    if model_kind == "mlp":
        return x
    return x.view(x.shape[0], *feature_map_shape(feature_name))


def train_audio_deep_feature(
    feature_name: str,
    ctx: ExperimentContext,
    epochs: int | None = None,
    max_samples: int = 0,
    model_kind: str = "mlp",
    early_stopping_patience: int = 10,
    min_delta: float = 1e-4,
) -> dict:
    epochs = epochs or ctx.train_cfg.epochs
    train_part, val_part = audio_frames(ctx, max_samples)
    x_train = torch.tensor(build_audio_feature_matrix(train_part, feature_name, ctx.data_cfg.train_crop, ctx), dtype=torch.float32)
    y_train = torch.tensor(train_part["label_id"].to_numpy(), dtype=torch.long)
    x_val = torch.tensor(build_audio_feature_matrix(val_part, feature_name, ctx.data_cfg.eval_crop, ctx), dtype=torch.float32)
    y_val = torch.tensor(val_part["label_id"].to_numpy(), dtype=torch.long)

    train_loader = DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=ctx.train_cfg.batch_size * 8,
        shuffle=True,
    )
    val_loader = DataLoader(
        torch.utils.data.TensorDataset(x_val, y_val),
        batch_size=ctx.train_cfg.batch_size * 8,
        shuffle=False,
    )
    model = build_audio_deep_model(model_kind, feature_name, x_train.shape[1]).to(ctx.device)
    criterion = nn.CrossEntropyLoss(
        weight=make_class_weight_tensor(train_part, ctx) if ctx.train_cfg.class_weights else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=ctx.train_cfg.lr, weight_decay=ctx.train_cfg.weight_decay)
    best_metrics = {"paper_f1": -1.0}
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = reshape_deep_features(xb.to(ctx.device), feature_name, model_kind)
            yb = yb.to(ctx.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = reshape_deep_features(xb.to(ctx.device), feature_name, model_kind)
                logits_list.append(model(xb).cpu())
                labels_list.append(yb)
        metrics = compute_multiclass_metrics(logits_list, labels_list)
        print(f"audio_deep_{model_kind}_{feature_name} epoch={epoch} val_f1={metrics['paper_f1']:.4f}")
        if metrics["paper_f1"] > best_metrics["paper_f1"] + min_delta:
            best_metrics = metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "mode": f"audio_deep_{model_kind}_{feature_name}",
                    "model": model.state_dict(),
                    "feature": feature_name,
                    "model_kind": model_kind,
                    "labels": LABELS,
                    "best_val_metrics": metrics,
                },
                ctx.output_dir / f"best_audio_deep_{model_kind}_{feature_name}_model.pt",
            )
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(
                    f"audio_deep_{model_kind}_{feature_name} early stopping at epoch={epoch}; "
                    f"best_epoch={best_epoch} best_val_f1={best_metrics['paper_f1']:.4f}"
                )
                break

    result = {
        "mode": f"audio_deep_{model_kind}_{feature_name}",
        "feature": feature_name,
        "model_kind": model_kind,
        "best_paper_f1": best_metrics["paper_f1"],
        "best_epoch": best_epoch,
        "early_stopping_patience": early_stopping_patience,
        "min_delta": min_delta,
        "best_val_metrics": best_metrics,
        "weight_file": str(ctx.output_dir / f"best_audio_deep_{model_kind}_{feature_name}_model.pt"),
    }
    (ctx.output_dir / f"audio_deep_{model_kind}_{feature_name}_results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


class AudioFeatureMapDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_name: str,
        ctx: ExperimentContext,
        crop_mode: str,
        specaugment: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.feature_name = feature_name
        self.ctx = ctx
        self.crop_mode = crop_mode
        self.specaugment = specaugment
        self.pipe = make_audio_pipeline(ctx)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        x = audio_feature_map(row.audio_path, self.feature_name, self.crop_mode, self.ctx, self.pipe)
        if self.specaugment:
            x = apply_specaugment(x)
        y = torch.tensor(row.label_id, dtype=torch.long)
        return x, y


def apply_specaugment(x: torch.Tensor, freq_mask: int = 10, time_mask: int = 10) -> torch.Tensor:
    x = x.clone()
    _, freq, time = x.shape
    if freq > 1 and freq_mask > 0:
        width = int(torch.randint(0, min(freq_mask, freq) + 1, (1,)).item())
        if width > 0:
            start = int(torch.randint(0, freq - width + 1, (1,)).item())
            x[:, start : start + width, :] = 0
    if time > 1 and time_mask > 0:
        width = int(torch.randint(0, min(time_mask, time) + 1, (1,)).item())
        if width > 0:
            start = int(torch.randint(0, time - width + 1, (1,)).item())
            x[:, :, start : start + width] = 0
    return x


class PSLAEfficientNet(nn.Module):
    """PSLA-style log-mel image classifier using an EfficientNet backbone."""

    def __init__(self, num_classes: int = 4, dropout: float = 0.25, pretrained: bool = True):
        super().__init__()
        from torchvision import models

        weights = None
        if pretrained:
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT
            except AttributeError:
                weights = None
        try:
            self.backbone = models.efficientnet_b0(weights=weights)
        except Exception as exc:
            print(f"Could not load EfficientNet pretrained weights; using random init. Reason: {exc}")
            self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


def smooth_one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    off_value = smoothing / max(num_classes - 1, 1)
    on_value = 1.0 - smoothing
    target = torch.full((labels.size(0), num_classes), off_value, device=labels.device)
    target.scatter_(1, labels.unsqueeze(1), on_value)
    return target


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, class_weight: torch.Tensor | None = None) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    if class_weight is not None:
        targets = targets * class_weight.unsqueeze(0)
    return -(targets * log_probs).sum(dim=1).mean()


def mixup_batch(x: torch.Tensor, y_soft: torch.Tensor, alpha: float = 0.2):
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y_soft
    beta = torch.distributions.Beta(alpha, alpha)
    lam = float(beta.sample().item())
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[perm], lam * y_soft + (1.0 - lam) * y_soft[perm]


def make_balanced_sampler(frame: pd.DataFrame) -> WeightedRandomSampler:
    counts = frame["label_id"].value_counts().reindex(range(len(LABELS))).fillna(1).to_dict()
    weights = frame["label_id"].map(lambda label_id: 1.0 / float(counts[label_id])).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def evaluate_map_model(model, frame: pd.DataFrame, feature_name: str, ctx: ExperimentContext, crop_modes: tuple[str, ...]):
    model.eval()
    logits_by_crop = []
    labels = None
    with torch.no_grad():
        for crop_mode in crop_modes:
            dataset = AudioFeatureMapDataset(frame, feature_name, ctx, crop_mode, specaugment=False)
            loader = DataLoader(dataset, batch_size=ctx.train_cfg.batch_size * 4, shuffle=False, num_workers=0)
            crop_logits = []
            crop_labels = []
            for xb, yb in loader:
                crop_logits.append(model(xb.to(ctx.device)).cpu())
                crop_labels.append(yb)
            logits_by_crop.append(torch.cat(crop_logits, dim=0))
            if labels is None:
                labels = torch.cat(crop_labels, dim=0)
    mean_logits = torch.stack(logits_by_crop, dim=0).mean(dim=0)
    return compute_multiclass_metrics([mean_logits], [labels])


def train_audio_psla_efficientnet(
    ctx: ExperimentContext,
    epochs: int | None = None,
    max_samples: int = 0,
    mixup_alpha: float = 0.2,
    label_smoothing: float = 0.1,
    aggregate_eval: bool = True,
    pretrained: bool = True,
    early_stopping_patience: int = 10,
    min_delta: float = 1e-4,
) -> dict:
    epochs = epochs or ctx.train_cfg.epochs
    train_part, val_part = audio_frames(ctx, max_samples)
    train_ds = AudioFeatureMapDataset(
        train_part,
        "psla_logmel",
        ctx,
        ctx.data_cfg.train_crop,
        specaugment=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=ctx.train_cfg.batch_size,
        sampler=make_balanced_sampler(train_part),
        num_workers=ctx.train_cfg.num_workers,
        pin_memory=True,
    )
    model = PSLAEfficientNet(len(LABELS), pretrained=pretrained).to(ctx.device)
    class_weight = make_class_weight_tensor(train_part, ctx) if ctx.train_cfg.class_weights else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=ctx.train_cfg.lr, weight_decay=ctx.train_cfg.weight_decay)
    best_metrics = {"paper_f1": -1.0}
    best_epoch = 0
    epochs_without_improvement = 0
    eval_crops = (ctx.data_cfg.eval_crop, "center") if aggregate_eval and ctx.data_cfg.eval_crop != "center" else (ctx.data_cfg.eval_crop,)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in tqdm(train_loader, leave=False):
            xb = xb.to(ctx.device)
            yb = yb.to(ctx.device)
            y_soft = smooth_one_hot(yb, len(LABELS), label_smoothing)
            xb, y_soft = mixup_batch(xb, y_soft, mixup_alpha)
            optimizer.zero_grad(set_to_none=True)
            loss = soft_cross_entropy(model(xb), y_soft, class_weight)
            loss.backward()
            optimizer.step()

        metrics = evaluate_map_model(model, val_part, "psla_logmel", ctx, eval_crops)
        print(f"audio_deep_psla_efficientnet epoch={epoch} val_f1={metrics['paper_f1']:.4f}")
        if metrics["paper_f1"] > best_metrics["paper_f1"] + min_delta:
            best_metrics = metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "mode": "audio_deep_psla_efficientnet",
                    "model": model.state_dict(),
                    "feature": "psla_logmel",
                    "model_kind": "efficientnet_b0_psla",
                    "labels": LABELS,
                    "psla_recipe": {
                        "pretraining": "ImageNet EfficientNet-B0 when weights are available",
                        "sampling": "class-balanced WeightedRandomSampler",
                        "labeling": "label smoothing plus mixup soft labels",
                        "augmentation": "SpecAugment frequency/time masking",
                        "aggregation": "mean logits over evaluation crops" if aggregate_eval else "single evaluation crop",
                    },
                    "best_val_metrics": metrics,
                },
                ctx.output_dir / "best_audio_deep_psla_efficientnet_model.pt",
            )
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(
                    f"audio_deep_psla_efficientnet early stopping at epoch={epoch}; "
                    f"best_epoch={best_epoch} best_val_f1={best_metrics['paper_f1']:.4f}"
                )
                break

    result = {
        "mode": "audio_deep_psla_efficientnet",
        "feature": "psla_logmel",
        "model_kind": "efficientnet_b0_psla",
        "best_paper_f1": best_metrics["paper_f1"],
        "best_epoch": best_epoch,
        "early_stopping_patience": early_stopping_patience,
        "min_delta": min_delta,
        "best_val_metrics": best_metrics,
        "weight_file": str(ctx.output_dir / "best_audio_deep_psla_efficientnet_model.pt"),
    }
    (ctx.output_dir / "audio_deep_psla_efficientnet_results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result
