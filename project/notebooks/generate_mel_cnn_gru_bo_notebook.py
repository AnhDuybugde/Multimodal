import json
from pathlib import Path


NOTEBOOK_CODE = r'''# Dataset root - edit this first if your Kaggle input path changes.
DATASET_ROOT = "/kaggle/input/datasets/duyem54/visual-audio/dataset"

# Smoke-test knobs. Full run defaults are N_BO_TRIALS=20 and EPOCHS_PER_TRIAL=20.
MAX_SAMPLES_PER_CLASS = 0
N_BO_TRIALS = 20
EPOCHS_PER_TRIAL = 20
EARLY_STOPPING_PATIENCE = 5
SEED = 42


import json
import math
import os
import random
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

LABELS = ("ambient", "leaf", "trunk", "twig")
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
TRAIN_SPLIT = "audio_visual_dataset_default"
ROBOT_SPLIT = "audio_visual_dataset_robo_default"


@dataclass(frozen=True)
class BaseAudioConfig:
    target_sample_rate: int = 16000
    audio_window_sec: float = 0.8
    train_crop: str = "random"
    eval_crop: str = "energy"

    @property
    def window_samples(self) -> int:
        return int(round(self.target_sample_rate * self.audio_window_sec))


BASE_AUDIO_CFG = BaseAudioConfig()
WAVEFORM_CACHE = {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_data_root(data_root: str | Path | None = None) -> Path:
    candidates = []
    if data_root is not None:
        root = Path(data_root)
        candidates.extend([root, root / "raw_dataset", root / "prepared_data", root / "dataset"])
    candidates.extend([
        Path("/kaggle/input/datasets/anhduy54/visual-audio/raw_dataset"),
        Path("/kaggle/input/visual-audio/raw_dataset"),
        Path("/kaggle/input/raw_dataset"),
        Path("dataset"),
        Path("."),
    ])
    for candidate in candidates:
        if (candidate / TRAIN_SPLIT / "dataset.csv").exists():
            return candidate
    return Path(data_root) if data_root is not None else candidates[0]


def build_index(data_root: str | Path, skip_missing_files: bool = True) -> pd.DataFrame:
    data_root = Path(data_root)
    rows = []
    skipped = 0
    for split_name in (TRAIN_SPLIT, ROBOT_SPLIT):
        split_dir = data_root / split_name
        csv_path = split_dir / "dataset.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for item in df.to_dict("records"):
            audio_path = split_dir / item["audio_file"]
            if skip_missing_files and not audio_path.exists():
                skipped += 1
                continue
            label = item["category"]
            rows.append({
                "split_name": split_name,
                "audio_path": str(audio_path),
                "label": label,
                "label_id": LABEL_TO_ID[label],
                "audio_file": item["audio_file"],
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise FileNotFoundError(f"No dataset rows found under {data_root}")
    print(f"indexed rows={len(out)} skipped_missing_audio={skipped}")
    print(out.groupby(["split_name", "label"]).size())
    return out


def sample_stratified(frame: pd.DataFrame, max_samples_per_class: int, seed: int) -> pd.DataFrame:
    if max_samples_per_class <= 0:
        return frame.reset_index(drop=True)
    parts = []
    rng = np.random.default_rng(seed)
    for label in LABELS:
        rows = frame[frame["label"] == label]
        if rows.empty:
            continue
        take = min(max_samples_per_class, len(rows))
        choices = rng.choice(len(rows), size=take, replace=False)
        parts.append(rows.iloc[choices])
    return pd.concat(parts, ignore_index=True) if parts else frame.head(0)


def _load_wav_stdlib(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        sign = (bytes_[:, 2] >= 128).astype(np.uint8) * 255
        padded = np.column_stack([bytes_, sign]).astype(np.uint8)
        data = padded.reshape(-1, 4).view("<i4").reshape(-1).astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), int(sample_rate)


def resample_waveform(waveform: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if sample_rate == target_sample_rate:
        return waveform.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly
        gcd = math.gcd(sample_rate, target_sample_rate)
        return resample_poly(waveform, target_sample_rate // gcd, sample_rate // gcd).astype(np.float32)
    except Exception:
        duration = len(waveform) / float(sample_rate)
        old_x = np.linspace(0.0, duration, num=len(waveform), endpoint=False)
        new_len = max(1, int(round(duration * target_sample_rate)))
        new_x = np.linspace(0.0, duration, num=new_len, endpoint=False)
        return np.interp(new_x, old_x, waveform).astype(np.float32)


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    waveform = np.nan_to_num(waveform.astype(np.float32, copy=False))
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 1e-6:
        waveform = waveform / peak
    return waveform.astype(np.float32)


def energy_start(waveform: np.ndarray, target_len: int) -> int:
    if len(waveform) <= target_len:
        return 0
    power = waveform.astype(np.float64) ** 2
    csum = np.concatenate([[0.0], np.cumsum(power)])
    window_energy = csum[target_len:] - csum[:-target_len]
    return int(np.argmax(window_energy))


def crop_or_pad(waveform: np.ndarray, target_len: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    total = len(waveform)
    if total < target_len:
        return np.pad(waveform, (0, target_len - total)).astype(np.float32)
    if total == target_len:
        return waveform.astype(np.float32, copy=False)
    max_start = total - target_len
    if mode == "random":
        start = int(rng.integers(0, max_start + 1))
    elif mode == "energy":
        start = energy_start(waveform, target_len)
    else:
        start = max_start // 2
    return waveform[start : start + target_len].astype(np.float32)


def load_processed_waveform(path: str | Path, crop_mode: str, seed: int) -> torch.Tensor:
    key = (str(path), crop_mode, seed if crop_mode == "random" else 0)
    if key not in WAVEFORM_CACHE:
        rng = np.random.default_rng(seed)
        waveform, sample_rate = _load_wav_stdlib(path)
        waveform = resample_waveform(waveform, sample_rate, BASE_AUDIO_CFG.target_sample_rate)
        waveform = normalize_waveform(waveform)
        waveform = crop_or_pad(waveform, BASE_AUDIO_CFG.window_samples, crop_mode, rng)
        WAVEFORM_CACHE[key] = torch.from_numpy(waveform).float()
    return WAVEFORM_CACHE[key]


def hz_to_mel(freq):
    return 2595.0 * torch.log10(1.0 + torch.as_tensor(freq).float() / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (torch.as_tensor(mel).float() / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, device=None) -> torch.Tensor:
    min_mel = hz_to_mel(torch.tensor(0.0))
    max_mel = hz_to_mel(torch.tensor(float(sample_rate) / 2.0))
    mels = torch.linspace(min_mel, max_mel, n_mels + 2)
    hz = mel_to_hz(mels)
    bins = torch.floor((n_fft + 1) * hz / sample_rate).long().clamp(0, n_fft // 2)
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        left, center, right = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb.to(device) if device is not None else fb


class LogMelTransform:
    def __init__(self, n_mels: int, n_fft: int, hop_length: int):
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.window = torch.hann_window(self.n_fft)
        self.mel_filter = mel_filterbank(BASE_AUDIO_CFG.target_sample_rate, self.n_fft, self.n_mels)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.float()
        window = self.window.to(waveform.device)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )
        power = spec.abs().pow(2).clamp_min(1e-10)
        mel = self.mel_filter.to(waveform.device) @ power
        mel_db = 10.0 * torch.log10(mel.clamp_min(1e-10))
        mel_db = torch.clamp(mel_db, min=float(mel_db.max()) - 80.0)
        mel_norm = ((mel_db + 80.0) / 80.0).clamp(0.0, 1.0)
        return mel_norm.unsqueeze(0)


class MelAudioDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, crop_mode: str, trial_seed: int, mel_cfg: dict):
        self.frame = frame.reset_index(drop=True)
        self.crop_mode = crop_mode
        self.trial_seed = int(trial_seed)
        self.mel = LogMelTransform(mel_cfg["n_mels"], mel_cfg["n_fft"], mel_cfg["hop_length"])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        waveform = load_processed_waveform(row.audio_path, self.crop_mode, self.trial_seed + idx)
        x = self.mel(waveform)
        y = torch.tensor(int(row.label_id), dtype=torch.long)
        return x, y


class MelCNNGRU(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        conv_filters_1: int,
        conv_filters_2: int,
        kernel_size: int,
        pool_freq: int,
        hidden_dim: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        num_classes: int = 4,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_filters_1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(conv_filters_1),
            nn.GELU(),
            nn.MaxPool2d((pool_freq, 1)),
            nn.Conv2d(conv_filters_1, conv_filters_2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(conv_filters_2),
            nn.GELU(),
            nn.MaxPool2d((pool_freq, 1)),
        )
        frames = 1 + max(0, BASE_AUDIO_CFG.window_samples - n_fft) // hop_length
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, max(frames, 1))
            _, conv_channels, conv_freq, _ = self.conv(dummy).shape
        self.rnn = nn.GRU(
            input_size=conv_channels * conv_freq,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2).flatten(2)
        out, _ = self.rnn(x)
        return self.classifier(out.mean(dim=1))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, paper_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    binary_true = (np.asarray(y_true) != LABEL_TO_ID["ambient"]).astype(np.int64)
    binary_pred = (np.asarray(y_pred) != LABEL_TO_ID["ambient"]).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "paper_precision": float(precision),
        "paper_recall": float(recall),
        "paper_f1": float(paper_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "binary_contact_f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
    }


def class_weight_tensor(frame: pd.DataFrame, device) -> torch.Tensor:
    counts = frame["label_id"].value_counts().reindex(range(len(LABELS)), fill_value=0).to_numpy(dtype=np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_model(model, loader, device) -> dict:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(yb.numpy())
    return classification_metrics(np.concatenate(labels), np.concatenate(preds))


def train_one_trial(train_df: pd.DataFrame, eval_df: pd.DataFrame, config: dict, trial_number: int) -> dict:
    set_seed(SEED + trial_number)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_cfg = {k: config[k] for k in ("n_mels", "n_fft", "hop_length")}
    train_ds = MelAudioDataset(train_df, BASE_AUDIO_CFG.train_crop, SEED + trial_number * 1000, mel_cfg)
    eval_ds = MelAudioDataset(eval_df, BASE_AUDIO_CFG.eval_crop, SEED + trial_number * 1000, mel_cfg)
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)

    model = MelCNNGRU(
        n_mels=int(config["n_mels"]),
        n_fft=int(config["n_fft"]),
        hop_length=int(config["hop_length"]),
        conv_filters_1=int(config["conv_filters_1"]),
        conv_filters_2=int(config["conv_filters_2"]),
        kernel_size=int(config["kernel_size"]),
        pool_freq=int(config["pool_freq"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        bidirectional=bool(config["bidirectional"]),
        dropout=float(config["dropout"]),
        num_classes=len(LABELS),
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(train_df, device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    best_metrics = {"paper_f1": -1.0}
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, EPOCHS_PER_TRIAL + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        metrics = evaluate_model(model, eval_loader, device)
        if metrics["paper_f1"] > best_metrics["paper_f1"] + 1e-4:
            best_metrics = metrics
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if EARLY_STOPPING_PATIENCE and no_improve >= EARLY_STOPPING_PATIENCE:
                break
    return {"best_epoch": best_epoch, **best_metrics}


def make_config(**kwargs) -> dict:
    return {
        "n_mels": int(kwargs["n_mels"]),
        "n_fft": int(kwargs["n_fft"]),
        "hop_length": int(kwargs["hop_length"]),
        "conv_filters_1": int(kwargs["conv_filters_1"]),
        "conv_filters_2": int(kwargs["conv_filters_2"]),
        "kernel_size": int(kwargs["kernel_size"]),
        "pool_freq": int(kwargs["pool_freq"]),
        "hidden_dim": int(kwargs["hidden_dim"]),
        "num_layers": int(kwargs["num_layers"]),
        "bidirectional": True,
        "dropout": float(kwargs["dropout"]),
        "lr": float(kwargs["lr"]),
        "weight_decay": float(kwargs["weight_decay"]),
        "batch_size": int(kwargs.get("batch_size", 32)),
    }


def decode_index(value, choices):
    idx = int(np.clip(round(float(value)), 0, len(choices) - 1))
    return choices[idx]


def config_from_bayes_params(**params) -> dict:
    return make_config(
        n_mels=decode_index(params["n_mels_i"], [64, 128]),
        n_fft=decode_index(params["n_fft_i"], [512, 1024]),
        hop_length=decode_index(params["hop_length_i"], [128, 256]),
        conv_filters_1=decode_index(params["conv_filters_1_i"], [16, 32, 64]),
        conv_filters_2=decode_index(params["conv_filters_2_i"], [32, 64, 128]),
        kernel_size=decode_index(params["kernel_size_i"], [3, 5]),
        pool_freq=decode_index(params["pool_freq_i"], [2, 4]),
        hidden_dim=decode_index(params["hidden_dim_i"], [64, 128, 256]),
        num_layers=decode_index(params["num_layers_i"], [1, 2]),
        dropout=float(params["dropout"]),
        lr=float(10 ** params["lr_log10"]),
        weight_decay=float(10 ** params["weight_decay_log10"]),
        batch_size=32,
    )


def run_bayes_opt_backend(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    try:
        from bayes_opt import BayesianOptimization
    except Exception as bayes_exc:
        try:
            import optuna
        except Exception as optuna_exc:
            raise RuntimeError(
                "Bayesian optimization requires either `bayesian-optimization` (`bayes_opt`) "
                "or `optuna`. Install one of them before running this notebook."
            ) from optuna_exc
        print(f"`bayes_opt` unavailable ({bayes_exc}); using Optuna TPE sampler fallback.")
        return run_optuna_backend(train_df, eval_df, optuna)

    rows = []
    trial_counter = {"n": 0}

    def objective(**params):
        trial_counter["n"] += 1
        trial = trial_counter["n"]
        config = config_from_bayes_params(**params)
        start = time.time()
        metrics = train_one_trial(train_df, eval_df, config, trial)
        row = {
            "trial": trial,
            "backend": "bayes_opt",
            "seconds": round(time.time() - start, 3),
            "config": json.dumps(config, sort_keys=True),
            **config,
            **metrics,
        }
        rows.append(row)
        print(
            f'trial={trial:03d} paper_f1={row["paper_f1"]:.4f} macro_f1={row["macro_f1"]:.4f} '
            f'epoch={row["best_epoch"]} config={row["config"]}'
        )
        return row["paper_f1"]

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            "n_mels_i": (0, 1),
            "n_fft_i": (0, 1),
            "hop_length_i": (0, 1),
            "conv_filters_1_i": (0, 2),
            "conv_filters_2_i": (0, 2),
            "kernel_size_i": (0, 1),
            "pool_freq_i": (0, 1),
            "hidden_dim_i": (0, 2),
            "num_layers_i": (0, 1),
            "dropout": (0.1, 0.5),
            "lr_log10": (math.log10(1e-4), math.log10(3e-3)),
            "weight_decay_log10": (math.log10(1e-5), math.log10(1e-3)),
        },
        random_state=SEED,
        verbose=0,
    )
    init_points = min(5, max(1, N_BO_TRIALS // 4))
    optimizer.maximize(init_points=init_points, n_iter=max(0, N_BO_TRIALS - init_points))
    return pd.DataFrame(rows)


def run_optuna_backend(train_df: pd.DataFrame, eval_df: pd.DataFrame, optuna) -> pd.DataFrame:
    rows = []

    def objective(trial):
        config = make_config(
            n_mels=trial.suggest_categorical("n_mels", [64, 128]),
            n_fft=trial.suggest_categorical("n_fft", [512, 1024]),
            hop_length=trial.suggest_categorical("hop_length", [128, 256]),
            conv_filters_1=trial.suggest_categorical("conv_filters_1", [16, 32, 64]),
            conv_filters_2=trial.suggest_categorical("conv_filters_2", [32, 64, 128]),
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5]),
            pool_freq=trial.suggest_categorical("pool_freq", [2, 4]),
            hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            num_layers=trial.suggest_categorical("num_layers", [1, 2]),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            batch_size=32,
        )
        start = time.time()
        metrics = train_one_trial(train_df, eval_df, config, trial.number + 1)
        row = {
            "trial": trial.number + 1,
            "backend": "optuna_tpe",
            "seconds": round(time.time() - start, 3),
            "config": json.dumps(config, sort_keys=True),
            **config,
            **metrics,
        }
        rows.append(row)
        print(
            f'trial={row["trial"]:03d} paper_f1={row["paper_f1"]:.4f} macro_f1={row["macro_f1"]:.4f} '
            f'epoch={row["best_epoch"]} config={row["config"]}'
        )
        return row["paper_f1"]

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_BO_TRIALS)
    return pd.DataFrame(rows)


def run_experiment() -> pd.DataFrame:
    set_seed(SEED)
    data_root = find_data_root(DATASET_ROOT)
    print("data_root =", data_root)
    index = build_index(data_root)
    train_df = index[index["split_name"] == TRAIN_SPLIT].reset_index(drop=True)
    eval_df = index[index["split_name"] == ROBOT_SPLIT].reset_index(drop=True)
    if eval_df.empty:
        from sklearn.model_selection import train_test_split
        train_df, eval_df = train_test_split(
            train_df,
            test_size=0.2,
            random_state=SEED,
            stratify=train_df["label_id"],
        )
        train_df = train_df.reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)
        print("ROBOT split not found; using stratified validation split from train data.")
    train_df = sample_stratified(train_df, MAX_SAMPLES_PER_CLASS, SEED)
    eval_df = sample_stratified(eval_df, MAX_SAMPLES_PER_CLASS, SEED + 1)
    print("train rows =", len(train_df), "eval rows =", len(eval_df))
    print("N_BO_TRIALS =", N_BO_TRIALS, "EPOCHS_PER_TRIAL =", EPOCHS_PER_TRIAL)
    results = run_bayes_opt_backend(train_df, eval_df)
    if not results.empty:
        results = results.sort_values(["paper_f1", "macro_f1"], ascending=False).reset_index(drop=True)
    return results


results = run_experiment()

print("\n=== ALL BO TRIALS SORTED BY PAPER F1 ===")
display(results)

print("\n=== TOP 10 CONFIGS ===")
display(results.head(10))

best = results.iloc[0].to_dict() if not results.empty else {}
best_metrics = {k: best.get(k) for k in ("paper_f1", "macro_f1", "accuracy", "binary_contact_f1")}

print("\n=== BEST CONFIG JSON ===")
print(best.get("config", "{}"))

print("\n=== BEST VALIDATION METRICS ===")
print(json.dumps(best_metrics, indent=2))
'''


def make_notebook() -> dict:
    markdown = """# Audio Deep Mel-CNN-GRU + Bayesian Optimization

Self-contained Kaggle-style notebook for the paper-aligned deep bee-sound direction: log-mel spectrogram maps, CNN feature extraction, GRU temporal modeling, and Bayesian optimization over core architecture/training hyperparameters.
"""
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": markdown.splitlines(True)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": NOTEBOOK_CODE.splitlines(True)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    path = out_dir / "audio-deep-mel-cnn-gru-bo.ipynb"
    path.write_text(json.dumps(make_notebook(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
