import json
from pathlib import Path


CLASSIFIER_SPACES = {
    "knn": {
        "title": "KNN",
        "imports": "from sklearn.neighbors import KNeighborsClassifier",
        "space": """[
    {"n_neighbors": int(rng.choice([3, 5, 7, 9, 11, 15, 21])), "weights": str(rng.choice(["uniform", "distance"])), "p": int(rng.choice([1, 2]))}
]""",
        "model": """KNeighborsClassifier(
        n_neighbors=params["n_neighbors"],
        weights=params["weights"],
        p=params["p"],
        n_jobs=-1,
    )""",
    },
    "svm": {
        "title": "SVM",
        "imports": "from sklearn.svm import SVC",
        "space": """[
    {"C": float(10 ** rng.uniform(-2, 2)), "kernel": str(rng.choice(["linear", "rbf"])), "gamma": str(rng.choice(["scale", "auto"]))}
]""",
        "model": """make_pipeline(
        StandardScaler(),
        SVC(
            C=params["C"],
            kernel=params["kernel"],
            gamma=params["gamma"],
            class_weight="balanced",
            random_state=SEED,
        ),
    )""",
    },
    "lr": {
        "title": "Logistic Regression",
        "imports": "from sklearn.linear_model import LogisticRegression",
        "space": """[
    {"C": float(10 ** rng.uniform(-3, 2)), "solver": str(rng.choice(["lbfgs", "saga"]))}
]""",
        "model": """make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=params["C"],
            solver=params["solver"],
            penalty="l2",
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    )""",
    },
    "rf": {
        "title": "Random Forest",
        "imports": "from sklearn.ensemble import RandomForestClassifier",
        "space": """[
    {
        "n_estimators": int(rng.choice([200, 300, 500, 800])),
        "max_depth": none_or_int(rng.choice([0, 8, 12, 16, 24])),
        "min_samples_split": int(rng.choice([2, 5, 10])),
        "min_samples_leaf": int(rng.choice([1, 2, 4])),
        "max_features": str(rng.choice(["sqrt", "log2"])),
    }
]""",
        "model": """RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )""",
    },
    "et": {
        "title": "Extra Trees",
        "imports": "from sklearn.ensemble import ExtraTreesClassifier",
        "space": """[
    {
        "n_estimators": int(rng.choice([200, 300, 500, 800])),
        "max_depth": none_or_int(rng.choice([0, 8, 12, 16, 24])),
        "min_samples_split": int(rng.choice([2, 5, 10])),
        "min_samples_leaf": int(rng.choice([1, 2, 4])),
        "max_features": str(rng.choice(["sqrt", "log2"])),
    }
]""",
        "model": """ExtraTreesClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )""",
    },
    "xgb": {
        "title": "XGBoost",
        "imports": "from xgboost import XGBClassifier",
        "space": """[
    {
        "n_estimators": int(rng.choice([200, 300, 500, 800])),
        "max_depth": int(rng.choice([3, 4, 5, 6])),
        "learning_rate": float(rng.choice([0.01, 0.03, 0.05, 0.1])),
        "subsample": float(rng.choice([0.7, 0.85, 1.0])),
        "colsample_bytree": float(rng.choice([0.7, 0.85, 1.0])),
        "min_child_weight": float(rng.choice([1.0, 3.0, 5.0])),
    }
]""",
        "model": """XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        objective="multi:softprob",
        num_class=len(LABELS),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )""",
    },
}


COMMON_CODE = r'''# Dataset root - edit this first if your Kaggle input path changes.
DATASET_ROOT = "/kaggle/input/datasets/duyem54/visual-audio/dataset"

# Random Search controls.
N_RANDOM_TRIALS = 60
MAX_SAMPLES_PER_CLASS = 0  # 0 means full data.
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
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


os.environ["CUDA_VISIBLE_DEVICES"] = ""

LABELS = ("ambient", "leaf", "trunk", "twig")
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
TRAIN_SPLIT = "audio_visual_dataset_default"
ROBOT_SPLIT = "audio_visual_dataset_robo_default"


@dataclass(frozen=True)
class AudioConfig:
    target_sample_rate: int = 16000
    audio_window_sec: float = 0.8
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    train_crop: str = "random"
    eval_crop: str = "energy"

    @property
    def window_samples(self) -> int:
        return int(round(self.target_sample_rate * self.audio_window_sec))


AUDIO_CFG = AudioConfig()
FEATURE_KINDS = ("fft", "stft", "mfcc", "mfcc_stft")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def none_or_int(value):
    value = int(value)
    return None if value == 0 else value


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
        idx = rng.choice(len(rows), size=take, replace=False)
        parts.append(rows.iloc[idx])
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


def load_processed_waveform(path: str | Path, crop_mode: str, rng: np.random.Generator) -> np.ndarray:
    waveform, sample_rate = _load_wav_stdlib(path)
    waveform = resample_waveform(waveform, sample_rate, AUDIO_CFG.target_sample_rate)
    waveform = normalize_waveform(waveform)
    return crop_or_pad(waveform, AUDIO_CFG.window_samples, crop_mode, rng)


def stft_complex(waveform: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    waveform = waveform.astype(np.float32, copy=False)
    if len(waveform) < n_fft:
        waveform = np.pad(waveform, (0, n_fft - len(waveform)))
    n_frames = int(math.ceil(max(0, len(waveform) - n_fft) / hop_length)) + 1
    padded_len = n_fft + hop_length * (n_frames - 1)
    if len(waveform) < padded_len:
        waveform = np.pad(waveform, (0, padded_len - len(waveform)))
    shape = (n_frames, n_fft)
    strides = (waveform.strides[0] * hop_length, waveform.strides[0])
    frames = np.lib.stride_tricks.as_strided(waveform, shape=shape, strides=strides).copy()
    frames *= np.hanning(n_fft).astype(np.float32)
    return np.fft.rfft(frames, n=n_fft, axis=1).T


def hz_to_mel(freq):
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2.0), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


def dct_matrix(n_mels: int, n_mfcc: int) -> np.ndarray:
    n = np.arange(n_mels, dtype=np.float32)
    k = np.arange(n_mfcc, dtype=np.float32)[:, None]
    basis = np.cos(math.pi / n_mels * (n + 0.5) * k).astype(np.float32)
    basis[0] *= math.sqrt(1.0 / n_mels)
    if n_mfcc > 1:
        basis[1:] *= math.sqrt(2.0 / n_mels)
    return basis


def resize_vector(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) == length:
        return values
    old_x = np.linspace(0.0, 1.0, len(values))
    new_x = np.linspace(0.0, 1.0, length)
    return np.interp(new_x, old_x, values).astype(np.float32)


MEL_FILTER = mel_filterbank(AUDIO_CFG.target_sample_rate, AUDIO_CFG.n_fft, AUDIO_CFG.n_mels)
DCT_CACHE = {}
FEATURE_CACHE = {}
WAVEFORM_CACHE = {}


def mfcc_map_from_waveform(waveform: np.ndarray, n_mfcc: int) -> np.ndarray:
    spec = stft_complex(waveform, AUDIO_CFG.n_fft, AUDIO_CFG.hop_length)
    power = np.maximum(np.abs(spec) ** 2, 1e-10).astype(np.float32)
    mel_power = np.maximum(MEL_FILTER @ power, 1e-10)
    mel_db = 10.0 * np.log10(mel_power)
    mel_db = np.maximum(mel_db, mel_db.max() - 80.0)
    if n_mfcc not in DCT_CACHE:
        DCT_CACHE[n_mfcc] = dct_matrix(AUDIO_CFG.n_mels, n_mfcc)
    return (DCT_CACHE[n_mfcc] @ mel_db).astype(np.float32)


def summarize_mfcc(waveform: np.ndarray, n_mfcc: int, stats: str) -> np.ndarray:
    mfcc = mfcc_map_from_waveform(waveform, n_mfcc)
    parts = [mfcc.mean(axis=1)]
    if stats == "mean_std":
        parts.append(mfcc.std(axis=1))
    return np.concatenate(parts, axis=0).astype(np.float32)


def summarize_stft(waveform: np.ndarray, bins: int) -> np.ndarray:
    spec = stft_complex(waveform, AUDIO_CFG.n_fft, AUDIO_CFG.hop_length)
    freq_mean = np.log1p(np.abs(spec)).mean(axis=1)
    return resize_vector(freq_mean, bins)


def summarize_fft(waveform: np.ndarray, bins: int) -> np.ndarray:
    fft_mag = np.log1p(np.abs(np.fft.rfft(waveform.astype(np.float32))))
    return resize_vector(fft_mag, bins)


def feature_dim(config: dict) -> int:
    kind = config["feature_kind"]
    if kind == "fft":
        return int(config["fft_bins"])
    if kind == "stft":
        return int(config["stft_bins"])
    if kind == "mfcc":
        return int(config["mfcc_n_coeffs"]) * (2 if config["mfcc_stats"] == "mean_std" else 1)
    if kind == "mfcc_stft":
        return int(config["mfcc_n_coeffs"]) + int(config["stft_bins"])
    raise ValueError(kind)


def extract_feature_vector(waveform: np.ndarray, config: dict) -> np.ndarray:
    kind = config["feature_kind"]
    if kind == "fft":
        return summarize_fft(waveform, int(config["fft_bins"]))
    if kind == "stft":
        return summarize_stft(waveform, int(config["stft_bins"]))
    if kind == "mfcc":
        return summarize_mfcc(waveform, int(config["mfcc_n_coeffs"]), str(config["mfcc_stats"]))
    if kind == "mfcc_stft":
        mfcc = summarize_mfcc(waveform, int(config["mfcc_n_coeffs"]), "mean")
        stft = summarize_stft(waveform, int(config["stft_bins"]))
        return np.concatenate([mfcc, stft], axis=0).astype(np.float32)
    raise ValueError(kind)


def get_waveform(path: str, crop_mode: str, seed: int) -> np.ndarray:
    key = (path, crop_mode, seed if crop_mode == "random" else 0)
    if key not in WAVEFORM_CACHE:
        rng = np.random.default_rng(seed)
        WAVEFORM_CACHE[key] = load_processed_waveform(path, crop_mode, rng)
    return WAVEFORM_CACHE[key]


def build_feature_matrix(frame: pd.DataFrame, config: dict, crop_mode: str, seed: int) -> np.ndarray:
    key = (json.dumps(config, sort_keys=True), crop_mode, seed, tuple(frame["audio_path"].tolist()))
    if key in FEATURE_CACHE:
        return FEATURE_CACHE[key]
    rows = []
    for item in tqdm(frame.to_dict("records"), desc=f'{config["feature_kind"]}/{feature_dim(config)}d/{crop_mode}', leave=False):
        waveform = get_waveform(item["audio_path"], crop_mode, seed)
        rows.append(extract_feature_vector(waveform, config))
    x = np.asarray(rows, dtype=np.float32)
    FEATURE_CACHE[key] = x
    return x


def sample_feature_config(rng: np.random.Generator) -> dict:
    kind = str(rng.choice(FEATURE_KINDS))
    if kind == "fft":
        return {"feature_kind": "fft", "fft_bins": int(rng.choice([40, 80, 128]))}
    if kind == "stft":
        return {"feature_kind": "stft", "stft_bins": int(rng.choice([80, 257]))}
    if kind == "mfcc":
        return {
            "feature_kind": "mfcc",
            "mfcc_n_coeffs": int(rng.choice([20, 40])),
            "mfcc_stats": str(rng.choice(["mean", "mean_std"])),
        }
    if kind == "mfcc_stft":
        return {
            "feature_kind": "mfcc_stft",
            "mfcc_n_coeffs": int(rng.choice([20, 40])),
            "mfcc_stats": "mean",
            "stft_bins": 80,
        }
    raise ValueError(kind)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    binary_true = (np.asarray(y_true) != LABEL_TO_ID["ambient"]).astype(np.int64)
    binary_pred = (np.asarray(y_pred) != LABEL_TO_ID["ambient"]).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "paper_precision": float(precision),
        "paper_recall": float(recall),
        "paper_f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "binary_contact_f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
    }
'''


RUN_TEMPLATE = r'''
{classifier_imports}

CLASSIFIER_NAME = "{classifier_name}"


def sample_model_params(rng: np.random.Generator) -> dict:
    candidates = {classifier_space}
    return candidates[0]


def build_model(params: dict):
    return {classifier_model}


def run_random_search() -> pd.DataFrame:
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
    y_train = train_df["label_id"].to_numpy()
    y_eval = eval_df["label_id"].to_numpy()

    print("classifier =", CLASSIFIER_NAME)
    print("feature_kind search =", FEATURE_KINDS)
    print("trials =", N_RANDOM_TRIALS)
    print("train rows =", len(train_df), "eval rows =", len(eval_df))

    rng = np.random.default_rng(SEED)
    rows = []
    for trial in range(1, N_RANDOM_TRIALS + 1):
        feature_config = sample_feature_config(rng)
        model_params = sample_model_params(rng)
        start = time.time()
        x_train = build_feature_matrix(train_df, feature_config, AUDIO_CFG.train_crop, SEED + trial)
        x_eval = build_feature_matrix(eval_df, feature_config, AUDIO_CFG.eval_crop, SEED + trial)
        model = build_model(model_params)
        model.fit(x_train, y_train)
        pred = model.predict(x_eval)
        metrics = classification_metrics(y_eval, pred)
        elapsed = time.time() - start
        row = {
            "trial": trial,
            "classifier": CLASSIFIER_NAME,
            "feature_kind": feature_config["feature_kind"],
            "feature_dim": feature_dim(feature_config),
            "feature_config": json.dumps(feature_config, sort_keys=True),
            "model_params": json.dumps(model_params, sort_keys=True),
            "seconds": round(elapsed, 3),
            **metrics,
        }
        rows.append(row)
        print(
            f'trial={trial:03d} feature={row["feature_kind"]} dim={row["feature_dim"]} '
            f'paper_f1={row["paper_f1"]:.4f} macro_f1={row["macro_f1"]:.4f} seconds={row["seconds"]:.1f}'
        )

    results = pd.DataFrame(rows).sort_values(["paper_f1", "macro_f1"], ascending=False).reset_index(drop=True)
    return results


results = run_random_search()

print("\n=== ALL RANDOM SEARCH TRIALS SORTED BY PAPER F1 ===")
display(results)

print("\n=== TOP 10 CONFIGS ===")
display(results.head(10))

best = results.iloc[0].to_dict()
print("\n=== BEST CONFIG JSON ===")
print(json.dumps(best, indent=2))
'''


def make_notebook(classifier_name: str, spec: dict) -> dict:
    title = f"# Audio ML Random Search - {spec['title']}\n\n"
    title += (
        "Paper-aligned low-dimensional feature search over `fft`, `stft`, `mfcc`, "
        "and `mfcc_stft`. This notebook tunes one classifier only so the six ML "
        "models can run in parallel.\n"
    )
    run_code = RUN_TEMPLATE
    run_code = run_code.replace("{classifier_imports}", spec["imports"])
    run_code = run_code.replace("{classifier_name}", classifier_name)
    run_code = run_code.replace("{classifier_space}", spec["space"])
    run_code = run_code.replace("{classifier_model}", spec["model"])
    code = COMMON_CODE + run_code
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": title.splitlines(True)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)},
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
    for classifier_name, spec in CLASSIFIER_SPACES.items():
        nb = make_notebook(classifier_name, spec)
        path = out_dir / f"audio-ml-random-search-{classifier_name}.ipynb"
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
