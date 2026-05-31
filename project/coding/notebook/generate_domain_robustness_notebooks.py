from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).resolve().parent

VARIANTS = [
    {
        "filename": "audio_ml_xgboost_mfcc_fft_h2_spectral_gate.ipynb",
        "title": "XGBoost MFCC+FFT + H2 Spectral Gate",
        "variant_name": "xgb_mfcc_fft_h2_spectral_gate",
        "use_spectral_gate": True,
        "use_noise_aug": False,
        "use_freq_mixstyle": False,
    },
    {
        "filename": "audio_ml_xgboost_mfcc_fft_h3_noise_aug.ipynb",
        "title": "XGBoost MFCC+FFT + H3 Noise Augmentation",
        "variant_name": "xgb_mfcc_fft_h3_noise_aug",
        "use_spectral_gate": False,
        "use_noise_aug": True,
        "use_freq_mixstyle": False,
    },
    {
        "filename": "audio_ml_xgboost_mfcc_fft_h4_freq_mixstyle.ipynb",
        "title": "XGBoost MFCC+FFT + H4 Freq-MixStyle",
        "variant_name": "xgb_mfcc_fft_h4_freq_mixstyle",
        "use_spectral_gate": False,
        "use_noise_aug": False,
        "use_freq_mixstyle": True,
    },
    {
        "filename": "audio_ml_xgboost_mfcc_fft_h2_h3_gate_noise_aug.ipynb",
        "title": "XGBoost MFCC+FFT + H2 Spectral Gate + H3 Noise Augmentation",
        "variant_name": "xgb_mfcc_fft_h2_h3_gate_noise_aug",
        "use_spectral_gate": True,
        "use_noise_aug": True,
        "use_freq_mixstyle": False,
    },
    {
        "filename": "audio_ml_xgboost_mfcc_fft_h2_h4_gate_mixstyle.ipynb",
        "title": "XGBoost MFCC+FFT + H2 Spectral Gate + H4 Freq-MixStyle",
        "variant_name": "xgb_mfcc_fft_h2_h4_gate_mixstyle",
        "use_spectral_gate": True,
        "use_noise_aug": False,
        "use_freq_mixstyle": True,
    },
    {
        "filename": "audio_ml_xgboost_mfcc_fft_h3_h4_noise_aug_mixstyle.ipynb",
        "title": "XGBoost MFCC+FFT + H3 Noise Augmentation + H4 Freq-MixStyle",
        "variant_name": "xgb_mfcc_fft_h3_h4_noise_aug_mixstyle",
        "use_spectral_gate": False,
        "use_noise_aug": True,
        "use_freq_mixstyle": True,
    },
    {
        "filename": "audio_ml_xgboost_mfcc_fft_h2_h3_h4_all.ipynb",
        "title": "XGBoost MFCC+FFT + H2 Spectral Gate + H3 Noise Augmentation + H4 Freq-MixStyle",
        "variant_name": "xgb_mfcc_fft_h2_h3_h4_all",
        "use_spectral_gate": True,
        "use_noise_aug": True,
        "use_freq_mixstyle": True,
    },
    {
        "filename": "audio_ml_xgboost_mfcc_fft_h2_domain_gate.ipynb",
        "title": "XGBoost MFCC+FFT + H2 Domain-Aware Spectral Gate",
        "variant_name": "xgb_mfcc_fft_h2_domain_gate",
        "use_spectral_gate": True,
        "spectral_gate_mode": "domain",
        "use_noise_aug": False,
        "use_freq_mixstyle": False,
    },
    {
        "filename": "audio_ml_xgboost_mfcc_fft_h2_domain_gate_h3_noise_aug.ipynb",
        "title": "XGBoost MFCC+FFT + H2 Domain-Aware Spectral Gate + H3 Noise Augmentation",
        "variant_name": "xgb_mfcc_fft_h2_domain_gate_h3_noise_aug",
        "use_spectral_gate": True,
        "spectral_gate_mode": "domain",
        "use_noise_aug": True,
        "use_freq_mixstyle": False,
    },
    {
        "filename": "audio_ml_xgboost_mfcc_fft_h2_domain_gate_h4_mixstyle.ipynb",
        "title": "XGBoost MFCC+FFT + H2 Domain-Aware Spectral Gate + H4 Freq-MixStyle",
        "variant_name": "xgb_mfcc_fft_h2_domain_gate_h4_mixstyle",
        "use_spectral_gate": True,
        "spectral_gate_mode": "domain",
        "use_noise_aug": False,
        "use_freq_mixstyle": True,
    },
    {
        "filename": "audio_ml_xgboost_mfcc_fft_h2_domain_gate_h3_h4_all.ipynb",
        "title": "XGBoost MFCC+FFT + H2 Domain-Aware Spectral Gate + H3 Noise Augmentation + H4 Freq-MixStyle",
        "variant_name": "xgb_mfcc_fft_h2_domain_gate_h3_h4_all",
        "use_spectral_gate": True,
        "spectral_gate_mode": "domain",
        "use_noise_aug": True,
        "use_freq_mixstyle": True,
    },
]


COMMON_CODE = r'''# Dataset root - edit this first if your Kaggle input path changes.
DATASET_ROOT = "/kaggle/input/datasets/anhduy54/visual-audio/raw_dataset"

# Optional smoke-test cap. Use 0 for the full Kaggle run.
MAX_SAMPLES_PER_CLASS = 0

# Output location. Kaggle writes to /kaggle/working.
from pathlib import Path

OUTPUT_DIR = (
    Path("/kaggle/working/outputs/domain_robustness")
    if Path("/kaggle/working").exists()
    else Path("outputs/domain_robustness")
)

# Variant toggles generated for this notebook.
VARIANT_NAME = "__VARIANT_NAME__"
USE_SPECTRAL_GATE = __USE_SPECTRAL_GATE__
SPECTRAL_GATE_MODE = "__SPECTRAL_GATE_MODE__"
USE_NOISE_AUG = __USE_NOISE_AUG__
USE_FREQ_MIXSTYLE = __USE_FREQ_MIXSTYLE__


import json
import math
import os
import random
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = ""


LABELS = ("ambient", "leaf", "trunk", "twig")
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
TRAIN_SPLIT = "audio_visual_dataset_default"
ROBOT_SPLIT = "audio_visual_dataset_robo_default"


@dataclass
class AudioConfig:
    target_sample_rate: int = 16000
    audio_window_sec: float = 0.8
    n_mels: int = 128
    n_mfcc: int = 40
    n_fft: int = 1024
    hop_length: int = 256
    mfcc_width: int = 32
    fft_bins: int = 512
    train_crop: str = "random"
    eval_crop: str = "energy"

    @property
    def window_samples(self) -> int:
        return int(round(self.target_sample_rate * self.audio_window_sec))


@dataclass
class VariantConfig:
    variant_name: str
    use_spectral_gate: bool = False
    spectral_gate_mode: str = "robot"  # "robot" keeps old behavior; "domain" routes train/test gates by split.
    use_noise_aug: bool = False
    use_freq_mixstyle: bool = False
    seed: int = 42
    profile_fraction: float = 0.20
    snr_db_values: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0)
    noise_gate_percentile: float = 60.0
    noise_gate_margin_db: float = 1.0
    noise_gate_min_gain: float = 0.20
    mixstyle_alpha: float = 0.3
    specaugment_freq_masks: int = 2
    specaugment_time_masks: int = 2
    specaugment_max_freq_width: int = 8
    specaugment_max_time_width: int = 6
    gaussian_noise_snr_range: tuple[float, float] = (10.0, 20.0)
    pink_noise_snr_range: tuple[float, float] = (10.0, 20.0)


AUDIO_CFG = AudioConfig()
VARIANT = VariantConfig(
    variant_name=VARIANT_NAME,
    use_spectral_gate=USE_SPECTRAL_GATE,
    spectral_gate_mode=SPECTRAL_GATE_MODE,
    use_noise_aug=USE_NOISE_AUG,
    use_freq_mixstyle=USE_FREQ_MIXSTYLE,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def find_data_root(data_root: str | Path | None = None) -> Path:
    candidates = []
    if data_root is not None:
        root = Path(data_root)
        candidates.extend(
            [
                root,
                root / "raw_dataset",
                root / "prepared_data",
                root / "dataset",
            ]
        )
    candidates.extend(
        [
            Path("/kaggle/input/datasets/anhduy54/visual-audio/raw_dataset"),
            Path("/kaggle/input/visual-audio/raw_dataset"),
            Path("/kaggle/input/raw_dataset"),
            Path("dataset"),
            Path("."),
        ]
    )
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
            image_path = split_dir / item["image_file"]
            exists = audio_path.exists()
            if skip_missing_files and not exists:
                skipped += 1
                continue
            label = item["category"]
            rows.append(
                {
                    "split_name": split_name,
                    "audio_path": str(audio_path),
                    "image_path": str(image_path),
                    "label": label,
                    "label_id": LABEL_TO_ID[label],
                    "audio_file": item["audio_file"],
                    "image_file": item["image_file"],
                    "files_exist": exists,
                }
            )
    index = pd.DataFrame(rows)
    index.attrs["skipped_missing_files"] = skipped
    return index


def sample_stratified_frame(frame: pd.DataFrame, max_samples_per_class: int, seed: int) -> pd.DataFrame:
    if max_samples_per_class <= 0 or frame.empty:
        return frame.reset_index(drop=True)
    parts = []
    for _, part in frame.groupby("label_id", sort=True):
        take = min(max_samples_per_class, len(part))
        parts.append(part.sample(n=take, random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_robot_profile(
    index: pd.DataFrame,
    variant: VariantConfig,
    max_samples_per_class: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train_df = index[index["split_name"] == TRAIN_SPLIT].reset_index(drop=True)
    robot_df = index[index["split_name"] == ROBOT_SPLIT].reset_index(drop=True)
    ambient_robot = robot_df[robot_df["label"] == "ambient"].reset_index(drop=True)
    if ambient_robot.empty:
        raise ValueError("No ambient robot rows are available for the reserved profile split.")

    n_profile = max(1, int(math.ceil(len(ambient_robot) * variant.profile_fraction)))
    n_profile = min(n_profile, len(ambient_robot))
    profile_df = ambient_robot.sample(n=n_profile, random_state=variant.seed).reset_index(drop=True)
    profile_paths = set(profile_df["audio_path"].tolist())
    eval_df = robot_df[~robot_df["audio_path"].isin(profile_paths)].reset_index(drop=True)

    train_df = sample_stratified_frame(train_df, max_samples_per_class, variant.seed)
    eval_df = sample_stratified_frame(eval_df, max_samples_per_class, variant.seed)
    train_profile_df = train_df[train_df["label"] == "ambient"].reset_index(drop=True)
    if train_profile_df.empty:
        raise ValueError("No ambient train/no-robo rows are available for the train-domain spectral gate.")

    overlap = sorted(profile_paths.intersection(set(eval_df["audio_path"].tolist())))
    split_info = {
        "train_rows": int(len(train_df)),
        "train_profile_rows": int(len(train_profile_df)),
        "robot_profile_rows": int(len(profile_df)),
        "robot_eval_rows": int(len(eval_df)),
        "profile_fraction_requested": float(variant.profile_fraction),
        "profile_eval_overlap_count": int(len(overlap)),
        "profile_eval_overlap": overlap,
        "train_profile_label_counts": train_profile_df["label"].value_counts().to_dict(),
        "robot_profile_label_counts": profile_df["label"].value_counts().to_dict(),
        "robot_eval_label_counts": eval_df["label"].value_counts().to_dict(),
        "train_label_counts": train_df["label"].value_counts().to_dict(),
    }
    if overlap:
        raise RuntimeError(f"Reserved profile rows leaked into eval: {overlap[:3]}")
    return train_df, train_profile_df, profile_df, eval_df, split_info


def _load_wav_stdlib(path: str | Path) -> tuple[np.ndarray, int]:
    path = str(path)
    with wave.open(path, "rb") as wav_file:
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
        up = target_sample_rate // gcd
        down = sample_rate // gcd
        return resample_poly(waveform, up, down).astype(np.float32)
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


def load_processed_waveform(
    path: str | Path,
    audio_cfg: AudioConfig,
    crop_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    waveform, sample_rate = _load_wav_stdlib(path)
    waveform = resample_waveform(waveform, sample_rate, audio_cfg.target_sample_rate)
    waveform = normalize_waveform(waveform)
    return crop_or_pad(waveform, audio_cfg.window_samples, crop_mode, rng)


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


def istft_complex(spec: np.ndarray, n_fft: int, hop_length: int, length: int) -> np.ndarray:
    n_frames = spec.shape[1]
    out_len = n_fft + hop_length * (n_frames - 1)
    frames = np.fft.irfft(spec.T, n=n_fft, axis=1).astype(np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    output = np.zeros(out_len, dtype=np.float32)
    weight = np.zeros(out_len, dtype=np.float32)
    for idx in range(n_frames):
        start = idx * hop_length
        output[start : start + n_fft] += frames[idx] * window
        weight[start : start + n_fft] += window**2
    valid = weight > 1e-8
    output[valid] /= weight[valid]
    return output[:length].astype(np.float32)


def estimate_noise_threshold(
    profile_df: pd.DataFrame,
    audio_cfg: AudioConfig,
    variant: VariantConfig,
    profile_name: str = "profile",
) -> np.ndarray:
    rng = np.random.default_rng(variant.seed + 1001)
    magnitudes = []
    for row in tqdm(profile_df.to_dict("records"), desc=f"estimating {profile_name} ambient threshold", leave=False):
        waveform = load_processed_waveform(row["audio_path"], audio_cfg, "center", rng)
        spec = stft_complex(waveform, audio_cfg.n_fft, audio_cfg.hop_length)
        magnitudes.append(np.abs(spec).astype(np.float32))
    if not magnitudes:
        raise ValueError("Cannot estimate spectral gate threshold without profile audio.")
    stacked = np.concatenate(magnitudes, axis=1)
    threshold = np.percentile(stacked, variant.noise_gate_percentile, axis=1).astype(np.float32)
    threshold *= float(10.0 ** (variant.noise_gate_margin_db / 20.0))
    return np.maximum(threshold, 1e-8)


def threshold_stats(threshold: np.ndarray | None) -> dict:
    if threshold is None:
        return {}
    values = np.asarray(threshold, dtype=np.float32)
    return {
        "bins": int(values.shape[0]),
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def apply_profile_spectral_gate(
    waveform: np.ndarray,
    threshold: np.ndarray | None,
    audio_cfg: AudioConfig,
    variant: VariantConfig,
) -> np.ndarray:
    if threshold is None:
        return waveform
    spec = stft_complex(waveform, audio_cfg.n_fft, audio_cfg.hop_length)
    magnitude = np.abs(spec).astype(np.float32)
    phase = spec / np.maximum(magnitude, 1e-8)
    threshold_2d = threshold[:, None]
    ratio = magnitude / np.maximum(threshold_2d, 1e-8)
    soft_mask = np.clip((ratio - 0.5) / 0.5, 0.0, 1.0)
    gain = variant.noise_gate_min_gain + (1.0 - variant.noise_gate_min_gain) * soft_mask
    gated = magnitude * gain * phase
    return normalize_waveform(istft_complex(gated, audio_cfg.n_fft, audio_cfg.hop_length, len(waveform)))


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    min_mel = hz_to_mel(0.0)
    max_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
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


MEL_FILTER = mel_filterbank(AUDIO_CFG.target_sample_rate, AUDIO_CFG.n_fft, AUDIO_CFG.n_mels)
DCT_BASIS = dct_matrix(AUDIO_CFG.n_mels, AUDIO_CFG.n_mfcc)


def resize_matrix(mat: np.ndarray, height: int, width: int) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"resize_matrix expects 2D input, got {mat.shape}")
    src_h, src_w = mat.shape
    x_old = np.linspace(0.0, 1.0, src_w)
    x_new = np.linspace(0.0, 1.0, width)
    tmp = np.vstack([np.interp(x_new, x_old, row) for row in mat])
    y_old = np.linspace(0.0, 1.0, src_h)
    y_new = np.linspace(0.0, 1.0, height)
    out = np.vstack([np.interp(y_new, y_old, tmp[:, col]) for col in range(width)]).T
    return out.astype(np.float32)


def resize_vector(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) == length:
        return values
    old_x = np.linspace(0.0, 1.0, len(values))
    new_x = np.linspace(0.0, 1.0, length)
    return np.interp(new_x, old_x, values).astype(np.float32)


def mfcc_fft_map_from_waveform(waveform: np.ndarray, audio_cfg: AudioConfig) -> np.ndarray:
    spec = stft_complex(waveform, audio_cfg.n_fft, audio_cfg.hop_length)
    power = np.maximum(np.abs(spec) ** 2, 1e-10).astype(np.float32)
    mel_power = np.maximum(MEL_FILTER @ power, 1e-10)
    mel_db = 10.0 * np.log10(mel_power)
    mel_db = np.maximum(mel_db, mel_db.max() - 80.0)
    mfcc = DCT_BASIS @ mel_db
    mfcc = resize_matrix(mfcc, audio_cfg.n_mfcc, audio_cfg.mfcc_width)

    fft_mag = np.log1p(np.abs(np.fft.rfft(waveform.astype(np.float32))))
    fft = resize_vector(fft_mag, audio_cfg.fft_bins).reshape(16, 32)
    return np.concatenate([mfcc, fft], axis=0).astype(np.float32)


def feature_map_from_path(
    path: str | Path,
    crop_mode: str,
    audio_cfg: AudioConfig,
    variant: VariantConfig,
    threshold: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    waveform = load_processed_waveform(path, audio_cfg, crop_mode, rng)
    if variant.use_spectral_gate:
        waveform = apply_profile_spectral_gate(waveform, threshold, audio_cfg, variant)
    return mfcc_fft_map_from_waveform(waveform, audio_cfg)


def waveform_from_path(
    path: str | Path,
    crop_mode: str,
    audio_cfg: AudioConfig,
    variant: VariantConfig,
    threshold: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    waveform = load_processed_waveform(path, audio_cfg, crop_mode, rng)
    if variant.use_spectral_gate:
        waveform = apply_profile_spectral_gate(waveform, threshold, audio_cfg, variant)
    return waveform


def match_noise_length(noise: np.ndarray, target_len: int, rng: np.random.Generator) -> np.ndarray:
    noise = np.asarray(noise, dtype=np.float32).reshape(-1)
    if len(noise) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(noise) < target_len:
        repeats = int(math.ceil(target_len / len(noise)))
        noise = np.tile(noise, repeats)
    max_start = max(0, len(noise) - target_len)
    start = int(rng.integers(0, max_start + 1)) if max_start else 0
    return noise[start : start + target_len].astype(np.float32)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2) + 1e-12))


def inject_noise_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal_rms = rms(signal)
    noise_rms = rms(noise)
    scale = signal_rms / (max(noise_rms, 1e-8) * (10.0 ** (snr_db / 20.0)))
    return normalize_waveform(signal + noise * scale)


def gaussian_noise_like(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=len(signal)).astype(np.float32)
    return noise / max(rms(noise), 1e-8)


def pink_noise_like(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size=len(signal)).astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(white), d=1.0 / AUDIO_CFG.target_sample_rate)
    weights = np.ones_like(freqs)
    weights[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * weights, n=len(white)).astype(np.float32)
    return pink / max(rms(pink), 1e-8)


def specaugment_map(feature_map: np.ndarray, variant: VariantConfig, rng: np.random.Generator) -> np.ndarray:
    out = np.array(feature_map, dtype=np.float32, copy=True)
    fill_value = float(out.mean())
    freq_bins, time_steps = out.shape
    for _ in range(variant.specaugment_freq_masks):
        width = int(rng.integers(1, variant.specaugment_max_freq_width + 1))
        start = int(rng.integers(0, max(1, freq_bins - width + 1)))
        out[start : start + width, :] = fill_value
    for _ in range(variant.specaugment_time_masks):
        width = int(rng.integers(1, variant.specaugment_max_time_width + 1))
        start = int(rng.integers(0, max(1, time_steps - width + 1)))
        out[:, start : start + width] = fill_value
    return out


def freq_mixstyle_map(
    content_map: np.ndarray,
    style_map: np.ndarray,
    variant: VariantConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    content = np.asarray(content_map, dtype=np.float32)
    style = np.asarray(style_map, dtype=np.float32)
    content_mu = content.mean(axis=1, keepdims=True)
    content_sigma = content.std(axis=1, keepdims=True) + 1e-6
    style_mu = style.mean(axis=1, keepdims=True)
    style_sigma = style.std(axis=1, keepdims=True) + 1e-6
    lam = float(rng.beta(variant.mixstyle_alpha, variant.mixstyle_alpha))
    mixed_mu = lam * content_mu + (1.0 - lam) * style_mu
    mixed_sigma = lam * content_sigma + (1.0 - lam) * style_sigma
    normalized = (content - content_mu) / content_sigma
    return (normalized * mixed_sigma + mixed_mu).astype(np.float32)


def prepare_noise_pool(
    profile_df: pd.DataFrame,
    audio_cfg: AudioConfig,
    variant: VariantConfig,
    threshold: np.ndarray | None,
) -> list[np.ndarray]:
    rng = np.random.default_rng(variant.seed + 2001)
    noise_pool = []
    for row in tqdm(profile_df.to_dict("records"), desc="loading robot ambient noise pool", leave=False):
        noise_pool.append(waveform_from_path(row["audio_path"], "center", audio_cfg, variant, threshold, rng))
    return noise_pool


def prepare_style_maps(
    profile_df: pd.DataFrame,
    audio_cfg: AudioConfig,
    variant: VariantConfig,
    threshold: np.ndarray | None,
) -> list[np.ndarray]:
    rng = np.random.default_rng(variant.seed + 3001)
    style_maps = []
    for row in tqdm(profile_df.to_dict("records"), desc="loading robot ambient style maps", leave=False):
        style_maps.append(feature_map_from_path(row["audio_path"], "center", audio_cfg, variant, threshold, rng))
    return style_maps


def build_variant_feature_matrix(
    frame: pd.DataFrame,
    audio_cfg: AudioConfig,
    variant: VariantConfig,
    thresholds: dict[str, np.ndarray | None] | np.ndarray | None,
    crop_mode: str,
    augment: bool,
    noise_pool: list[np.ndarray] | None = None,
    style_maps: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(variant.seed + (5001 if augment else 6001))
    features = []
    labels = []
    sources = []
    noise_pool = noise_pool or []
    style_maps = style_maps or []

    records = frame.to_dict("records")
    desc = f"{variant.variant_name}/{'train' if augment else 'eval'}"
    for row_idx, row in enumerate(tqdm(records, desc=desc, leave=False)):
        if isinstance(thresholds, dict):
            if variant.spectral_gate_mode == "domain":
                threshold = thresholds.get(row.get("split_name"), thresholds.get("robot"))
            else:
                threshold = thresholds.get("robot")
        else:
            threshold = thresholds
        waveform = waveform_from_path(row["audio_path"], crop_mode, audio_cfg, variant, threshold, rng)
        base_map = mfcc_fft_map_from_waveform(waveform, audio_cfg)
        label = int(row["label_id"])

        features.append(base_map.reshape(-1))
        labels.append(label)
        sources.append("original")

        if not augment:
            continue

        if variant.use_noise_aug and noise_pool:
            for snr_db in variant.snr_db_values:
                noise = match_noise_length(noise_pool[int(rng.integers(0, len(noise_pool)))], len(waveform), rng)
                noisy = inject_noise_at_snr(waveform, noise, snr_db)
                features.append(mfcc_fft_map_from_waveform(noisy, audio_cfg).reshape(-1))
                labels.append(label)
                sources.append(f"robot_noise_{snr_db:g}db")

            gaussian_snr = float(rng.uniform(*variant.gaussian_noise_snr_range))
            gaussian = gaussian_noise_like(waveform, rng)
            features.append(mfcc_fft_map_from_waveform(inject_noise_at_snr(waveform, gaussian, gaussian_snr), audio_cfg).reshape(-1))
            labels.append(label)
            sources.append("gaussian_noise")

            pink_snr = float(rng.uniform(*variant.pink_noise_snr_range))
            pink = pink_noise_like(waveform, rng)
            features.append(mfcc_fft_map_from_waveform(inject_noise_at_snr(waveform, pink, pink_snr), audio_cfg).reshape(-1))
            labels.append(label)
            sources.append("pink_noise")

            features.append(specaugment_map(base_map, variant, rng).reshape(-1))
            labels.append(label)
            sources.append("specaugment")

        if variant.use_freq_mixstyle and style_maps:
            style_map = style_maps[int(rng.integers(0, len(style_maps)))]
            features.append(freq_mixstyle_map(base_map, style_map, variant, rng).reshape(-1))
            labels.append(label)
            sources.append("freq_mixstyle")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return x, y, sources


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    binary_true = (np.asarray(y_true) != LABEL_TO_ID["ambient"]).astype(np.int64)
    binary_pred = (np.asarray(y_pred) != LABEL_TO_ID["ambient"]).astype(np.int64)
    binary_contact_f1 = f1_score(binary_true, binary_pred, zero_division=0)
    per_class = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "paper_precision": float(precision),
        "paper_recall": float(recall),
        "paper_f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "binary_contact_f1": float(binary_contact_f1),
        "per_class_precision": {LABELS[i]: float(per_class[0][i]) for i in range(len(LABELS))},
        "per_class_recall": {LABELS[i]: float(per_class[1][i]) for i in range(len(LABELS))},
        "per_class_f1": {LABELS[i]: float(per_class[2][i]) for i in range(len(LABELS))},
    }


def run_xgboost_mfcc_fft_variant(
    data_root: str | Path,
    output_dir: str | Path,
    audio_cfg: AudioConfig,
    variant: VariantConfig,
    max_samples_per_class: int = 0,
) -> dict:
    from xgboost import XGBClassifier

    set_seed(variant.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = find_data_root(data_root)

    index = build_index(data_root, skip_missing_files=True)
    train_df, train_profile_df, profile_df, eval_df, split_info = split_robot_profile(index, variant, max_samples_per_class)
    print("DATA_ROOT =", data_root)
    print("OUTPUT_DIR =", output_dir)
    print("variant =", variant.variant_name)
    print("toggles =", {"spectral_gate": variant.use_spectral_gate, "spectral_gate_mode": variant.spectral_gate_mode, "noise_aug": variant.use_noise_aug, "freq_mixstyle": variant.use_freq_mixstyle})
    print("split_info =", split_info)

    train_threshold = None
    robot_threshold = None
    if variant.use_spectral_gate:
        robot_threshold = estimate_noise_threshold(profile_df, audio_cfg, variant, "robot")
        if variant.spectral_gate_mode == "domain":
            train_threshold = estimate_noise_threshold(train_profile_df, audio_cfg, variant, "train/no-robo")
        else:
            train_threshold = robot_threshold
    thresholds = {
        TRAIN_SPLIT: train_threshold,
        ROBOT_SPLIT: robot_threshold,
        "train": train_threshold,
        "robot": robot_threshold,
    }

    noise_pool = prepare_noise_pool(profile_df, audio_cfg, variant, robot_threshold) if variant.use_noise_aug else []
    style_maps = prepare_style_maps(profile_df, audio_cfg, variant, robot_threshold) if variant.use_freq_mixstyle else []

    x_train, y_train, train_sources = build_variant_feature_matrix(
        train_df,
        audio_cfg,
        variant,
        thresholds,
        audio_cfg.train_crop,
        augment=True,
        noise_pool=noise_pool,
        style_maps=style_maps,
    )
    x_eval, y_eval, eval_sources = build_variant_feature_matrix(
        eval_df,
        audio_cfg,
        variant,
        thresholds,
        audio_cfg.eval_crop,
        augment=False,
        noise_pool=None,
        style_maps=None,
    )

    print("x_train shape =", x_train.shape, "x_eval shape =", x_eval.shape)
    print("train source counts =", pd.Series(train_sources).value_counts().to_dict())

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(LABELS),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=variant.seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_eval)
    metrics = classification_metrics(y_eval, pred)

    result = {
        "group": "handcrafted_domain_robustness",
        "mode": variant.variant_name,
        "input": "mfcc_fft",
        "feature": "mfcc_fft",
        "encoder": "handcrafted",
        "fusion": "none",
        "classifier": "xgboost",
        "pretrained": False,
        "frozen": False,
        "cpu_only": True,
        "variant": asdict(variant),
        "audio_config": asdict(audio_cfg),
        "split_info": split_info,
        "threshold_stats": {
            "train": threshold_stats(train_threshold),
            "robot": threshold_stats(robot_threshold),
        },
        "feature_shapes": {
            "x_train": list(x_train.shape),
            "x_eval": list(x_eval.shape),
        },
        "train_source_counts": pd.Series(train_sources).value_counts().to_dict(),
        "eval_source_counts": pd.Series(eval_sources).value_counts().to_dict(),
        "best_val_metrics": metrics,
    }

    result_json = output_dir / f"{variant.variant_name}_results.json"
    split_json = output_dir / f"{variant.variant_name}_split_check.json"
    result_csv = output_dir / f"{variant.variant_name}_results.csv"

    with result_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with split_json.open("w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    flat_row = {
        "group": result["group"],
        "mode": result["mode"],
        "input": result["input"],
        "feature": result["feature"],
        "encoder": result["encoder"],
        "fusion": result["fusion"],
        "classifier": result["classifier"],
        "pretrained": result["pretrained"],
        "frozen": result["frozen"],
        "cpu_only": result["cpu_only"],
        "use_spectral_gate": variant.use_spectral_gate,
        "spectral_gate_mode": variant.spectral_gate_mode,
        "use_noise_aug": variant.use_noise_aug,
        "use_freq_mixstyle": variant.use_freq_mixstyle,
        "train_rows": split_info["train_rows"],
        "train_profile_rows": split_info["train_profile_rows"],
        "robot_profile_rows": split_info["robot_profile_rows"],
        "robot_eval_rows": split_info["robot_eval_rows"],
        "profile_eval_overlap_count": split_info["profile_eval_overlap_count"],
        "x_train_rows": x_train.shape[0],
        "x_train_dim": x_train.shape[1],
        "x_eval_rows": x_eval.shape[0],
        "x_eval_dim": x_eval.shape[1],
        **metrics,
    }
    for key, value in list(flat_row.items()):
        if isinstance(value, dict):
            flat_row[key] = json.dumps(value, sort_keys=True)
    pd.DataFrame([flat_row]).to_csv(result_csv, index=False)

    print("metrics =", metrics)
    print("wrote", result_csv)
    print("wrote", result_json)
    print("wrote", split_json)
    return result


result = run_xgboost_mfcc_fft_variant(
    data_root=DATASET_ROOT,
    output_dir=OUTPUT_DIR,
    audio_cfg=AUDIO_CFG,
    variant=VARIANT,
    max_samples_per_class=MAX_SAMPLES_PER_CLASS,
)
pd.DataFrame([{
    "mode": result["mode"],
    "paper_f1": result["best_val_metrics"]["paper_f1"],
    "macro_f1": result["best_val_metrics"]["macro_f1"],
    "binary_contact_f1": result["best_val_metrics"]["binary_contact_f1"],
    "accuracy": result["best_val_metrics"]["accuracy"],
    "profile_eval_overlap_count": result["split_info"]["profile_eval_overlap_count"],
}])
'''


def make_cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line for line in source.splitlines(keepends=True)],
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def make_notebook(variant: dict) -> dict:
    markdown = f"""# {variant['title']}

Self-contained Kaggle CPU notebook for `XGBoost + MFCC+FFT`.

Protocol:
- Train on `audio_visual_dataset_default`.
- Reserve 20% of robot `ambient` rows as noise/style profile data.
- Evaluate on `audio_visual_dataset_robo_default` after removing reserved profile rows.
- Save split leakage checks and result CSV/JSON under `OUTPUT_DIR`.
"""
    code = (
        COMMON_CODE.replace("__VARIANT_NAME__", variant["variant_name"])
        .replace("__USE_SPECTRAL_GATE__", str(variant["use_spectral_gate"]))
        .replace("__SPECTRAL_GATE_MODE__", variant.get("spectral_gate_mode", "robot"))
        .replace("__USE_NOISE_AUG__", str(variant["use_noise_aug"]))
        .replace("__USE_FREQ_MIXSTYLE__", str(variant["use_freq_mixstyle"]))
    )
    return {
        "cells": [make_cell("markdown", markdown), make_cell("code", code)],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for variant in VARIANTS:
        path = NOTEBOOK_DIR / variant["filename"]
        notebook = make_notebook(variant)
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
