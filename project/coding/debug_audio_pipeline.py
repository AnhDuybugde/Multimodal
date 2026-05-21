from __future__ import annotations

import argparse
import math
import random
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .audio import AudioPipeline
    from .config import DataConfig, LABELS
    from .data import build_index, make_train_val_split
except ImportError:
    from audio import AudioPipeline
    from config import DataConfig, LABELS
    from data import build_index, make_train_val_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug the audio path before training AST/CLAP/audio-only models."
    )
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--sample-per-label", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--baseline", action="store_true", help="Run simple audio-feature baselines.")
    parser.add_argument(
        "--baseline-max-samples",
        type=int,
        default=4000,
        help="Cap samples for quick baselines. Use 0 to run on all valid samples.",
    )
    parser.add_argument("--processor-check", action="store_true", help="Check AST/CLAP processor outputs.")
    parser.add_argument("--ast-model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--clap-model-name", default="laion/clap-htsat-unfused")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    index = build_index(args.data_root, skip_missing_files=True)
    if index.empty:
        raise SystemExit(f"No valid samples found under {args.data_root}")

    print_header("Index")
    print(f"valid_samples={len(index)} skipped_missing={index.attrs.get('skipped_missing_files', 0)}")
    print(index["split_name"].value_counts().to_string())
    print(index["label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_string())
    print_majority_baseline(index)

    sampled = sample_frame(index, args.sample_per_label, rng)
    print_header("Raw WAV Stats")
    raw_rows = add_raw_stats(sampled)
    print_group_summary(raw_rows, ["raw_sr", "raw_duration", "raw_rms", "raw_peak", "raw_zcr"])

    print_header("Processed Waveform Stats")
    for window_sec in (0.8, 1.0):
        for crop in ("center", "energy"):
            processed = add_processed_stats(
                raw_rows,
                target_sample_rate=args.target_sample_rate,
                window_sec=window_sec,
                crop=crop,
            )
            print(f"\nwindow_sec={window_sec} crop={crop}")
            print_group_summary(
                processed,
                ["proc_len", "proc_rms", "proc_peak", "proc_zero_frac", "proc_center_edge_ratio"],
            )

    print_header("Crop Agreement")
    print_crop_agreement(raw_rows, args.target_sample_rate)

    print_header("Mini Batch Sanity")
    print_minibatch_sanity(index, args.target_sample_rate)

    if args.baseline:
        print_header("Simple Audio Baselines")
        run_simple_baselines(index, args.val_size, args.seed, args.baseline_max_samples, args.target_sample_rate)

    if args.processor_check:
        print_header("AST/CLAP Processor Check")
        run_processor_check(raw_rows, args)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_majority_baseline(index) -> None:
    counts = index["label"].value_counts().reindex(LABELS).fillna(0).astype(int)
    majority = counts.idxmax()
    support = int(counts[majority])
    total = int(counts.sum())
    majority_f1 = 2 * support / (total + support)
    macro_f1 = majority_f1 / len(LABELS)
    print(f"all-{majority} macro_f1_baseline={macro_f1:.6f}")


def sample_frame(index, per_label: int, rng: random.Random):
    parts = []
    for label in LABELS:
        rows = index[index["label"] == label]
        if rows.empty:
            continue
        take = min(per_label, len(rows))
        positions = rng.sample(range(len(rows)), take)
        parts.append(rows.iloc[positions])
    return pd.DataFrame() if not parts else pd.concat(parts, ignore_index=True)


def read_wav(path: str | Path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / float(2**15)
    elif sample_width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 255.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes for {path}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sample_rate, channels, sample_width


def raw_features(path: str | Path):
    data, sample_rate, channels, sample_width = read_wav(path)
    abs_data = np.abs(data)
    rms = float(np.sqrt(np.mean(data * data)))
    peak = float(abs_data.max(initial=0.0))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(data))).astype(np.float32))) if len(data) > 1 else 0.0
    clipping_frac = float(np.mean(abs_data > 0.999))
    return {
        "raw_sr": sample_rate,
        "raw_channels": channels,
        "raw_width": sample_width,
        "raw_duration": len(data) / sample_rate,
        "raw_rms": rms,
        "raw_peak": peak,
        "raw_zcr": zcr,
        "raw_clipping_frac": clipping_frac,
    }


def add_raw_stats(frame):
    rows = []
    for row in frame.to_dict("records"):
        item = dict(row)
        item.update(raw_features(row["audio_path"]))
        rows.append(item)
    import pandas as pd

    return pd.DataFrame(rows)


def add_processed_stats(frame, target_sample_rate: int, window_sec: float, crop: str):
    audio = AudioPipeline(target_sample_rate=target_sample_rate, window_sec=window_sec)
    rows = []
    for row in frame.to_dict("records"):
        item = dict(row)
        waveform = audio.load_processed_waveform(row["audio_path"], crop).squeeze(0)
        values = waveform.detach().cpu().numpy().astype(np.float32)
        item.update(waveform_stats(values, prefix="proc_"))
        rows.append(item)
    import pandas as pd

    return pd.DataFrame(rows)


def waveform_stats(values: np.ndarray, prefix: str = ""):
    abs_values = np.abs(values)
    rms = float(np.sqrt(np.mean(values * values)))
    peak = float(abs_values.max(initial=0.0))
    zero_frac = float(np.mean(abs_values < 1e-8))
    center_edge_ratio = segment_energy_ratio(values)
    return {
        f"{prefix}len": int(values.shape[-1]),
        f"{prefix}rms": rms,
        f"{prefix}peak": peak,
        f"{prefix}zero_frac": zero_frac,
        f"{prefix}center_edge_ratio": center_edge_ratio,
    }


def segment_energy_ratio(values: np.ndarray) -> float:
    if values.size < 10:
        return 0.0
    edge = max(1, int(values.size * 0.1))
    center_start = int(values.size * 0.1)
    center_end = int(values.size * 0.9)
    edge_values = np.concatenate([values[:edge], values[-edge:]])
    center_values = values[center_start:center_end]
    edge_energy = float(np.mean(edge_values * edge_values))
    center_energy = float(np.mean(center_values * center_values))
    return center_energy / (edge_energy + 1e-12)


def print_group_summary(frame, columns):
    available = [col for col in columns if col in frame.columns]
    summary = frame.groupby("label")[available].agg(["mean", "median", "min", "max"])
    print(summary.round(6).to_string())


def print_crop_agreement(frame, target_sample_rate: int):
    audio = AudioPipeline(target_sample_rate=target_sample_rate, window_sec=0.8)
    rows = []
    for row in frame.to_dict("records"):
        full = audio.load_waveform(row["audio_path"])
        center = audio.crop_or_pad(full, "center").squeeze(0).numpy()
        energy = audio.crop_or_pad(full, "energy").squeeze(0).numpy()
        full_np = full.squeeze(0).numpy()
        rows.append(
            {
                "label": row["label"],
                "center_kept_energy": kept_energy(full_np, center),
                "energy_kept_energy": kept_energy(full_np, energy),
                "center_rms": float(np.sqrt(np.mean(center * center))),
                "energy_rms": float(np.sqrt(np.mean(energy * energy))),
            }
        )
    import pandas as pd

    out = pd.DataFrame(rows)
    print(out.groupby("label").agg(["mean", "median", "min", "max"]).round(6).to_string())


def kept_energy(full: np.ndarray, crop: np.ndarray) -> float:
    return float(np.sum(crop * crop) / (np.sum(full * full) + 1e-12))


def print_minibatch_sanity(index, target_sample_rate: int) -> None:
    data_cfg = DataConfig(data_root=Path("."), target_sample_rate=target_sample_rate)
    train_df, val_df = make_train_val_split(index, val_size=0.2, seed=42)
    audio_train = AudioPipeline(data_cfg.target_sample_rate, data_cfg.audio_window_sec)
    audio_val = AudioPipeline(data_cfg.target_sample_rate, data_cfg.audio_window_sec)

    for name, frame, crop, audio in (
        ("train/random", train_df, data_cfg.train_crop, audio_train),
        ("val/center", val_df, data_cfg.eval_crop, audio_val),
        ("val/energy", val_df, "energy", audio_val),
    ):
        rows = []
        for row in frame.head(32).to_dict("records"):
            waveform = audio.load_processed_waveform(row["audio_path"], crop).squeeze(0).numpy()
            item = {"label": row["label"]}
            item.update(waveform_stats(waveform, prefix=""))
            rows.append(item)
        import pandas as pd

        batch = pd.DataFrame(rows)
        print(f"\n{name}")
        print(batch.groupby("label")[["rms", "peak", "zero_frac", "center_edge_ratio"]].agg(["mean", "median"]).round(6).to_string())


def run_simple_baselines(index, val_size: float, seed: int, max_samples: int, target_sample_rate: int) -> None:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        print(f"sklearn is not available, skipping baselines: {exc}")
        return

    if max_samples and len(index) > max_samples:
        sampled = sample_stratified(index, max_samples, seed)
        print(f"baseline_samples={len(sampled)} from total={len(index)}")
    else:
        sampled = index
        print(f"baseline_samples={len(sampled)}")

    train_df, val_df = make_train_val_split(sampled, val_size=val_size, seed=seed)
    feature_sets = {
        "raw_cheap": make_feature_matrix,
        "processed_center_0.8": lambda df: make_processed_feature_matrix(df, target_sample_rate, 0.8, "center"),
        "processed_energy_0.8": lambda df: make_processed_feature_matrix(df, target_sample_rate, 0.8, "energy"),
        "processed_full_1.0": lambda df: make_processed_feature_matrix(df, target_sample_rate, 1.0, "center"),
    }
    y_train = train_df["label_id"].to_numpy()
    y_val = val_df["label_id"].to_numpy()

    for name, builder in feature_sets.items():
        x_train = builder(train_df)
        x_val = builder(val_df)
        models = {
            "logreg": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                random_state=seed,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
        }
        print(f"\nfeature_set={name} x_train={x_train.shape} x_val={x_val.shape}")
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            print(f"{model_name} macro_f1={f1_score(y_val, pred, average='macro', zero_division=0):.6f}")
            print(confusion_matrix(y_val, pred, labels=list(range(len(LABELS)))))
            print(classification_report(y_val, pred, target_names=LABELS, zero_division=0))


def sample_stratified(index, max_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    counts = index["label"].value_counts().reindex(LABELS).fillna(0).astype(int)
    total = int(counts.sum())
    parts = []
    for label in LABELS:
        rows = index[index["label"] == label]
        if rows.empty:
            continue
        quota = max(1, int(round(max_samples * len(rows) / total)))
        take = min(quota, len(rows))
        choices = rng.choice(len(rows), size=take, replace=False)
        parts.append(rows.iloc[choices])
    return pd.concat(parts, ignore_index=True)


def make_feature_matrix(frame):
    features = []
    for row in frame.to_dict("records"):
        data, _, _, _ = read_wav(row["audio_path"])
        features.append(cheap_features(data))
    return np.asarray(features, dtype=np.float32)


def make_processed_feature_matrix(frame, target_sample_rate: int, window_sec: float, crop: str):
    audio = AudioPipeline(target_sample_rate=target_sample_rate, window_sec=window_sec)
    features = []
    for row in frame.to_dict("records"):
        waveform = audio.load_processed_waveform(row["audio_path"], crop).squeeze(0).numpy()
        features.append(cheap_features(waveform))
    return np.asarray(features, dtype=np.float32)


def cheap_features(values: np.ndarray):
    values = values.astype(np.float32)
    abs_values = np.abs(values)
    chunks = np.array_split(values, 8)
    chunk_rms = [math.sqrt(float(np.mean(chunk * chunk))) for chunk in chunks]
    chunk_peak = [float(np.max(np.abs(chunk))) for chunk in chunks]
    rms = math.sqrt(float(np.mean(values * values)))
    peak = float(abs_values.max(initial=0.0))
    mean_abs = float(np.mean(abs_values))
    std = float(np.std(values))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(values))).astype(np.float32))) if len(values) > 1 else 0.0
    ratio = segment_energy_ratio(values)
    return [
        rms,
        math.log10(rms + 1e-10),
        peak,
        math.log10(peak + 1e-10),
        mean_abs,
        std,
        zcr,
        ratio,
        *chunk_rms,
        *chunk_peak,
    ]


def run_processor_check(frame, args) -> None:
    try:
        from transformers import ASTFeatureExtractor, AutoProcessor
    except Exception as exc:
        print(f"transformers is not available, skipping processor check: {exc}")
        return

    audio = AudioPipeline(target_sample_rate=args.target_sample_rate, window_sec=0.8)
    samples = []
    labels = []
    for row in frame.head(12).to_dict("records"):
        waveform = audio.load_processed_waveform(row["audio_path"], "center").squeeze(0).numpy()
        samples.append(waveform.astype("float32"))
        labels.append(row["label"])

    checks = []
    ast = ASTFeatureExtractor.from_pretrained(args.ast_model_name)
    ast_rate = getattr(ast, "sampling_rate", args.target_sample_rate)
    ast_samples = adapt_sample_rate(samples, args.target_sample_rate, ast_rate)
    ast_inputs = ast(ast_samples, sampling_rate=ast_rate, return_tensors="pt", padding=True)
    checks.append(("ASTFeatureExtractor", ast_inputs))

    clap = AutoProcessor.from_pretrained(args.clap_model_name)
    clap_rate = getattr(getattr(clap, "feature_extractor", None), "sampling_rate", args.target_sample_rate)
    clap_samples = adapt_sample_rate(samples, args.target_sample_rate, clap_rate)
    clap_inputs = clap(audio=clap_samples, sampling_rate=clap_rate, return_tensors="pt", padding=True)
    checks.append(("CLAPProcessor", clap_inputs))

    print(f"labels={labels}")
    print(f"pipeline_rate={args.target_sample_rate} ast_rate={ast_rate} clap_rate={clap_rate}")
    for name, inputs in checks:
        print(f"\n{name}")
        for key, value in inputs.items():
            if torch.is_tensor(value):
                finite = torch.isfinite(value).float().mean().item()
                print(
                    f"{key}: shape={tuple(value.shape)} "
                    f"min={value.min().item():.6f} max={value.max().item():.6f} "
                    f"mean={value.float().mean().item():.6f} std={value.float().std().item():.6f} "
                    f"finite_frac={finite:.6f}"
                )


def adapt_sample_rate(samples, orig_rate: int, target_rate: int):
    if orig_rate == target_rate:
        return samples
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise ImportError("scipy is required for processor sample-rate checks.") from exc

    gcd = math.gcd(orig_rate, target_rate)
    up = target_rate // gcd
    down = orig_rate // gcd
    return [resample_poly(item, up=up, down=down).astype("float32") for item in samples]


if __name__ == "__main__":
    main()
