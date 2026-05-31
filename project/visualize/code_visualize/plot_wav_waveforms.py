"""Plot side-by-side WAV waveforms for no-robot and robot samples.

The script reads the released audio-visual CSV files, chooses a fixed number
of existing WAV samples per class, and exports comparison figures under
project/visualize/imgs.
"""

from __future__ import annotations

import argparse
import csv
import wave
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.io import wavfile
except ImportError:  # pragma: no cover - optional dependency fallback
    wavfile = None


CLASSES = ("ambient", "leaf", "twig", "trunk")
DOMAIN_LABELS = {
    "norobo": "No-robo",
    "robo": "Robo",
}


@dataclass(frozen=True)
class Sample:
    audio_path: Path
    image_path: Path
    category: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_dataset(dataset_dir: Path) -> list[Sample]:
    csv_path = dataset_dir / "dataset.csv"
    samples: list[Sample] = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = row["category"].strip()
            audio_path = dataset_dir / row["audio_file"]
            image_path = dataset_dir / row["image_file"]
            if category in CLASSES and audio_path.exists() and image_path.exists():
                samples.append(Sample(audio_path, image_path, category))

    return samples


def choose_samples(samples: list[Sample], category: str, count: int) -> list[Sample]:
    selected = [sample for sample in samples if sample.category == category]
    if len(selected) < count:
        raise ValueError(
            f"Not enough {category!r} samples: found {len(selected)}, need {count}."
        )
    return selected[:count]


def pcm_bytes_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        sign = (bytes_[:, 2] & 0x80) > 0
        padded = np.zeros((bytes_.shape[0], 4), dtype=np.uint8)
        padded[:, :3] = bytes_
        padded[sign, 3] = 0xFF
        return padded.view("<i4").reshape(-1).astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")


def read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    if wavfile is not None:
        sample_rate, data = wavfile.read(path)
        waveform = np.asarray(data)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = waveform.astype(np.float32)

        if np.issubdtype(np.asarray(data).dtype, np.integer):
            info = np.iinfo(np.asarray(data).dtype)
            scale = max(abs(info.min), abs(info.max))
            waveform = waveform / float(scale)
        return int(sample_rate), waveform

    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    waveform = pcm_bytes_to_float(frames, sample_width)
    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(axis=1)
    return sample_rate, waveform


def plot_waveform(ax: plt.Axes, sample: Sample, color: str) -> None:
    sample_rate, waveform = read_wav_mono(sample.audio_path)
    time = np.arange(waveform.shape[0], dtype=np.float32) / float(sample_rate)

    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        waveform = waveform / peak

    ax.plot(time, waveform, color=color, linewidth=0.75)
    ax.axhline(0, color="#2F3136", linewidth=0.45, alpha=0.35)
    ax.set_xlim(float(time[0]) if time.size else 0.0, float(time[-1]) if time.size else 1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, axis="x", color="#D8DEE9", linewidth=0.45, alpha=0.7)
    ax.grid(True, axis="y", color="#E5E9F0", linewidth=0.35, alpha=0.55)
    ax.tick_params(axis="both", labelsize=8, length=2)
    ax.set_ylabel(sample.audio_path.stem[:30], fontsize=7)


def save_class_figure(
    category: str,
    norobo_samples: list[Sample],
    robo_samples: list[Sample],
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(
        nrows=len(norobo_samples),
        ncols=2,
        figsize=(14, 8.5),
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )
    fig.suptitle(
        f"WAV waveform comparison - {category}",
        fontsize=16,
        fontweight="bold",
    )

    for idx, (norobo_sample, robo_sample) in enumerate(zip(norobo_samples, robo_samples)):
        plot_waveform(axes[idx, 0], norobo_sample, "#2A6F97")
        plot_waveform(axes[idx, 1], robo_sample, "#B85C38")
        axes[idx, 0].set_title(DOMAIN_LABELS["norobo"] if idx == 0 else "", fontsize=12)
        axes[idx, 1].set_title(DOMAIN_LABELS["robo"] if idx == 0 else "", fontsize=12)
        axes[idx, 0].set_xlabel("Time (s)", fontsize=8)
        axes[idx, 1].set_xlabel("Time (s)", fontsize=8)

    output_path = output_dir / f"waveform_compare_{category}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_overview_figure(
    selected: dict[str, dict[str, list[Sample]]],
    output_dir: Path,
) -> Path:
    rows = len(CLASSES)
    cols = 10
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(24, 10),
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )
    fig.suptitle(
        "WAV waveform overview - 5 no-robo vs 5 robo samples per class",
        fontsize=18,
        fontweight="bold",
    )

    for row_idx, category in enumerate(CLASSES):
        row_samples = selected[category]["norobo"] + selected[category]["robo"]
        for col_idx, sample in enumerate(row_samples):
            domain = "norobo" if col_idx < 5 else "robo"
            color = "#2A6F97" if domain == "norobo" else "#B85C38"
            ax = axes[row_idx, col_idx]
            plot_waveform(ax, sample, color)
            ax.set_xlabel("")
            ax.set_ylabel(category if col_idx == 0 else "", fontsize=11, fontweight="bold")
            if row_idx == 0:
                ax.set_title(f"{DOMAIN_LABELS[domain]} {col_idx % 5 + 1}", fontsize=9)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    output_path = output_dir / "waveform_compare_all_classes.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Visualize WAV waveforms for 4 classes in robo/no-robo datasets."
    )
    parser.add_argument(
        "--norobo-dir",
        type=Path,
        default=root / "img_audio" / "audio_visual_dataset_default",
        help="Path to the no-robot dataset directory.",
    )
    parser.add_argument(
        "--robo-dir",
        type=Path,
        default=root / "img_audio" / "audio_visual_dataset_robo_default",
        help="Path to the robot dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "project" / "visualize" / "imgs",
        help="Directory for generated PNG figures.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=5,
        help="Number of WAV files to plot per class and per domain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    norobo_samples = read_dataset(args.norobo_dir)
    robo_samples = read_dataset(args.robo_dir)

    selected: dict[str, dict[str, list[Sample]]] = {}
    class_paths: list[Path] = []
    for category in CLASSES:
        selected[category] = {
            "norobo": choose_samples(norobo_samples, category, args.samples_per_class),
            "robo": choose_samples(robo_samples, category, args.samples_per_class),
        }
        class_paths.append(
            save_class_figure(
                category,
                selected[category]["norobo"],
                selected[category]["robo"],
                args.output_dir,
            )
        )

    overview_path = save_overview_figure(selected, args.output_dir)

    print("Generated waveform visualizations:")
    for path in class_paths:
        print(f"- {path}")
    print(f"- {overview_path}")


if __name__ == "__main__":
    main()
