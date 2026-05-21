from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
except ImportError as exc:
    raise ImportError("This evaluator needs scikit-learn for paper-style metrics.") from exc


LABELS = ("ambient", "leaf", "trunk", "twig")
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
SPLITS = ("audio_visual_dataset_default", "audio_visual_dataset_robo_default")


@dataclass
class DataConfig:
    data_root: Path
    target_sample_rate: int = 16000
    audio_window_sec: float = 0.8
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    image_size: int = 224
    skip_missing_files: bool = True
    train_crop: str = "energy"
    eval_crop: str = "energy"
    spectral_gate: bool = True
    spectral_gate_noise_percentile: float = 20.0
    spectral_gate_strength: float = 1.0


@dataclass
class ModelConfig:
    ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    clap_model_name: str = "laion/clap-htsat-unfused"
    fusion_dim: int = 256
    fusion_heads: int = 4
    fusion_layers: int = 2
    fusion_dropout: float = 0.1
    freeze_pretrained: bool = True
    ast_input_source: str = "mel"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_index(data_root: Path, skip_missing_files: bool = True) -> pd.DataFrame:
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


class AudioPipeline:
    def __init__(
        self,
        target_sample_rate: int = 16000,
        window_sec: float = 0.8,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
        spectral_gate: bool = True,
        spectral_gate_noise_percentile: float = 20.0,
        spectral_gate_strength: float = 1.0,
    ) -> None:
        try:
            import torchaudio
        except (ImportError, OSError):
            torchaudio = None
        self.torchaudio = torchaudio
        self.target_sample_rate = target_sample_rate
        self.window_samples = int(round(target_sample_rate * window_sec))
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.spectral_gate = spectral_gate
        self.spectral_gate_noise_percentile = spectral_gate_noise_percentile
        self.spectral_gate_strength = spectral_gate_strength
        self._resamplers = {}
        if torchaudio is not None:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
            )
            self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        else:
            self.mel = None
            self.to_db = None

    def load_waveform(self, path: str) -> torch.Tensor:
        if self.torchaudio is not None:
            waveform, sample_rate = self.torchaudio.load(path)
        else:
            waveform, sample_rate = self._load_wav_stdlib(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.target_sample_rate:
            waveform = self._resample(waveform, sample_rate)
        return waveform

    def _resample(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if self.torchaudio is not None:
            if sample_rate not in self._resamplers:
                self._resamplers[sample_rate] = self.torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.target_sample_rate,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser",
                )
            return self._resamplers[sample_rate](waveform)
        from scipy.signal import resample_poly

        gcd = math.gcd(sample_rate, self.target_sample_rate)
        up = self.target_sample_rate // gcd
        down = sample_rate // gcd
        out = resample_poly(waveform.numpy(), up=up, down=down, axis=-1)
        return torch.from_numpy(out.copy()).float()

    def crop_or_pad(self, waveform: torch.Tensor, mode: str = "center") -> torch.Tensor:
        total = waveform.shape[-1]
        target = self.window_samples
        if total < target:
            return torch.nn.functional.pad(waveform, (0, target - total))
        if total == target:
            return waveform
        max_start = total - target
        if mode == "random":
            start = int(torch.randint(0, max_start + 1, (1,)).item())
        elif mode == "energy":
            energy = waveform.pow(2).mean(dim=0, keepdim=True).unsqueeze(0)
            kernel = torch.ones(1, 1, target)
            start = int(torch.nn.functional.conv1d(energy, kernel).argmax(dim=-1).item())
        else:
            start = max_start // 2
        return waveform[..., start : start + target]

    def apply_spectral_gate(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.spectral_gate or waveform.shape[-1] < 1024:
            return waveform
        n_fft = 1024
        hop = 256
        window = torch.hann_window(n_fft, device=waveform.device)
        spec = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        mag = spec.abs()
        noise = torch.quantile(
            mag,
            self.spectral_gate_noise_percentile / 100.0,
            dim=-1,
            keepdim=True,
        )
        gated_mag = (mag - self.spectral_gate_strength * noise).clamp_min(0.0)
        phase = spec / mag.clamp_min(1e-8)
        gated = gated_mag * phase
        return torch.istft(
            gated,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            length=waveform.shape[-1],
        )

    def __call__(self, path: str, crop_mode: str = "center") -> torch.Tensor:
        waveform = self.load_waveform(path)
        waveform = self.apply_spectral_gate(waveform)
        waveform = self.crop_or_pad(waveform, crop_mode)
        return waveform.squeeze(0)

    def waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.mel is None:
            raise ImportError("torchaudio is required for mel-spectrogram inference in this standalone script.")
        mel = self.to_db(self.mel(waveform))
        mel = ((mel + 80.0) / 80.0).clamp(0.0, 1.0)
        return mel

    @staticmethod
    def _load_wav_stdlib(path: str):
        import wave

        with wave.open(path, "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        if sample_width == 2:
            data = torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / float(2**15)
        elif sample_width == 1:
            data = (torch.frombuffer(bytearray(frames), dtype=torch.uint8).float() - 128.0) / 255.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")
        return data.view(-1, channels).t().contiguous(), sample_rate


class ContactDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, config: DataConfig, mode: str = "fusion") -> None:
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.mode = mode
        self.audio = AudioPipeline(
            config.target_sample_rate,
            config.audio_window_sec,
            config.n_mels,
            config.n_fft,
            config.hop_length,
            config.spectral_gate,
            config.spectral_gate_noise_percentile,
            config.spectral_gate_strength,
        )
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        out = {"label": torch.tensor(row.label_id, dtype=torch.long)}
        if self.mode in ("audio", "fusion"):
            waveform = self.audio(row.audio_path, self.config.eval_crop).unsqueeze(0)
            out["waveform"] = waveform.squeeze(0)
            out["audio"] = self.audio.waveform_to_mel(waveform)
        if self.mode in ("video", "fusion"):
            image = Image.open(row.image_path).convert("RGB")
            out["image"] = self.image_transform(image)
        out["binary_label"] = torch.tensor(0 if row.label == "ambient" else 1, dtype=torch.long)
        return out


class TransformerFusionHead(nn.Module):
    def __init__(
        self,
        input_dims,
        num_classes: int = 4,
        fusion_dim: int = 256,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(dim, fusion_dim) for dim in input_dims])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, embeddings):
        tokens = torch.stack([proj(emb) for proj, emb in zip(self.projections, embeddings)], dim=1)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        fused = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.classifier(fused[:, 0])


def extract_model_embedding(output):
    if torch.is_tensor(output):
        return output
    pooler = getattr(output, "pooler_output", None)
    if torch.is_tensor(pooler):
        return pooler
    last_hidden = getattr(output, "last_hidden_state", None)
    if torch.is_tensor(last_hidden):
        return last_hidden[:, 0]
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item[:, 0] if item.ndim == 3 else item
            try:
                nested = extract_model_embedding(item)
                if torch.is_tensor(nested):
                    return nested
            except TypeError:
                pass
    raise TypeError(f"Could not extract tensor embedding from output type: {type(output)!r}")


class ASTCLAPAudioModel(nn.Module):
    def __init__(self, cfg: ModelConfig, sample_rate: int = 16000, num_classes: int = 4) -> None:
        super().__init__()
        from transformers import ASTFeatureExtractor, ASTModel, AutoProcessor, ClapModel

        self.sample_rate = sample_rate
        self.ast_input_source = cfg.ast_input_source
        self.ast_feature_extractor = ASTFeatureExtractor.from_pretrained(cfg.ast_model_name)
        self.ast_model = ASTModel.from_pretrained(cfg.ast_model_name)
        self.clap_processor = AutoProcessor.from_pretrained(cfg.clap_model_name)
        self.clap_model = ClapModel.from_pretrained(cfg.clap_model_name)
        self.ast_sample_rate = getattr(self.ast_feature_extractor, "sampling_rate", sample_rate)
        self.clap_sample_rate = getattr(
            getattr(self.clap_processor, "feature_extractor", None),
            "sampling_rate",
            sample_rate,
        )
        ast_dim = self.ast_model.config.hidden_size
        clap_dim = self.clap_model.config.projection_dim
        self.head = TransformerFusionHead(
            [ast_dim, clap_dim],
            num_classes,
            cfg.fusion_dim,
            cfg.fusion_heads,
            cfg.fusion_layers,
            cfg.fusion_dropout,
        )
        if cfg.freeze_pretrained:
            for module in (self.ast_model, self.clap_model):
                for param in module.parameters():
                    param.requires_grad = False

    def encode_ast(self, waveform: torch.Tensor) -> torch.Tensor:
        arrays = self.to_processor_arrays(waveform, self.ast_sample_rate)
        inputs = self.ast_feature_extractor(
            arrays,
            sampling_rate=self.ast_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(waveform.device) for key, value in inputs.items()}
        return extract_model_embedding(self.ast_model(**inputs))

    def encode_clap(self, waveform: torch.Tensor) -> torch.Tensor:
        arrays = self.to_processor_arrays(waveform, self.clap_sample_rate)
        inputs = self.clap_processor(
            audio=arrays,
            sampling_rate=self.clap_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(waveform.device) for key, value in inputs.items()}
        return extract_model_embedding(self.clap_model.get_audio_features(**inputs))

    def to_processor_arrays(self, waveform: torch.Tensor, target_rate: int):
        arrays = [item.detach().float().cpu().numpy() for item in waveform]
        if target_rate == self.sample_rate:
            return arrays
        from scipy.signal import resample_poly

        gcd = math.gcd(self.sample_rate, target_rate)
        up = target_rate // gcd
        down = self.sample_rate // gcd
        return [resample_poly(item, up=up, down=down).astype("float32") for item in arrays]

    def encode_ast_mel(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim == 3:
            mel = mel.unsqueeze(1)
        mel_db = mel.squeeze(1)
        if mel_db.min() >= 0.0 and mel_db.max() <= 1.0:
            mel_db = mel_db * 80.0 - 80.0
        input_values = self._fit_ast_frames(mel_db.transpose(1, 2))
        mean = float(getattr(self.ast_feature_extractor, "mean", 0.0))
        std = float(getattr(self.ast_feature_extractor, "std", 1.0))
        input_values = (input_values - mean) / max(std, 1e-8)
        return extract_model_embedding(self.ast_model(input_values=input_values.to(mel.device)))

    def _fit_ast_frames(self, input_values: torch.Tensor) -> torch.Tensor:
        max_length = int(getattr(self.ast_feature_extractor, "max_length", input_values.shape[1]))
        if input_values.shape[1] > max_length:
            return input_values[:, :max_length, :]
        if input_values.shape[1] == max_length:
            return input_values
        pad = input_values.new_zeros(
            input_values.shape[0],
            max_length - input_values.shape[1],
            input_values.shape[2],
        )
        return torch.cat([input_values, pad], dim=1)

    def forward(self, waveform=None, image=None, audio=None):
        ast_emb = self.encode_ast_mel(audio) if self.ast_input_source == "mel" and audio is not None else self.encode_ast(waveform)
        return self.head([ast_emb, self.encode_clap(waveform)])


class ViTVideoModel(nn.Module):
    def __init__(self, cfg: ModelConfig, num_classes: int = 4) -> None:
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        vit_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        self.classifier = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(cfg.fusion_dim, num_classes),
        )
        if cfg.freeze_pretrained:
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(self, waveform=None, image=None):
        return self.classifier(self.vit(image))


class PaperFusionModel(nn.Module):
    def __init__(self, cfg: ModelConfig, sample_rate: int = 16000, num_classes: int = 4) -> None:
        super().__init__()
        from transformers import ASTFeatureExtractor, ASTModel, AutoProcessor, ClapModel

        self.sample_rate = sample_rate
        self.ast_input_source = cfg.ast_input_source
        self.ast_feature_extractor = ASTFeatureExtractor.from_pretrained(cfg.ast_model_name)
        self.ast_model = ASTModel.from_pretrained(cfg.ast_model_name)
        self.clap_processor = AutoProcessor.from_pretrained(cfg.clap_model_name)
        self.clap_model = ClapModel.from_pretrained(cfg.clap_model_name)
        self.ast_sample_rate = getattr(self.ast_feature_extractor, "sampling_rate", sample_rate)
        self.clap_sample_rate = getattr(
            getattr(self.clap_processor, "feature_extractor", None),
            "sampling_rate",
            sample_rate,
        )
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        vit_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        ast_dim = self.ast_model.config.hidden_size
        clap_dim = self.clap_model.config.projection_dim
        self.head = TransformerFusionHead(
            [ast_dim, clap_dim, vit_dim],
            num_classes,
            cfg.fusion_dim,
            cfg.fusion_heads,
            cfg.fusion_layers,
            cfg.fusion_dropout,
        )
        if cfg.freeze_pretrained:
            for module in (self.ast_model, self.clap_model, self.vit):
                for param in module.parameters():
                    param.requires_grad = False

    def encode_ast(self, waveform: torch.Tensor) -> torch.Tensor:
        arrays = self.to_processor_arrays(waveform, self.ast_sample_rate)
        inputs = self.ast_feature_extractor(
            arrays,
            sampling_rate=self.ast_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(waveform.device) for key, value in inputs.items()}
        return extract_model_embedding(self.ast_model(**inputs))

    def encode_clap(self, waveform: torch.Tensor) -> torch.Tensor:
        arrays = self.to_processor_arrays(waveform, self.clap_sample_rate)
        inputs = self.clap_processor(
            audio=arrays,
            sampling_rate=self.clap_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(waveform.device) for key, value in inputs.items()}
        return extract_model_embedding(self.clap_model.get_audio_features(**inputs))

    def to_processor_arrays(self, waveform: torch.Tensor, target_rate: int):
        arrays = [item.detach().float().cpu().numpy() for item in waveform]
        if target_rate == self.sample_rate:
            return arrays
        from scipy.signal import resample_poly

        gcd = math.gcd(self.sample_rate, target_rate)
        up = target_rate // gcd
        down = self.sample_rate // gcd
        return [resample_poly(item, up=up, down=down).astype("float32") for item in arrays]

    def encode_ast_mel(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim == 3:
            mel = mel.unsqueeze(1)
        mel_db = mel.squeeze(1)
        if mel_db.min() >= 0.0 and mel_db.max() <= 1.0:
            mel_db = mel_db * 80.0 - 80.0
        input_values = self._fit_ast_frames(mel_db.transpose(1, 2))
        mean = float(getattr(self.ast_feature_extractor, "mean", 0.0))
        std = float(getattr(self.ast_feature_extractor, "std", 1.0))
        input_values = (input_values - mean) / max(std, 1e-8)
        return extract_model_embedding(self.ast_model(input_values=input_values.to(mel.device)))

    def _fit_ast_frames(self, input_values: torch.Tensor) -> torch.Tensor:
        max_length = int(getattr(self.ast_feature_extractor, "max_length", input_values.shape[1]))
        if input_values.shape[1] > max_length:
            return input_values[:, :max_length, :]
        if input_values.shape[1] == max_length:
            return input_values
        pad = input_values.new_zeros(
            input_values.shape[0],
            max_length - input_values.shape[1],
            input_values.shape[2],
        )
        return torch.cat([input_values, pad], dim=1)

    def forward(self, waveform=None, image=None, audio=None):
        ast_emb = self.encode_ast_mel(audio) if self.ast_input_source == "mel" and audio is not None else self.encode_ast(waveform)
        return self.head([ast_emb, self.encode_clap(waveform), self.vit(image)])


def config_from_checkpoint(checkpoint: dict, data_root: Path | None) -> tuple[DataConfig, ModelConfig, str]:
    data_dict = dict(checkpoint.get("data_config", {}))
    model_dict = dict(checkpoint.get("model_config", {}))
    if data_root is not None:
        data_dict["data_root"] = data_root
    elif "data_root" in data_dict:
        data_dict["data_root"] = Path(data_dict["data_root"])
    else:
        data_dict["data_root"] = Path("dataset")

    data_fields = DataConfig.__dataclass_fields__
    model_fields = ModelConfig.__dataclass_fields__
    data_cfg = DataConfig(**{key: value for key, value in data_dict.items() if key in data_fields})
    model_cfg = ModelConfig(**{key: value for key, value in model_dict.items() if key in model_fields})
    mode = checkpoint.get("mode", "fusion")
    return data_cfg, model_cfg, mode


def build_model(mode: str, model_cfg: ModelConfig, sample_rate: int, num_classes: int):
    if mode == "audio":
        return ASTCLAPAudioModel(model_cfg, sample_rate, num_classes)
    if mode == "video":
        return ViTVideoModel(model_cfg, num_classes)
    if mode == "fusion":
        return PaperFusionModel(model_cfg, sample_rate, num_classes)
    raise ValueError(f"Unknown checkpoint mode: {mode}")


def select_eval_frame(index: pd.DataFrame, split: str, val_size: float, seed: int) -> pd.DataFrame:
    if split == "default":
        return index[index["split_name"] == "audio_visual_dataset_default"].reset_index(drop=True)
    if split == "robo":
        return index[index["split_name"] == "audio_visual_dataset_robo_default"].reset_index(drop=True)
    if split == "all":
        return index.reset_index(drop=True)
    if split == "random-val":
        from sklearn.model_selection import train_test_split

        _, val_df = train_test_split(
            index,
            test_size=val_size,
            random_state=seed,
            stratify=index["label_id"],
        )
        return val_df.reset_index(drop=True)
    raise ValueError(f"Unknown eval split: {split}")


@torch.inference_mode()
def predict(model, loader, device: str):
    model.eval()
    logits_all = []
    labels_all = []
    for batch in tqdm(loader, desc="infer", leave=False):
        labels = batch["label"].to(device)
        waveform = batch.get("waveform")
        audio = batch.get("audio")
        image = batch.get("image")
        if waveform is not None:
            waveform = waveform.to(device)
        if audio is not None:
            audio = audio.to(device)
        if image is not None:
            image = image.to(device)
        logits = model(waveform=waveform, image=image, audio=audio)
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    return labels, preds, probs


def metric_block(y_true, y_pred, average: str) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )
    return {
        "average": average,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def compute_metrics(y_true, y_pred, paper_average: str, labels=LABELS, task: str = "multiclass") -> dict:
    if task == "binary":
        return compute_binary_metrics(y_true, y_pred)
    ambient_id = LABEL_TO_ID["ambient"]
    y_true_contact = np.asarray(y_true) != ambient_id
    y_pred_contact = np.asarray(y_pred) != ambient_id

    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    report = {
        "paper_style_multiclass": metric_block(y_true, y_pred, paper_average),
        "multiclass_macro": metric_block(y_true, y_pred, "macro"),
        "multiclass_weighted": metric_block(y_true, y_pred, "weighted"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "binary_contact": metric_block(y_true_contact, y_pred_contact, paper_average),
        "per_class": {
            label: {
                "precision": float(per_class_precision[idx]),
                "recall": float(per_class_recall[idx]),
                "f1": float(per_class_f1[idx]),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist(),
    }
    return report


def compute_binary_metrics(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "paper_style_binary": {
            "average": "binary",
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        },
        "binary_contact": {
            "average": "binary",
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            "non_contact": {
                "precision": float(per_class_precision[0]),
                "recall": float(per_class_recall[0]),
                "f1": float(per_class_f1[0]),
                "support": int(support[0]),
            },
            "contact": {
                "precision": float(per_class_precision[1]),
                "recall": float(per_class_recall[1]),
                "f1": float(per_class_f1[1]),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def save_predictions(path: Path, y_true, y_pred, probs, frame: pd.DataFrame, labels=LABELS) -> None:
    out = frame.copy()
    out["y_true"] = [labels[idx] for idx in y_true]
    out["y_pred"] = [labels[idx] for idx in y_pred]
    for idx, label in enumerate(labels):
        out[f"prob_{label}"] = probs[:, idx]
    out.to_csv(path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference from a Kaggle notebook checkpoint and compute paper-style metrics."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--eval-split", choices=("robo", "default", "all", "random-val"), default="robo")
    parser.add_argument("--paper-average", choices=("weighted", "macro", "micro"), default="weighted")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-preds", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data_cfg, model_cfg, mode = config_from_checkpoint(checkpoint, args.data_root)
    task = checkpoint.get("task", "multiclass")
    labels = tuple(checkpoint.get("labels", LABELS if task == "multiclass" else ("non_contact", "contact")))
    if task == "multiclass" and labels != LABELS:
        raise ValueError(f"Unexpected checkpoint labels {labels}; expected {LABELS}")

    index = build_index(data_cfg.data_root, data_cfg.skip_missing_files)
    eval_df = select_eval_frame(index, args.eval_split, args.val_size, args.seed)
    print(f"checkpoint: {args.checkpoint}")
    print(f"mode: {mode}")
    print(f"data_root: {data_cfg.data_root}")
    print(f"eval_split: {args.eval_split}, samples: {len(eval_df)}")
    print(eval_df["label"].value_counts().reindex(LABELS).fillna(0).astype(int))

    dataset = ContactDataset(eval_df, data_cfg, mode=mode)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    model = build_model(mode, model_cfg, data_cfg.target_sample_rate, len(labels)).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)

    y_true, y_pred, probs = predict(model, loader, args.device)
    if task == "binary":
        y_true = (eval_df["label"].to_numpy() != "ambient").astype(int)
    metrics = compute_metrics(y_true, y_pred, args.paper_average, labels=labels, task=task)
    payload = {
        "checkpoint": str(args.checkpoint),
        "mode": mode,
        "task": task,
        "eval_split": args.eval_split,
        "paper_average": args.paper_average,
        "num_samples": len(eval_df),
        "metrics": metrics,
    }

    if task == "binary":
        print("\nPaper-style binary:")
        print(json.dumps(metrics["paper_style_binary"], indent=2))
    else:
        print("\nPaper-style multiclass:")
        print(json.dumps(metrics["paper_style_multiclass"], indent=2))
        print("\nBinary contact collapsed from multiclass:")
        print(json.dumps(metrics["binary_contact"], indent=2))
        print("\nMacro / weighted sanity check:")
        print(json.dumps({k: metrics[k] for k in ("multiclass_macro", "multiclass_weighted", "accuracy")}, indent=2))
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    print("Confusion matrix rows=true, cols=pred, label order:", labels)
    print(np.asarray(metrics["confusion_matrix"]))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved metrics: {args.out_json}")
    if args.out_preds is not None:
        args.out_preds.parent.mkdir(parents=True, exist_ok=True)
        save_predictions(args.out_preds, y_true, y_pred, probs, eval_df, labels)
        print(f"saved predictions: {args.out_preds}")


if __name__ == "__main__":
    main()
