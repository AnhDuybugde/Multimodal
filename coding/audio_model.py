from __future__ import annotations

import math

import torch
from torch import nn


class AudioMLPHead(nn.Module):
    def __init__(
        self,
        ast_dim: int,
        clap_dim: int,
        num_classes: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.ast_proj = nn.Sequential(nn.LayerNorm(ast_dim), nn.Linear(ast_dim, hidden_dim), nn.GELU())
        self.clap_proj = nn.Sequential(nn.LayerNorm(clap_dim), nn.Linear(clap_dim, hidden_dim), nn.GELU())
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, ast_emb: torch.Tensor, clap_emb: torch.Tensor) -> torch.Tensor:
        ast_token = self.ast_proj(ast_emb)
        clap_token = self.clap_proj(clap_emb)
        return self.classifier(torch.cat([ast_token, clap_token], dim=-1))


class ASTCLAPAudioClassifier(nn.Module):
    """Audio-only AST + CLAP feature extractor with a trainable MLP head."""

    def __init__(
        self,
        num_classes: int = 4,
        sample_rate: int = 16000,
        ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        clap_model_name: str = "laion/clap-htsat-unfused",
        hidden_dim: int = 256,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import ASTFeatureExtractor, ASTModel, AutoProcessor, ClapModel
        except ImportError as exc:
            raise ImportError("Install transformers to train the AST/CLAP audio model.") from exc

        self.sample_rate = sample_rate
        self.freeze_backbone = freeze_backbone
        self.ast_feature_extractor = ASTFeatureExtractor.from_pretrained(ast_model_name)
        self.ast_model = ASTModel.from_pretrained(ast_model_name)
        self.clap_processor = AutoProcessor.from_pretrained(clap_model_name)
        self.clap_model = ClapModel.from_pretrained(clap_model_name)
        self.ast_sample_rate = getattr(self.ast_feature_extractor, "sampling_rate", sample_rate)
        self.clap_sample_rate = getattr(
            getattr(self.clap_processor, "feature_extractor", None),
            "sampling_rate",
            sample_rate,
        )

        ast_dim = self.ast_model.config.hidden_size
        clap_dim = self.clap_model.config.projection_dim
        self.head = AudioMLPHead(ast_dim, clap_dim, num_classes, hidden_dim, dropout)

        if freeze_backbone:
            for module in (self.ast_model, self.clap_model):
                for param in module.parameters():
                    param.requires_grad = False
            self.ast_model.eval()
            self.clap_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.ast_model.eval()
            self.clap_model.eval()
        return self

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                ast_emb = self.encode_ast(waveform)
                clap_emb = self.encode_clap(waveform)
        else:
            ast_emb = self.encode_ast(waveform)
            clap_emb = self.encode_clap(waveform)
        return self.head(ast_emb, clap_emb)

    def encode_ast(self, waveform: torch.Tensor) -> torch.Tensor:
        device = waveform.device
        arrays = self._to_processor_arrays(waveform, self.ast_sample_rate)
        inputs = self.ast_feature_extractor(
            arrays,
            sampling_rate=self.ast_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        return _extract_model_embedding(self.ast_model(**inputs))

    def encode_clap(self, waveform: torch.Tensor) -> torch.Tensor:
        device = waveform.device
        arrays = self._to_processor_arrays(waveform, self.clap_sample_rate)
        inputs = self.clap_processor(
            audio=arrays,
            sampling_rate=self.clap_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        return _extract_model_embedding(self.clap_model.get_audio_features(**inputs))

    def _to_processor_arrays(self, waveform: torch.Tensor, target_rate: int):
        arrays = [item.detach().float().cpu().numpy() for item in waveform]
        if target_rate == self.sample_rate:
            return arrays
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise ImportError("scipy is required to adapt waveform sample rates for AST/CLAP.") from exc

        gcd = math.gcd(self.sample_rate, target_rate)
        up = target_rate // gcd
        down = self.sample_rate // gcd
        return [resample_poly(item, up=up, down=down).astype("float32") for item in arrays]


def _extract_model_embedding(output):
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
            nested = _extract_model_embedding(item)
            if torch.is_tensor(nested):
                return nested
    raise TypeError(f"Could not extract tensor embedding from output type: {type(output)!r}")
