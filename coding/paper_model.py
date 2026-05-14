from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class PaperLikeFusionNet(nn.Module):
    """AST + CLAP + ViT-B/16 with lightweight Transformer fusion."""

    def __init__(
        self,
        num_classes: int = 4,
        sample_rate: int = 22000,
        ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        clap_model_name: str = "laion/clap-htsat-unfused",
        fusion_dim: int = 256,
        fusion_heads: int = 4,
        fusion_layers: int = 2,
        fusion_dropout: float = 0.1,
        freeze_pretrained: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import ASTFeatureExtractor, ASTModel, AutoProcessor, ClapModel
        except ImportError as exc:
            raise ImportError(
                "PaperLikeFusionNet requires transformers. On Kaggle, install/enable "
                "`transformers` and allow model weights from the Kaggle cache/input or internet."
            ) from exc

        self.sample_rate = sample_rate
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

        vit_weights = models.ViT_B_16_Weights.DEFAULT
        self.image_model = models.vit_b_16(weights=vit_weights)
        vit_dim = self.image_model.heads.head.in_features
        self.image_model.heads = nn.Identity()

        ast_dim = self.ast_model.config.hidden_size
        clap_dim = self.clap_model.config.projection_dim
        self.ast_proj = nn.Linear(ast_dim, fusion_dim)
        self.clap_proj = nn.Linear(clap_dim, fusion_dim)
        self.image_proj = nn.Linear(vit_dim, fusion_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=fusion_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=fusion_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=fusion_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, num_classes),
        )

        if freeze_pretrained:
            self._freeze_pretrained()

    def _freeze_pretrained(self) -> None:
        for module in (self.ast_model, self.clap_model, self.image_model):
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, waveform: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        ast_emb = self.encode_ast(waveform)
        clap_emb = self.encode_clap(waveform)
        image_emb = self.image_model(image)

        tokens = torch.stack(
            [
                self.ast_proj(ast_emb),
                self.clap_proj(clap_emb),
                self.image_proj(image_emb),
            ],
            dim=1,
        )
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        fused = self.fusion(torch.cat([cls, tokens], dim=1))
        return self.classifier(fused[:, 0])

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
        outputs = self.ast_model(**inputs)
        return _extract_model_embedding(outputs)

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
            raise ImportError("scipy is required to adapt waveform sample rates for pretrained processors.") from exc

        import math

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
