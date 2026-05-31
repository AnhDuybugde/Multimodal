from __future__ import annotations

try:
    from .paper_model import PaperLikeFusionNet
except ImportError:
    from paper_model import PaperLikeFusionNet


AudioVisualFusionNet = PaperLikeFusionNet
