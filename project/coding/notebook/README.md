# Main pretrained fusion notebooks

This folder now keeps only the paper-aligned training notebooks that use the three frozen pretrained backbones:

- `fusion_early_concat.ipynb`: AST + CLAP + ViT embeddings, concatenation head.
- `fusion_middle_transformer.ipynb`: AST + CLAP + ViT modality tokens, lightweight Transformer fusion head.
- `fusion_late_logits.ipynb`: AST+CLAP audio branch logits and ViT image branch logits, learned logit fusion.

The removed `audio_ml_*` and `audio_deep_*` notebooks were auxiliary handcrafted-feature or from-scratch baselines. They did not represent the current main setup of frozen AST/CLAP/ViT backbones plus task-specific heads.
