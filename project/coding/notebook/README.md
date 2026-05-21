# Notebook organization

Main paper-aligned fusion notebooks using the three frozen pretrained backbones:

- `fusion_early_concat.ipynb`: AST + CLAP + ViT embeddings, concatenation head.
- `fusion_middle_transformer.ipynb`: AST + CLAP + ViT modality tokens, lightweight Transformer fusion head.
- `fusion_late_logits.ipynb`: AST+CLAP audio branch logits and ViT image branch logits, learned logit fusion.

Auxiliary test/ablation notebooks:

- `audio_ml_*`: handcrafted audio feature baselines with classical ML heads.
- `audio_deep_*`: audio feature/deep head baselines for testing PSLA-logmel, STFT, MFCC, FFT, and MFCC+FFT behavior.

Use the fusion notebooks as the main pretrained-backbone experiments. Use the audio notebooks as supporting experiments, not as replacements for the AST/CLAP/ViT paper-aligned setup.
