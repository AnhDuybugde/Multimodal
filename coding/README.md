# Audio-Visual Contact Classification

Kaggle-friendly, paper-aligned implementation for the released processed dataset.

## Goal
This code trains an audio-image fusion classifier for:

- `ambient`
- `leaf`
- `twig`
- `trunk`

It keeps the original dataset unchanged and handles paper-aligned preprocessing in code.

## Dataset Assumption
Expected local structure:

```text
dataset/
  audio_visual_dataset_default/
    audio/
    images/
    dataset.csv
  audio_visual_dataset_robo_default/
    audio/
    images/
    dataset.csv
```

## Paper-Aligned Choices
- Skip rows with missing audio/image files.
- Resample audio to `22000 Hz` at load time.
- Use `0.8s` audio windows:
  - train: random crop by default;
  - validation/test: center crop by default;
  - optional: energy-aware crop.
- Build mel-spectrograms in the pipeline.
- Use AST + CLAP + ViT-B/16 encoders.
- Fuse modality embeddings with a lightweight Transformer encoder.
- Report multiclass F1 and binary contact F1.

## Architecture
- Audio branch 1: pretrained AST from Hugging Face.
- Audio branch 2: pretrained CLAP from Hugging Face.
- Vision branch: pretrained ViT-B/16 from torchvision.
- Fusion: project each embedding to a shared dimension, prepend a learnable CLS token, run a small Transformer encoder, classify from the CLS token.

Default pretrained model names:

```text
AST:  MIT/ast-finetuned-audioset-10-10-0.4593
CLAP: laion/clap-htsat-unfused
ViT:  torchvision ViT_B_16_Weights.DEFAULT
```

## Quick Start

From the project root:

```bash
python coding/train_paper.py --data-root dataset --epochs 5 --batch-size 4
```

For Kaggle, open `coding/kaggle_train.ipynb` and adjust `DATA_ROOT` to your Kaggle dataset path.

## Notes
- This is not a raw ROS bag reproduction because the released local data does not include ROS bags or force-torque streams.
- The training script downloads/loads pretrained AST, CLAP, and ViT weights. On Kaggle, enable internet or attach a Kaggle dataset/cache that contains those weights.
- If GPU memory is limited, reduce `--batch-size` or use `--freeze-pretrained`.
