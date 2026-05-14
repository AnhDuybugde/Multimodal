# Audio-Visual Contact Classification

Kaggle-friendly implementation for the released processed dataset.

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
- Use audio-image fusion for classification.

## Quick Start

From the project root:

```bash
python coding/train.py --data-root dataset --epochs 5 --batch-size 16
```

For Kaggle, open `coding/kaggle_train.ipynb` and adjust `DATA_ROOT` to your Kaggle dataset path.

## Notes
- This is not a raw ROS bag reproduction because the released local data does not include ROS bags or force-torque streams.
- The model is a compact fusion baseline designed to run easily on Kaggle. Encoders can be replaced later with AST/CLAP/ViT-style backbones.
