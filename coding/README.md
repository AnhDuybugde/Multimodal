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
- Resample audio to `16000 Hz` for model input.
- Use `0.8s` audio windows:
  - train/validation/test: energy-aware crop by default.
- Apply a lightweight spectral-gating-style denoising step before cropping.
- Build mel-spectrograms in the pipeline.
- Use AST + CLAP + ViT-B/16 encoders.
- Fuse modality embeddings with a lightweight Transformer encoder.
- Use class-weighted cross entropy by default to handle the released dataset's class imbalance.
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
python coding/train_paper.py --data-root dataset --epochs 5 --batch-size 4 --task multiclass
python coding/train_paper.py --data-root dataset --epochs 5 --batch-size 4 --task binary
```

`train_paper.py` defaults to the paper-style domain protocol: train on
`audio_visual_dataset_default/` and validate/test on
`audio_visual_dataset_robo_default/`. It reports F1/precision/recall for the
selected task, plus macro/weighted sanity metrics. The AST branch consumes the
explicit mel-spectrogram from `AudioPipeline` by default
(`--ast-input-source mel`); CLAP still uses its Hugging Face processor because
the pretrained CLAP checkpoint expects its own HTSAT audio features.

## Audio Debug First

Before spending GPU time on AST/CLAP training, run the audio data checks:

```bash
python coding/debug_audio_pipeline.py --data-root dataset --sample-per-label 80 --baseline
```

Useful variants:

```bash
python coding/debug_audio_pipeline.py --data-root dataset --sample-per-label 20 --processor-check
python coding/train_audio_debug.py --data-root dataset --epochs 5 --batch-size 4 --class-weights --weighted-sampler
```

For a Kaggle notebook focused on the no-class-weight audio comparison, use:

```text
coding/kaggle_audio_paper_aligned.ipynb
```

For the full paper-aligned audio/video/fusion run, use:

```text
coding/kaggle_paper_aligned_full.ipynb
```

The audio debug script checks raw WAV statistics, processed waveform statistics after resample/crop, center-vs-energy crop retention, mini-batch waveform sanity, simple RMS/energy baselines, and optional AST/CLAP processor tensor statistics.

The audio training debug entrypoint focuses on AST+CLAP audio-only training. It freezes AST/CLAP by default and trains only the MLP head, then prints trainable parameters, batch waveform stats, prediction distribution, confusion matrix, per-class F1, logits stats, and gradient norms.

For Kaggle, open `coding/kaggle_train.ipynb` and adjust `DATA_ROOT` to your Kaggle dataset path.
The notebook is self-contained, so it does not need to import helper functions from the `.py` files.
It trains three separate runs:

```text
audio  -> best_audio_model.pt,  audio_results.json
video  -> best_video_model.pt,  video_results.json
fusion -> best_fusion_model.pt, fusion_results.json
```

The `video` branch uses the released image frame for each sample because the processed dataset does not include full video clips.

## Notes
- This is not a raw ROS bag reproduction because the released local data does not include ROS bags or force-torque streams.
- The training script downloads/loads pretrained AST, CLAP, and ViT weights. On Kaggle, enable internet or attach a Kaggle dataset/cache that contains those weights.
- If GPU memory is limited, reduce `--batch-size` or use `--freeze-pretrained`.
