# Paper Note: Audio-Visual Contact Classification for Tree Structures in Agriculture

## Reference
- Title: **Audio-Visual Contact Classification for Tree Structures in Agriculture**
- arXiv: `2505.12665v1 [cs.RO]`
- Date on paper: 19 May 2025
- Authors: Ryan Spears, Moonyoung Lee, George Kantor, Oliver Kroemer
- Institution: Carnegie Mellon University
- Website/source code: https://tree-classification.vercel.app/

## One-Sentence Summary
The paper proposes a multimodal robot perception system that fuses contact audio and camera images to classify whether a robot is touching **leaf**, **twig**, **trunk**, or **ambient/no contact** in cluttered orchard environments.

## Problem
- Agricultural robots must work in dense foliage where cameras are often blocked.
- Safe manipulation requires knowing what the robot is contacting.
- Vision alone is weak under occlusion and limited viewpoints.
- Traditional tactile sensors can be hard to scale over large contact areas and may be fragile in field use.

## Core Idea
- Use a low-cost contact microphone as vibrotactile sensing.
- Contact-induced vibration carries material-specific information.
- Audio helps detect **when** contact happens.
- Vision helps identify **what** object/material is involved.
- Combining both modalities improves contact classification.

## Contact Classes
- `leaf`
- `twig`
- `trunk`
- `ambient` / no contact

## Sensor Setup
- Camera: Intel RealSense D435.
- Audio: piezoelectric contact microphone.
- Audio interface: UMC404HD, 24-bit / 192 kHz converters.
- Force-torque sensor: ATI FT24252, used for annotation verification, not as model input.
- Synchronization: ROS.
- Sampling rates:
  - Audio: 22 kHz.
  - Force-torque: 250 Hz.
  - Camera: 30 Hz.

## Dataset
- Location: University of Massachusetts Amherst Cold Spring Orchard.
- Apple varieties: Fuji, Gala, Honeycrisp.
- Total recordings: 300 ROS bag files.
- Hand-held probe: 250 bags.
- Robot-mounted probe: 50 bags.
- Each bag contains two contact interactions with one object class over about 15 seconds.
- Robot deployment used a UR5 arm mounted on a mobile platform.

## Annotation Method
- Contact and no-contact intervals are automatically segmented from audio amplitude.
- The pipeline uses a smoothed moving average and dynamic threshold.
- Minimum contact segment duration: 1 second.
- Nearby contact segments are merged.
- Force-torque data is used to verify annotation quality.

## Preprocessing
- Audio is transformed into mel-spectrograms.
- Spectral gating is used to reduce background noise.
- Data augmentation includes motor/noise injection to bridge the hand-held-to-robot domain gap.
- The chosen audio window length is 0.8 seconds.

## Model Architecture
- Audio encoder 1: Audio Spectrogram Transformer (AST).
- Audio encoder 2: CLAP (Contrastive Language-Audio Pretraining).
- Vision encoder: ViT-B/16 pretrained on ImageNet.
- Fusion: concatenate AST, CLAP, and ViT embeddings, then pass through a lightweight transformer encoder and classifier.
- Purpose of fusion: combine fine-grained audio vibration patterns with visual semantic context.

## Main Results
- Zero-shot transfer from hand-held data to robot-mounted probe is demonstrated.
- Best reported multiclass F1 score: **0.82**.
- Best reported binary contact detection F1 score: **0.94**.
- Audio-only models are stronger than image-only models for detecting contact.
- Multimodal fusion is strongest for fine-grained class prediction.

## Important Comparisons
- Image-only ViT multiclass F1 with pretraining: **0.35**.
- Audio-only DualAudio multiclass F1 with pretraining: **0.53**.
- Multimodal fusion multiclass F1: **0.74** in the table discussion, and **0.82** as the best overall reported result.
- Binary contact detection:
  - Image-only ViT with pretraining: about **0.77 F1**.
  - Audio-only AST with pretraining: about **0.92 F1**.
  - Best overall binary result: **0.94 F1**.

## Audio Window Ablation
- Tested window lengths from 0.1 s to 1.0 s.
- Very short windows, 0.1-0.4 s, perform worse because they lack temporal context.
- Accuracy improves sharply around 0.5 s.
- Best performance occurs around **0.8 s**.
- Longer windows may add redundant or noisy information.

## Contributions
- Builds an audio-visual framework for tree-contact classification.
- Shows that contact microphones can provide useful vibrotactile signals in orchards.
- Demonstrates transfer from easier hand-held data collection to robot deployment.
- Releases an open-source multisensory dataset/code resource.

## Limitations / Future Work
- The classifier is not yet integrated into a closed-loop motion planner.
- The current setup classifies contact category but does not fully reason about contact geometry or force strategy.
- Future systems could use classification outputs for reactive robot control in pruning or harvesting.

## Useful Presentation Talking Points
- Main motivation: robots need touch-aware perception when vision is blocked by leaves and branches.
- Key insight: contact audio is not just noise; it is a useful tactile signal.
- Strongest argument: audio and vision solve different parts of the problem, so fusion is better than either alone.
- Practical value: hand-held data collection makes field dataset creation easier than collecting everything directly with the robot.
