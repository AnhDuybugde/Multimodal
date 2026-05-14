# Difference From Paper: Local Dataset vs. Paper Description

## Purpose
- This note records how the local `dataset/` folder differs from the dataset description in the paper notes.
- The local dataset appears to be a processed/exported version from the source release, not the original raw ROS bag dataset described in the paper.

## Files Checked
- Local dataset root: `dataset/`
- Hand-held/processed split: `dataset/audio_visual_dataset_default/`
- Robot/processed split: `dataset/audio_visual_dataset_robo_default/`
- Paper notes: `.agent/paper_note.md`
- Source archive: `contact_data.zip`

## Paper Description Summary
According to `.agent/paper_note.md`, the paper describes:
- Total recordings: 300 ROS bag files.
- Hand-held probe: 250 bags.
- Robot-mounted probe: 50 bags.
- Classes:
  - `leaf`
  - `twig`
  - `trunk`
  - `ambient` / no contact
- Sensors/streams:
  - Camera images.
  - Contact microphone audio.
  - Force-torque data used for annotation verification.
- Sampling rates:
  - Audio: 22 kHz.
  - Force-torque: 250 Hz.
  - Camera: 30 Hz.
- Preprocessing:
  - Audio transformed into mel-spectrograms.
  - Spectral gating for background noise reduction.
  - Audio window length reported as 0.8 seconds.

## Local Dataset Structure
The local `dataset/` folder contains two processed audio-visual datasets:

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

Each `dataset.csv` has the columns:

```text
audio_file,image_file,category
```

This means the local data is organized as paired audio-image classification samples, not raw multi-sensor ROS bags.

## What Matches the Paper
- The local dataset has the same four classification labels:
  - `ambient`
  - `leaf`
  - `twig`
  - `trunk`
- The local dataset separates hand-held/default data from robot/default data:
  - `audio_visual_dataset_default`
  - `audio_visual_dataset_robo_default`
- Each row contains one audio file, one image file, and one category label, matching the audio-visual contact classification task.
- Image samples checked locally are `640x480`, which is consistent with camera-frame style data.
- The local dataset supports the same broad task described by the paper: audio-visual classification of contact state/material in tree structures.

## Key Differences

### 1. Processed samples instead of raw ROS bags
The paper describes 300 ROS bag files:
- 250 hand-held bags.
- 50 robot-mounted bags.

The local folder does not contain `.bag` files. It contains already-windowed `.wav` and `.jpg` files plus CSV labels.

Interpretation:
- The local dataset is likely a processed release derived from the raw recordings.
- It is suitable for training/evaluating classifiers.
- It is not the full raw dataset described in the collection section of the paper.

### 2. Sample counts do not equal paper bag counts
Observed local counts:

| Split | CSV rows | Audio files | Image files |
|---|---:|---:|---:|
| `audio_visual_dataset_default` | 10,676 | 10,676 | 10,676 |
| `audio_visual_dataset_robo_default` | 2,219 | 2,219 | 2,218 |

These are window-level samples, not recording-level ROS bags.

Approximate unique recording prefixes inferred from filenames:

| Split | Unique inferred prefixes |
|---|---:|
| `audio_visual_dataset_default` | 235 |
| `audio_visual_dataset_robo_default` | 45 |

These inferred prefixes are close to, but not exactly the same as, the paper's 250 hand-held and 50 robot bag counts. This is expected if some recordings were filtered, renamed, excluded, or split differently during preprocessing.

### 3. Audio metadata differs from paper note
Paper note says:
- Audio sampling rate: 22 kHz.
- Chosen audio window length: 0.8 seconds.

Local WAV files checked:
- Sampling rate: 44.1 kHz.
- Bit depth: 16-bit.
- Channels: mono.
- Duration: 1.0 second.

Example checked files:
- `dataset/audio_visual_dataset_default/audio/letwig1.4_segment_0_window_0_ambient.wav`
- `dataset/audio_visual_dataset_robo_default/audio/umass_tree7_robot_trunk2_segment_0_window_0_contact.wav`

Interpretation:
- The local processed dataset may use a different export/resampling configuration than the paper note.
- Another possibility is that the paper note's 22 kHz value describes training preprocessing, while released WAV files are stored at 44.1 kHz and later resampled by code.
- This should be verified against the actual source code before implementing preprocessing.

### 4. No force-torque stream in local processed dataset
The paper says force-torque data is used to verify annotation quality.

The local `dataset/` folder only contains:
- `.wav` audio files.
- `.jpg` image files.
- `dataset.csv` label files.

It does not include force-torque measurements or raw synchronized ROS topics.

Interpretation:
- Force-torque data may have been used during annotation, but it is not included in this processed training dataset.

### 5. No mel-spectrogram files are stored
The paper note says audio is transformed into mel-spectrograms during preprocessing.

The local dataset stores raw audio windows as `.wav`, not precomputed mel-spectrogram files.

Interpretation:
- Mel-spectrogram generation is probably expected to happen in the training/preprocessing pipeline.
- The local dataset should not be treated as already containing model-ready spectrogram tensors.

### 6. One missing robot image
The robot split has one CSV row whose image file is missing locally.

Missing image:

```text
dataset/audio_visual_dataset_robo_default/images/umass_tree2_robot_leaf2_segment_2_window_2_ambient.jpg
```

Corresponding CSV row:

```text
audio/umass_tree2_robot_leaf2_segment_2_window_2_ambient.wav,
images/umass_tree2_robot_leaf2_segment_2_window_2_ambient.jpg,
ambient
```

The source archive `contact_data.zip` was also checked and does not contain this missing image, so this appears to be an issue in the released/archive dataset rather than a local extraction mistake.

## Class Distribution

### `audio_visual_dataset_default`

| Class | Count |
|---|---:|
| `ambient` | 5,966 |
| `leaf` | 1,670 |
| `trunk` | 1,476 |
| `twig` | 1,564 |
| **Total** | **10,676** |

### `audio_visual_dataset_robo_default`

| Class | Count |
|---|---:|
| `ambient` | 1,132 |
| `leaf` | 293 |
| `trunk` | 461 |
| `twig` | 333 |
| **Total** | **2,219** |

## Practical Conclusion
The local `dataset/` folder is mostly consistent with the paper at the task level:
- same contact classification problem;
- same label set;
- same broad hand-held vs. robot data split;
- same audio-visual sample pairing idea.

However, it is not identical to the paper's raw dataset description:
- it does not contain the 300 ROS bag files;
- it is already converted into window-level `.wav` and `.jpg` samples;
- it does not include force-torque data;
- local audio files checked are 44.1 kHz and 1.0 second, while the paper note says 22 kHz and 0.8 second;
- one robot image referenced by the CSV is missing.

For implementation, treat this folder as a processed classification dataset. Do not assume it contains every raw sensor stream or every exact preprocessing detail described in the paper.

## Recommended Next Steps
- Before training, handle or remove the one robot sample with the missing image.
- Check the released source code to confirm whether WAV files are resampled to 22 kHz during training.
- Check whether the model expects 0.8-second crops from 1.0-second WAV files or uses the full 1.0-second windows.
- Generate mel-spectrograms in preprocessing if the training pipeline expects them.
