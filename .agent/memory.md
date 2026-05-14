# Agent Memory

## Project Context
- Folder: `DPL302m`
- Main document: `2505.12665v1.pdf`
- Paper topic: audio-visual contact classification for tree structures in agriculture.
- Current task focus: maintain `.agent` notes so future work is clearer and faster.

## File Roles
- `.agent/skills.md`: working rules for Codex in this project.
- `.agent/memory.md`: persistent context, workflow, file purpose, update history.
- `.agent/paper_note.md`: structured notes about the active research paper.
- `.agent/difference_from_paper.md`: differences between the released local dataset and the paper's raw dataset description.
- `coding/`: implementation folder for Kaggle-friendly training code.
- `2505.12665v1.pdf`: source paper used for the notes.

## Dataset Implementation Decision
- Treat `dataset/` as a released processed classification dataset, not the full raw ROS bag dataset.
- Do not modify source data files or original CSV files directly.
- Handle dataset issues in code:
  - skip samples whose audio/image file is missing;
  - resample audio at runtime/preprocessing time when matching paper settings;
  - use 0.8 second crop strategy in code instead of rewriting WAV files;
  - generate mel-spectrograms in the data pipeline;
  - keep force-torque/ROS bag reproduction out of scope because those streams are not provided locally.
- Missing robot image is low-impact: one CSV row references an image not present in both extracted data and `contact_data.zip`.
- Best wording for reports: "paper-aligned reproduction using the released processed audio-visual dataset," not exact raw-data reproduction.

## Current Code Architecture Decision
- The active training entrypoint is `coding/train.py`, which delegates to `coding/train_paper.py`.
- The active model is paper-like:
  - AST pretrained audio encoder;
  - CLAP pretrained audio encoder;
  - ViT-B/16 pretrained image encoder;
  - lightweight Transformer encoder for fusion;
  - classifier head over the fused CLS token.
- The dataset returns both:
  - `waveform`: 0.8 second, 22 kHz audio for AST/CLAP processors;
  - `audio`: mel-spectrogram kept for inspection/backward utility.
- Evaluation reports multiclass macro/weighted F1 and binary contact F1, with accuracy as secondary context.
- `coding/kaggle_train.ipynb` is self-contained for Kaggle use and does not depend on importing local `.py` helper modules.
- The Kaggle notebook trains three separate modes and writes three weight files plus three result JSON files:
  - `audio`: AST + CLAP + Transformer head;
  - `video`: ViT-B/16 over released image frames;
  - `fusion`: AST + CLAP + ViT-B/16 + Transformer fusion.

## Workflow To Follow
1. Read the user's request and identify whether it is about notes, paper understanding, or code.
2. Inspect the relevant files before making edits.
3. For paper tasks, extract factual details from the PDF before summarizing.
4. Update `.agent` files with clear headings, short bullets, and useful context.
5. Verify the edited files after writing.

## Expected Outputs
- Cleaner `.agent` documentation.
- Paper notes that can support class discussion, report writing, or slides.
- Concise summaries in Vietnamese unless the user asks otherwise.
- Kaggle-ready code/notebook for training without mutating original dataset files.

## Update History
- 2026-05-14 17:15 +07:00: Reworked Kaggle notebook to be self-contained and train audio/video/fusion modes separately.
- 2026-05-14 16:55 +07:00: Replaced compact baseline training path with paper-like AST + CLAP + ViT + Transformer fusion architecture.
- 2026-05-14 14:30 +07:00: Decided to preserve raw released data and implement paper-like preprocessing in code.
- 2026-05-14 09:37 +07:00: Reorganized `.agent` folder for clarity; expanded workflow, file roles, and paper notes.
