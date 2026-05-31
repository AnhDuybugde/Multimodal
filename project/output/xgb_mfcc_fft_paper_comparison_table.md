| Models | Mốc paper so sánh | F1 (Paper) | F1 (Tôi) | Recall (Paper) | Recall (Tôi) | Precision (Paper) | Precision (Tôi) |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost + MFCC+FFT + H2 spectral gating | Audio-only DualAudio, no pretraining, Table I | 0.460 | 0.603 | 0.480 | 0.636 | 0.520 | 0.655 |
| XGBoost + MFCC+FFT + H3 noise augmentation | Audio-only DualAudio, no pretraining, Table I | 0.460 | 0.639 | 0.480 | 0.665 | 0.520 | 0.675 |
| XGBoost + MFCC+FFT + H4 Freq-MixStyle | Audio-only DualAudio, no pretraining, Table I | 0.460 | 0.625 | 0.480 | 0.655 | 0.520 | 0.662 |
| XGBoost + MFCC+FFT + H2+H3 | Audio-only DualAudio, no pretraining, Table I | 0.460 | 0.588 | 0.480 | 0.626 | 0.520 | 0.612 |
| XGBoost + MFCC+FFT + H2+H4 | Audio-only DualAudio, no pretraining, Table I | 0.460 | 0.579 | 0.480 | 0.621 | 0.520 | 0.602 |
| XGBoost + MFCC+FFT + H3+H4 | Audio-only DualAudio, no pretraining, Table I | 0.460 | 0.645 | 0.480 | 0.669 | 0.520 | 0.685 |
| XGBoost + MFCC+FFT + H2+H3+H4 | Audio-only DualAudio, no pretraining, Table I | 0.460 | 0.560 | 0.480 | 0.609 | 0.520 | 0.581 |
| Fusion Early AST+CLAP+ViT | Audio-image DualAudio, pretrained, Table I | 0.740 | 0.777 | 0.730 | 0.789 | 0.750 | 0.790 |
| Fusion Middle AST+CLAP+ViT | Audio-image DualAudio, pretrained, Table I | 0.740 | 0.766 | 0.730 | 0.782 | 0.750 | 0.780 |

Note: Paper metrics are from Table I multiclass classification. Local metrics use the CSV columns `paper_f1`, `paper_recall`, and `paper_precision` from the seven `xgb_mfcc_fft_*_results.csv` files and the two `fusion_*_single_result.csv` files.
