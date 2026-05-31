import json
from pathlib import Path

from generate_mel_cnn_gru_bo_notebook import NOTEBOOK_CODE as BASE_NOTEBOOK_CODE


PREFIX = BASE_NOTEBOOK_CODE.split("class MelCNNGRU", 1)[0]
PREFIX = PREFIX.replace(
    "# Smoke-test knobs. Full run defaults are N_BO_TRIALS=20 and EPOCHS_PER_TRIAL=20.\n"
    "MAX_SAMPLES_PER_CLASS = 0\n"
    "N_BO_TRIALS = 20\n"
    "EPOCHS_PER_TRIAL = 20\n"
    "EARLY_STOPPING_PATIENCE = 5\n"
    "SEED = 42\n",
    "# Smoke-test knobs. Full run defaults are N_BO_TRIALS=20 and EPOCHS_PER_TRIAL=20.\n"
    "MAX_SAMPLES_PER_CLASS = 0\n"
    "N_BO_TRIALS = 20\n"
    "EPOCHS_PER_TRIAL = 20\n"
    "EARLY_STOPPING_PATIENCE = 5\n"
    "SEED = 42\n",
)


SEQUENCE_BO_CODE = r'''
SEQUENCE_MODEL = "__SEQUENCE_MODEL__"
SEQUENCE_SEED_OFFSET = {"gru": 0, "lstm": 1, "transformer": 2}[SEQUENCE_MODEL]


class MelCNNSequence(nn.Module):
    def __init__(
        self,
        sequence_model: str,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        conv_filters_1: int,
        conv_filters_2: int,
        kernel_size: int,
        pool_freq: int,
        hidden_dim: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        num_classes: int = 4,
    ):
        super().__init__()
        self.sequence_model = sequence_model
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_filters_1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(conv_filters_1),
            nn.GELU(),
            nn.MaxPool2d((pool_freq, 1)),
            nn.Conv2d(conv_filters_1, conv_filters_2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(conv_filters_2),
            nn.GELU(),
            nn.MaxPool2d((pool_freq, 1)),
        )
        frames = 1 + max(0, BASE_AUDIO_CFG.window_samples - n_fft) // hop_length
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, max(frames, 1))
            _, conv_channels, conv_freq, conv_time = self.conv(dummy).shape
        token_dim = conv_channels * conv_freq

        if sequence_model == "gru":
            self.sequence = nn.GRU(
                input_size=token_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            out_dim = hidden_dim * (2 if bidirectional else 1)
            self.post_norm = nn.LayerNorm(out_dim)
        elif sequence_model == "lstm":
            self.sequence = nn.LSTM(
                input_size=token_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            out_dim = hidden_dim * (2 if bidirectional else 1)
            self.post_norm = nn.LayerNorm(out_dim)
        elif sequence_model == "transformer":
            heads = 4 if hidden_dim % 4 == 0 else 2
            self.input_proj = nn.Linear(token_dim, hidden_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, conv_time, hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.sequence = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            out_dim = hidden_dim
            self.post_norm = nn.LayerNorm(out_dim)
        else:
            raise ValueError(f"Unknown sequence_model: {sequence_model}")

        self.classifier = nn.Sequential(
            self.post_norm,
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2).flatten(2)
        if self.sequence_model in ("gru", "lstm"):
            out, _ = self.sequence(x)
            pooled = out.mean(dim=1)
        else:
            tokens = self.input_proj(x)
            pos_embed = self.pos_embed
            if pos_embed.shape[1] != tokens.shape[1]:
                pos_embed = F.interpolate(
                    pos_embed.transpose(1, 2),
                    size=tokens.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            pooled = self.sequence(tokens + pos_embed).mean(dim=1)
        return self.classifier(pooled)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, paper_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    binary_true = (np.asarray(y_true) != LABEL_TO_ID["ambient"]).astype(np.int64)
    binary_pred = (np.asarray(y_pred) != LABEL_TO_ID["ambient"]).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "paper_precision": float(precision),
        "paper_recall": float(recall),
        "paper_f1": float(paper_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "binary_contact_f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
    }


def class_weight_tensor(frame: pd.DataFrame, device) -> torch.Tensor:
    counts = frame["label_id"].value_counts().reindex(range(len(LABELS)), fill_value=0).to_numpy(dtype=np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_model(model, loader, device) -> dict:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(yb.numpy())
    return classification_metrics(np.concatenate(labels), np.concatenate(preds))


def make_config(sequence_model: str, **kwargs) -> dict:
    return {
        "sequence_model": sequence_model,
        "n_mels": int(kwargs["n_mels"]),
        "n_fft": int(kwargs["n_fft"]),
        "hop_length": int(kwargs["hop_length"]),
        "conv_filters_1": int(kwargs["conv_filters_1"]),
        "conv_filters_2": int(kwargs["conv_filters_2"]),
        "kernel_size": int(kwargs["kernel_size"]),
        "pool_freq": int(kwargs["pool_freq"]),
        "hidden_dim": int(kwargs["hidden_dim"]),
        "num_layers": int(kwargs["num_layers"]),
        "bidirectional": True,
        "dropout": float(kwargs["dropout"]),
        "lr": float(kwargs["lr"]),
        "weight_decay": float(kwargs["weight_decay"]),
        "batch_size": int(kwargs.get("batch_size", 32)),
    }


def train_one_trial(train_df: pd.DataFrame, eval_df: pd.DataFrame, config: dict, trial_seed: int) -> dict:
    set_seed(trial_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_cfg = {k: config[k] for k in ("n_mels", "n_fft", "hop_length")}
    train_ds = MelAudioDataset(train_df, BASE_AUDIO_CFG.train_crop, trial_seed, mel_cfg)
    eval_ds = MelAudioDataset(eval_df, BASE_AUDIO_CFG.eval_crop, trial_seed, mel_cfg)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    model = MelCNNSequence(
        sequence_model=config["sequence_model"],
        n_mels=config["n_mels"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        conv_filters_1=config["conv_filters_1"],
        conv_filters_2=config["conv_filters_2"],
        kernel_size=config["kernel_size"],
        pool_freq=config["pool_freq"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        bidirectional=config["bidirectional"],
        dropout=config["dropout"],
        num_classes=len(LABELS),
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(train_df, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    best_metrics = {"paper_f1": -1.0}
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, EPOCHS_PER_TRIAL + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        metrics = evaluate_model(model, eval_loader, device)
        if metrics["paper_f1"] > best_metrics["paper_f1"] + 1e-4:
            best_metrics = metrics
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if EARLY_STOPPING_PATIENCE and no_improve >= EARLY_STOPPING_PATIENCE:
                break
    return {"best_epoch": best_epoch, **best_metrics}


def run_optuna_backend(train_df: pd.DataFrame, eval_df: pd.DataFrame, optuna, sequence_model: str) -> pd.DataFrame:
    rows = []

    def objective(trial):
        config = make_config(
            sequence_model,
            n_mels=trial.suggest_categorical("n_mels", [64, 128]),
            n_fft=trial.suggest_categorical("n_fft", [512, 1024]),
            hop_length=trial.suggest_categorical("hop_length", [128, 256]),
            conv_filters_1=trial.suggest_categorical("conv_filters_1", [16, 32, 64]),
            conv_filters_2=trial.suggest_categorical("conv_filters_2", [32, 64, 128]),
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5]),
            pool_freq=trial.suggest_categorical("pool_freq", [2, 4]),
            hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            num_layers=trial.suggest_categorical("num_layers", [1, 2]),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            batch_size=32,
        )
        start = time.time()
        metrics = train_one_trial(train_df, eval_df, config, SEED + 10000 * SEQUENCE_SEED_OFFSET + trial.number + 1)
        row = {
            "trial": trial.number + 1,
            "backend": "optuna_tpe",
            "seconds": round(time.time() - start, 3),
            "config": json.dumps(config, sort_keys=True),
            **config,
            **metrics,
        }
        rows.append(row)
        print(
            f'sequence={sequence_model} trial={row["trial"]:03d} '
            f'paper_f1={row["paper_f1"]:.4f} macro_f1={row["macro_f1"]:.4f} '
            f'epoch={row["best_epoch"]} config={row["config"]}'
        )
        return row["paper_f1"]

    sampler = optuna.samplers.TPESampler(seed=SEED + SEQUENCE_SEED_OFFSET)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_BO_TRIALS)
    return pd.DataFrame(rows)


def decode_index(value, choices):
    idx = int(np.clip(round(float(value)), 0, len(choices) - 1))
    return choices[idx]


def config_from_bayes_params(sequence_model: str, **params) -> dict:
    return make_config(
        sequence_model,
        n_mels=decode_index(params["n_mels_i"], [64, 128]),
        n_fft=decode_index(params["n_fft_i"], [512, 1024]),
        hop_length=decode_index(params["hop_length_i"], [128, 256]),
        conv_filters_1=decode_index(params["conv_filters_1_i"], [16, 32, 64]),
        conv_filters_2=decode_index(params["conv_filters_2_i"], [32, 64, 128]),
        kernel_size=decode_index(params["kernel_size_i"], [3, 5]),
        pool_freq=decode_index(params["pool_freq_i"], [2, 4]),
        hidden_dim=decode_index(params["hidden_dim_i"], [64, 128, 256]),
        num_layers=decode_index(params["num_layers_i"], [1, 2]),
        dropout=float(params["dropout"]),
        lr=float(10 ** params["lr_log10"]),
        weight_decay=float(10 ** params["weight_decay_log10"]),
        batch_size=32,
    )


def run_bayes_opt_backend(train_df: pd.DataFrame, eval_df: pd.DataFrame, sequence_model: str) -> pd.DataFrame:
    try:
        from bayes_opt import BayesianOptimization
    except Exception as bayes_exc:
        try:
            import optuna
        except Exception as optuna_exc:
            raise RuntimeError(
                "Bayesian optimization requires either `bayesian-optimization` (`bayes_opt`) "
                "or `optuna`. Install one of them before running this notebook."
            ) from optuna_exc
        print(f"`bayes_opt` unavailable ({bayes_exc}); using Optuna TPE sampler fallback for {sequence_model}.")
        return run_optuna_backend(train_df, eval_df, optuna, sequence_model)

    rows = []
    counter = {"n": 0}

    def objective(**params):
        counter["n"] += 1
        trial = counter["n"]
        config = config_from_bayes_params(sequence_model, **params)
        start = time.time()
        metrics = train_one_trial(train_df, eval_df, config, SEED + 10000 * SEQUENCE_SEED_OFFSET + trial)
        row = {
            "trial": trial,
            "backend": "bayes_opt",
            "seconds": round(time.time() - start, 3),
            "config": json.dumps(config, sort_keys=True),
            **config,
            **metrics,
        }
        rows.append(row)
        print(
            f'sequence={sequence_model} trial={trial:03d} '
            f'paper_f1={row["paper_f1"]:.4f} macro_f1={row["macro_f1"]:.4f} '
            f'epoch={row["best_epoch"]} config={row["config"]}'
        )
        return row["paper_f1"]

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            "n_mels_i": (0, 1),
            "n_fft_i": (0, 1),
            "hop_length_i": (0, 1),
            "conv_filters_1_i": (0, 2),
            "conv_filters_2_i": (0, 2),
            "kernel_size_i": (0, 1),
            "pool_freq_i": (0, 1),
            "hidden_dim_i": (0, 2),
            "num_layers_i": (0, 1),
            "dropout": (0.1, 0.5),
            "lr_log10": (math.log10(1e-4), math.log10(3e-3)),
            "weight_decay_log10": (math.log10(1e-5), math.log10(1e-3)),
        },
        random_state=SEED + SEQUENCE_SEED_OFFSET,
        verbose=0,
    )
    init_points = min(5, max(1, N_BO_TRIALS // 4))
    optimizer.maximize(init_points=init_points, n_iter=max(0, N_BO_TRIALS - init_points))
    return pd.DataFrame(rows)


def run_sequence_bo() -> pd.DataFrame:
    set_seed(SEED)
    data_root = find_data_root(DATASET_ROOT)
    print("data_root =", data_root)
    index = build_index(data_root)
    train_df = index[index["split_name"] == TRAIN_SPLIT].reset_index(drop=True)
    eval_df = index[index["split_name"] == ROBOT_SPLIT].reset_index(drop=True)
    if eval_df.empty:
        from sklearn.model_selection import train_test_split
        train_df, eval_df = train_test_split(
            train_df,
            test_size=0.2,
            random_state=SEED,
            stratify=train_df["label_id"],
        )
        train_df = train_df.reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)
        print("ROBOT split not found; using stratified validation split from train data.")
    train_df = sample_stratified(train_df, MAX_SAMPLES_PER_CLASS, SEED)
    eval_df = sample_stratified(eval_df, MAX_SAMPLES_PER_CLASS, SEED + 1)
    print("train rows =", len(train_df), "eval rows =", len(eval_df))
    print("sequence model =", SEQUENCE_MODEL)
    print("N_BO_TRIALS =", N_BO_TRIALS, "EPOCHS_PER_TRIAL =", EPOCHS_PER_TRIAL)

    print(f"\n=== RUNNING BO FOR {SEQUENCE_MODEL.upper()} ===")
    results = run_bayes_opt_backend(train_df, eval_df, SEQUENCE_MODEL)
    if not results.empty:
        results = results.sort_values(["paper_f1", "macro_f1"], ascending=False).reset_index(drop=True)
    return results


results = run_sequence_bo()

print("\n=== ALL BO TRIALS SORTED BY PAPER F1 ===")
display(results)

print("\n=== TOP 10 CONFIGS OVERALL ===")
display(results.head(10))

best = results.iloc[0].to_dict() if not results.empty else {}
print("\n=== BEST CONFIG JSON ===")
print(best.get("config", "{}"))

print("\n=== BEST VALIDATION METRICS ===")
best_metrics = {k: best.get(k) for k in ("sequence_model", "paper_f1", "macro_f1", "accuracy", "binary_contact_f1")}
print(json.dumps(best_metrics, indent=2))
'''


NOTEBOOK_CODE_TEMPLATE = PREFIX + SEQUENCE_BO_CODE


def make_notebook(sequence_model: str) -> dict:
    title = f"Audio Deep Mel-CNN {sequence_model.upper()} BO"
    notebook_code = NOTEBOOK_CODE_TEMPLATE.replace("__SEQUENCE_MODEL__", sequence_model)
    markdown = f"""# {title}

Self-contained Kaggle-style notebook running Bayesian Optimization for log-Mel + CNN + {sequence_model.upper()}.
"""
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": markdown.splitlines(True)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": notebook_code.splitlines(True)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    targets = {
        "gru": "audio-deep-mel-cnn-gru-sequence-bo.ipynb",
        "lstm": "audio-deep-mel-cnn-lstm-bo.ipynb",
        "transformer": "audio-deep-mel-cnn-transformer-bo.ipynb",
    }
    for sequence_model, filename in targets.items():
        path = out_dir / filename
        path.write_text(json.dumps(make_notebook(sequence_model), ensure_ascii=False, indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
