from __future__ import annotations

import numpy as np
import torch

try:
    from .config import LABEL_TO_ID
except ImportError:
    from config import LABEL_TO_ID


def f1_scores(y_true, y_pred, num_classes: int = 4):
    try:
        from sklearn.metrics import f1_score

        return {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    except Exception:
        return _manual_f1_scores(y_true, y_pred, num_classes)


def binary_contact_f1(y_true, y_pred):
    ambient_id = LABEL_TO_ID["ambient"]
    true_contact = np.asarray(y_true) != ambient_id
    pred_contact = np.asarray(y_pred) != ambient_id
    try:
        from sklearn.metrics import f1_score

        return float(f1_score(true_contact, pred_contact, zero_division=0))
    except Exception:
        tp = float(np.logical_and(true_contact, pred_contact).sum())
        fp = float(np.logical_and(~true_contact, pred_contact).sum())
        fn = float(np.logical_and(true_contact, ~pred_contact).sum())
        denom = (2 * tp + fp + fn)
        return 0.0 if denom == 0 else (2 * tp / denom)


def collect_predictions(logits_list, labels_list):
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    preds = logits.argmax(dim=1)
    return labels.cpu().numpy(), preds.cpu().numpy()


def _manual_f1_scores(y_true, y_pred, num_classes: int):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_class = []
    supports = []
    for cls in range(num_classes):
        true_cls = y_true == cls
        pred_cls = y_pred == cls
        tp = float(np.logical_and(true_cls, pred_cls).sum())
        fp = float(np.logical_and(~true_cls, pred_cls).sum())
        fn = float(np.logical_and(true_cls, ~pred_cls).sum())
        support = float(true_cls.sum())
        denom = 2 * tp + fp + fn
        per_class.append(0.0 if denom == 0 else 2 * tp / denom)
        supports.append(support)
    supports = np.asarray(supports)
    per_class = np.asarray(per_class)
    weighted = 0.0 if supports.sum() == 0 else float((per_class * supports).sum() / supports.sum())
    return {"macro_f1": float(per_class.mean()), "weighted_f1": weighted}
