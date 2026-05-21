from __future__ import annotations

import numpy as np
import torch

try:
    from .config import LABEL_TO_ID, LABELS
except ImportError:
    from config import LABEL_TO_ID, LABELS


def f1_scores(y_true, y_pred, num_classes: int = 4):
    try:
        from sklearn.metrics import f1_score

        return {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    except Exception:
        return _manual_f1_scores(y_true, y_pred, num_classes)


def paper_classification_metrics(y_true, y_pred, labels=None, paper_average: str = "weighted"):
    labels = tuple(labels or LABELS)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    try:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        paper_p, paper_r, paper_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=paper_average, zero_division=0
        )
        per_p, per_r, per_f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(labels))), zero_division=0
        )
        return {
            "paper_average": paper_average,
            "paper_f1": float(paper_f1),
            "paper_precision": float(paper_p),
            "paper_recall": float(paper_r),
            "macro_f1": float(macro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "weighted_f1": float(weighted_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "per_class": {
                label: {
                    "precision": float(per_p[idx]),
                    "recall": float(per_r[idx]),
                    "f1": float(per_f1[idx]),
                    "support": int(support[idx]),
                }
                for idx, label in enumerate(labels)
            },
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=list(range(len(labels)))
            ).tolist(),
        }
    except Exception:
        out = _manual_f1_scores(y_true, y_pred, len(labels))
        out.update(
            {
                "paper_average": paper_average,
                "paper_f1": out["weighted_f1"] if paper_average == "weighted" else out["macro_f1"],
                "accuracy": float((y_true == y_pred).mean()),
            }
        )
        return out


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


def binary_contact_metrics(y_true, y_pred, paper_average: str = "binary"):
    ambient_id = LABEL_TO_ID["ambient"]
    true_contact = np.asarray(y_true) != ambient_id
    pred_contact = np.asarray(y_pred) != ambient_id
    return binary_metrics_from_ids(true_contact.astype(int), pred_contact.astype(int), paper_average)


def binary_metrics_from_ids(y_true, y_pred, paper_average: str = "binary"):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    try:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )

        average = "binary" if paper_average == "binary" else paper_average
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, pos_label=1, zero_division=0
        )
        per_p, per_r, per_f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )
        return {
            "paper_average": paper_average,
            "paper_f1": float(f1),
            "paper_precision": float(p),
            "paper_recall": float(r),
            "binary_contact_f1": float(f1),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "per_class": {
                "non_contact": {
                    "precision": float(per_p[0]),
                    "recall": float(per_r[0]),
                    "f1": float(per_f1[0]),
                    "support": int(support[0]),
                },
                "contact": {
                    "precision": float(per_p[1]),
                    "recall": float(per_r[1]),
                    "f1": float(per_f1[1]),
                    "support": int(support[1]),
                },
            },
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        }
    except Exception:
        f1 = _binary_f1_np(y_true, y_pred)
        return {
            "paper_average": paper_average,
            "paper_f1": f1,
            "binary_contact_f1": f1,
            "accuracy": float((y_true == y_pred).mean()),
        }


def _binary_f1_np(y_true, y_pred):
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = float(np.logical_and(y_true, y_pred).sum())
    fp = float(np.logical_and(~y_true, y_pred).sum())
    fn = float(np.logical_and(y_true, ~y_pred).sum())
    denom = (2 * tp + fp + fn)
    return 0.0 if denom == 0 else float(2 * tp / denom)


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
