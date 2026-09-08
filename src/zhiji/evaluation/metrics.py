"""Risk is the positive class throughout evaluation."""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metric_values(y_correct, p_correct, threshold: float) -> dict:
    y = 1 - np.asarray(y_correct, dtype=int)
    risk = 1 - np.asarray(p_correct, dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Evaluation requires both outcome classes")
    if not np.isfinite(risk).all() or (risk < 0).any() or (risk > 1).any():
        raise ValueError("Invalid probabilities")
    pred = risk >= threshold
    bins = np.minimum((risk * 10).astype(int), 9)
    ece = sum(
        np.mean(bins == b) * abs(y[bins == b].mean() - risk[bins == b].mean())
        for b in range(10)
        if (bins == b).any()
    )
    observed, predicted = calibration_curve(y, risk, n_bins=10)
    return {
        "roc_auc": float(roc_auc_score(y, risk)),
        "pr_auc_risk": float(average_precision_score(y, risk)),
        "f1_risk": float(f1_score(y, pred, zero_division=0)),
        "recall_risk": float(recall_score(y, pred, zero_division=0)),
        "precision_risk": float(precision_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, risk)),
        "ece": float(ece),
        "threshold": float(threshold),
        "risk_prevalence": float(y.mean()),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "calibration_curve": {
            "predicted": predicted.tolist(),
            "observed": observed.tolist(),
        },
    }


def select_threshold(y_correct, p_correct, minimum_recall=0.8) -> float:
    precision, recall, thresholds = precision_recall_curve(
        1 - np.asarray(y_correct), 1 - np.asarray(p_correct)
    )
    f1 = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    valid = recall[:-1] >= minimum_recall
    if not valid.any():
        raise ValueError("No validation threshold meets recall requirement")
    return float(thresholds[np.where(valid, f1, -1).argmax()])


def evaluate_predictions(
    y_correct,
    p_correct,
    student_ids,
    *,
    risk_threshold: float,
    bootstrap_repeats: int = 1000,
    seed: int = 42,
):
    result = metric_values(y_correct, p_correct, risk_threshold)
    y, p = np.asarray(y_correct), np.asarray(p_correct)
    _, inverse = np.unique(np.asarray(student_ids), return_inverse=True)
    student_count = int(inverse.max() + 1)
    if student_count < 2:
        raise ValueError("Student bootstrap requires at least two students")
    # Exact tied-score rank statistics: sort once, then reweight student clusters.
    # This is algebraically equivalent to sklearn sample_weight, without 1,000 sorts.
    risk = 1 - p
    order = np.argsort(-risk, kind="stable")
    sorted_risk = risk[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_risk) != 0) + 1]
    sorted_y = (1 - y)[order]
    sorted_students = inverse[order]
    n_by_student = np.bincount(inverse, minlength=student_count)
    squared_by_student = np.bincount(
        inverse, weights=(y - p) ** 2, minlength=student_count
    )
    positive = (1 - y).astype(bool)
    predicted = risk >= risk_threshold
    tp = np.bincount(inverse, weights=positive & predicted, minlength=student_count)
    fp = np.bincount(inverse, weights=~positive & predicted, minlength=student_count)
    fn = np.bincount(inverse, weights=positive & ~predicted, minlength=student_count)
    ece_bins = np.minimum((risk * 10).astype(int), 9)
    ece_sums = np.bincount(
        inverse * 10 + ece_bins, weights=(1 - y) - risk, minlength=student_count * 10
    ).reshape(student_count, 10)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        k: []
        for k in (
            "roc_auc",
            "pr_auc_risk",
            "brier_score",
            "f1_risk",
            "recall_risk",
            "precision_risk",
            "ece",
        )
    }
    for _ in range(bootstrap_repeats):
        counts = np.bincount(
            rng.integers(student_count, size=student_count), minlength=student_count
        )
        weights = counts[sorted_students]
        pos = np.add.reduceat(weights * sorted_y, boundaries)
        neg = np.add.reduceat(weights * (1 - sorted_y), boundaries)
        total_pos, total_neg = pos.sum(), neg.sum()
        if min(total_pos, total_neg) == 0:
            continue
        cumulative_pos = np.cumsum(pos)
        total = np.cumsum(pos + neg)
        auc = np.sum(neg * (cumulative_pos - pos / 2)) / (total_pos * total_neg)
        ap = (
            np.sum(
                pos
                * np.divide(
                    cumulative_pos,
                    total,
                    out=np.zeros_like(total, dtype=float),
                    where=total > 0,
                )
            )
            / total_pos
        )
        true_pos, false_pos, false_neg = (
            float(counts @ tp),
            float(counts @ fp),
            float(counts @ fn),
        )
        values["ece"].append(
            float(np.abs(counts @ ece_sums).sum() / (counts @ n_by_student))
        )
        values["roc_auc"].append(float(auc))
        values["pr_auc_risk"].append(float(ap))
        values["brier_score"].append(
            float((counts @ squared_by_student) / (counts @ n_by_student))
        )
        values["recall_risk"].append(float(true_pos / max(true_pos + false_neg, 1)))
        values["precision_risk"].append(float(true_pos / max(true_pos + false_pos, 1)))
        values["f1_risk"].append(
            float(2 * true_pos / max(2 * true_pos + false_pos + false_neg, 1))
        )
    intervals = {
        k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in values.items() if v
    }
    return result, intervals
