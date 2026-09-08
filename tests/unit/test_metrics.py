import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from zhiji.evaluation.metrics import evaluate_predictions, metric_values


def test_exact_cluster_bootstrap_with_ties():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    p = np.array([0.2, 0.8, 0.3, 0.6, 0.6, 0.6, 0.8, 0.2])
    ids = np.repeat(np.arange(4), 2)
    _, intervals = evaluate_predictions(
        y, p, ids, risk_threshold=0.4, bootstrap_repeats=20
    )
    rng = np.random.default_rng(42)
    auc = []
    ap = []
    for _ in range(20):
        w = np.bincount(rng.integers(4, size=4), minlength=4)[ids]
        auc.append(roc_auc_score(1 - y, 1 - p, sample_weight=w))
        ap.append(average_precision_score(1 - y, 1 - p, sample_weight=w))
    assert np.allclose(intervals["roc_auc"], np.quantile(auc, [0.025, 0.975]))
    assert np.allclose(intervals["pr_auc_risk"], np.quantile(ap, [0.025, 0.975]))


def test_single_class_rejected():
    with pytest.raises(ValueError, match="both outcome"):
        metric_values([1, 1], [0.5, 0.7], 0.4)
