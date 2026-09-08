import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from zhiji.api.service import ModelBundle
from zhiji.evaluation.metrics import evaluate_predictions


def evaluate_frozen_models(
    feature_dir: Path, artifact_root: Path, *, bootstrap_repeats=1000
):
    destination = Path("reports/evaluation/test_metrics.json")
    if destination.exists():
        raise FileExistsError(
            "Frozen test evaluation already exists; do not tune on test results"
        )
    manifests = json.loads(
        Path("reports/evaluation/validation_ablations.json").read_text()
    )
    test = pl.read_parquet(feature_dir / "test.parquet").to_pandas()
    reports = {}
    for manifest in manifests:
        bundle = ModelBundle(artifact_root / "models" / manifest["model_version"])
        p = bundle.model.predict_proba(
            bundle.encoder.transform(test[manifest["columns"]])
        )[:, 1]
        metrics, intervals = evaluate_predictions(
            test.target_correct,
            p,
            test.student_id,
            risk_threshold=manifest["risk_thresholds"]["high"],
            bootstrap_repeats=bootstrap_repeats,
        )
        reports[manifest["model_version"]] = {
            "metrics": metrics,
            "student_bootstrap_95_ci": intervals,
            "bootstrap_repeats": bootstrap_repeats,
            "context": "known" if manifest["known_context"] else "unknown",
        }
        print(f"Test evaluation frozen: {manifest['model_version']}", flush=True)
    for name in ["constant", "A", "AB", "ABC"]:
        path = (
            artifact_root
            / "models/baselines-v2"
            / ("constant.joblib" if name == "constant" else f"lr-{name}.joblib")
        )
        baseline = joblib.load(path)
        if name == "constant":
            p = np.full(len(test), baseline["correct_rate"])
            threshold = 1 - baseline["correct_rate"]
        else:
            p = baseline["pipeline"].predict_proba(test[baseline["columns"]])[:, 1]
            threshold = baseline["threshold"]
        metrics, intervals = evaluate_predictions(
            test.target_correct,
            p,
            test.student_id,
            risk_threshold=threshold,
            bootstrap_repeats=bootstrap_repeats,
        )
        reports["baseline-" + name] = {
            "metrics": metrics,
            "student_bootstrap_95_ci": intervals,
            "bootstrap_repeats": bootstrap_repeats,
            "context": "unknown",
        }
        print(f"Test baseline frozen: {name}", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(reports, indent=2))
    return reports
