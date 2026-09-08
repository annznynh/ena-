import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from zhiji.evaluation.metrics import metric_values, select_threshold
from zhiji.features.encoders import TabularEncoder


def train_baselines(feature_dir: Path, artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    train = pl.read_parquet(feature_dir / "train.parquet").to_pandas()
    val = pl.read_parquet(feature_dir / "validation.parquet").to_pandas()
    reports = {}
    rate = float(train.target_correct.mean())
    reports["constant"] = metric_values(
        val.target_correct, np.full(len(val), rate), 1 - rate
    )
    joblib.dump({"correct_rate": rate}, artifact_dir / "constant.joblib")
    for group in ["A", "AB", "ABC"]:
        columns = [c for c in train if c[:2] in [g + "_" for g in group]]
        pipe = make_pipeline(
            TabularEncoder(),
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=42),
        )
        pipe.fit(train[columns], train.target_correct)
        p = pipe.predict_proba(val[columns])[:, 1]
        threshold = select_threshold(val.target_correct, p)
        reports[group] = metric_values(val.target_correct, p, threshold)
        joblib.dump(
            {"pipeline": pipe, "columns": columns, "threshold": threshold},
            artifact_dir / f"lr-{group}.joblib",
        )
        print(f"Logistic {group}: {reports[group]}", flush=True)
    (artifact_dir / "validation.json").write_text(json.dumps(reports, indent=2))
    return reports
