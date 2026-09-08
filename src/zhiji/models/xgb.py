import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from xgboost import XGBClassifier

from zhiji.data.download import sha256
from zhiji.evaluation.metrics import metric_values, select_threshold
from zhiji.features.encoders import TabularEncoder


def train_xgboost(
    train_path: Path, validation_path: Path, artifact_dir: Path, *, config: dict
) -> dict:
    group = config.get("feature_set", "ABC")
    known = config.get("known_context", False)
    digest = hashlib.sha256(
        (json.dumps(config, sort_keys=True) + "encoder-v2").encode()
    ).hexdigest()[:10]
    version = f"xgb-{group}-{'known' if known else 'unknown'}-{digest}"
    out = artifact_dir / version
    if out.exists():
        raise FileExistsError(f"Frozen artifact exists: {out}")
    train = pl.read_parquet(train_path).to_pandas()
    val = pl.read_parquet(validation_path).to_pandas()
    prefixes = [g + "_" for g in group] + (["N_"] if known else [])
    columns = [c for c in train if c[:2] in prefixes]
    encoder = TabularEncoder().fit(train[columns])
    xtrain, xval = encoder.transform(train[columns]), encoder.transform(val[columns])
    params = {
        k: v for k, v in config.items() if k not in {"feature_set", "known_context"}
    }
    model = XGBClassifier(**params, eval_metric="logloss", n_jobs=4, random_state=42)
    model.fit(
        xtrain, train.target_correct, eval_set=[(xval, val.target_correct)], verbose=100
    )
    p = model.predict_proba(xval)[:, 1]
    threshold = select_threshold(val.target_correct, p)
    report = metric_values(val.target_correct, p, threshold)
    out.mkdir(parents=True)
    model.save_model(out / "model.json")
    joblib.dump(encoder, out / "preprocessor.joblib")
    manifest = {
        "artifact_type": "prediction_model",
        "model_name": "xgboost",
        "model_version": version,
        "feature_set": group,
        "known_context": known,
        "target": "next_correct",
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "validation": report,
        "columns": columns,
        "encoder_version": "v2-onehot-frequency",
        "encoded_columns": encoder.output_names_,
        "train_sha256": sha256(train_path),
        "validation_sha256": sha256(validation_path),
        "uv_lock_sha256": sha256(Path("uv.lock")),
        "model_sha256": sha256(out / "model.json"),
        "risk_thresholds": {"medium": threshold * 0.65, "high": threshold},
        "best_iteration": model.best_iteration,
        "calibration": "raw; validation curve reported",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "thresholds.json").write_text(
        (train_path.parent / "thresholds.json").read_text()
    )
    reloaded = XGBClassifier()
    reloaded.load_model(out / "model.json")
    if not np.allclose(
        reloaded.predict_proba(xval[:1000])[:, 1], p[:1000], atol=1e-6, rtol=0
    ):
        raise ValueError("Model roundtrip failed")
    print(json.dumps({"version": version, "validation": report}), flush=True)
    return manifest


def train_ablations(feature_dir: Path, artifact_dir: Path, *, config: dict) -> list:
    reports = []
    for known in [False, True]:
        for group in ["A", "AB", "ABC"]:
            reports.append(
                train_xgboost(
                    feature_dir / "train.parquet",
                    feature_dir / "validation.parquet",
                    artifact_dir,
                    config=config | {"feature_set": group, "known_context": known},
                )
            )
    # Deployment model selection uses validation only and keeps both context modes.
    registry = {}
    for known in [False, True]:
        candidates = [r for r in reports if r["known_context"] == known]
        best = max(candidates, key=lambda r: r["validation"]["pr_auc_risk"])
        registry[
            "current_prediction_model_known" if known else "current_prediction_model"
        ] = best["model_version"]
    (artifact_dir.parent / "registry.json").write_text(json.dumps(registry, indent=2))
    Path("reports/evaluation").mkdir(parents=True, exist_ok=True)
    Path("reports/evaluation/validation_ablations.json").write_text(
        json.dumps(reports, indent=2)
    )
    return reports
