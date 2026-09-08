import csv
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from zhiji.api.main import create_app
from zhiji.data.clean import build_canonical_interactions
from zhiji.data.ingest import USE_COLUMNS, ingest_csv_to_parquet
from zhiji.data.schema import AFFECT
from zhiji.data.split import build_prediction_examples, create_student_splits
from zhiji.ena.codes import export_ena_input
from zhiji.ena.contract import export_results
from zhiji.features.tabular import build_tabular_features
from zhiji.models.xgb import train_xgboost


def test_csv_to_model_rena_api(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[2]
    if not shutil.which("Rscript") or not (project / "renv/library").exists():
        pytest.skip("Full ENA integration requires renv::restore()")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uv.lock").write_text("synthetic fixture environment")
    rng = np.random.default_rng(42)
    rows = []
    for student in range(20):
        for event in range(40):
            row = {c: "" for c in USE_COLUMNS}
            row.update(
                problem_log_id=f"e{student}_{event}",
                problemlogid=f"e{student}_{event}",
                user_id=f"synthetic_{student}",
                problem_id="p",
                skill_id="s",
                skill="S",
                assignment_id="a",
                start_time=(
                    datetime(2012, 1, 1, tzinfo=UTC) + timedelta(minutes=event)
                ).strftime("%Y-%m-%d %H:%M:%S"),
                problem_type="algebra",
                correct=str(int(rng.random() < (student + 1) / 22)),
                hint_count=str(int(rng.integers(0, 3))),
                attempt_count=str(int(rng.integers(1, 4))),
                bottom_hint=str(event % 3 == 0)
                .lower()
                .replace("true", "1")
                .replace("false", "0"),
                ms_first_response=str(int(rng.integers(100, 10000))),
                original="1",
                position=str(event),
            )
            row.update({c: str(rng.random()) for c in AFFECT})
            rows.append(row)
    raw = tmp_path / "input.csv"
    with raw.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=USE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    ingest_csv_to_parquet(raw, tmp_path / "parts", columns=USE_COLUMNS, chunk_size=250)
    canonical = tmp_path / "interactions.parquet"
    build_canonical_interactions(
        str(tmp_path / "parts/*.parquet"),
        canonical,
        tmp_path / "conflicts.parquet",
        config={"exclude_problem_types": ["open_response"]},
    )
    splits = tmp_path / "splits.parquet"
    examples = tmp_path / "examples.parquet"
    features = tmp_path / "features"
    create_student_splits(canonical, splits)
    build_prediction_examples(canonical, splits, examples)
    build_tabular_features(canonical, examples, features)
    root = tmp_path / "artifacts"
    manifest = train_xgboost(
        features / "train.parquet",
        features / "validation.parquet",
        root / "models",
        config={"n_estimators": 10, "max_depth": 2, "early_stopping_rounds": 3},
    )
    assert manifest["validation"]["roc_auc"] >= 0
    thresholds = json.loads((features / "thresholds.json").read_text()) | {
        "affect_threshold": 0.5
    }
    ena_input = tmp_path / "ena.csv"
    export_ena_input(canonical, splits, ena_input, thresholds=thresholds)
    subprocess.run(
        [
            "Rscript",
            "--vanilla",
            str(project / "analysis/ena/run_ena.R"),
            str(ena_input),
            "4",
            "0.5",
        ],
        env=os.environ | {"ZHIJI_ENA_OUTPUT_ROOT": str(root / "ena")},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    ena = next((root / "ena").glob("ena-*"))
    assert export_results(ena)["passed"]
    (root / "registry.json").write_text(
        json.dumps(
            {
                "current_prediction_model": manifest["model_version"],
                "current_ena_analysis": ena.name,
            }
        )
    )
    with TestClient(create_app(root)) as client:
        assert client.get("/health").json()["ena_artifacts_loaded"]
        assert client.get("/api/v1/ena/groups/stable/network").status_code == 200
        assert client.get("/api/v1/ena/difference").status_code == 200
