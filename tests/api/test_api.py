import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient

from zhiji.api.main import create_app
from zhiji.api.service import risk_level
from zhiji.models.xgb import train_xgboost


def request_body():
    return {
        "student_id": "stu_synthetic",
        "history": [
            {
                "event_id": f"e{i}",
                "occurred_at": (
                    datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=i)
                ).isoformat(),
                "correct": i % 2,
                "attempt_count": 1,
                "hint_count": 0,
            }
            for i in range(5)
        ],
    }


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("api")
    features = root / "features"
    features.mkdir()
    frame = pl.DataFrame(
        {
            "A_last_correct": [i % 2 for i in range(40)],
            "target_correct": [i % 3 == 0 for i in range(40)],
        }
    )
    for split in ["train", "validation"]:
        frame.write_parquet(features / f"{split}.parquet")
    (features / "thresholds.json").write_text(
        json.dumps({"slow_by_type": {}, "slow_global": 9.0})
    )
    manifest = train_xgboost(
        features / "train.parquet",
        features / "validation.parquet",
        root / "models",
        config={"n_estimators": 5, "max_depth": 2, "early_stopping_rounds": 2},
    )
    (root / "registry.json").write_text(
        json.dumps({"current_prediction_model": manifest["model_version"]})
    )
    with TestClient(create_app(root)) as test_client:
        yield test_client


def test_health_and_prediction(client):
    assert client.get("/health").json()["model_loaded"]
    response = client.post("/api/v1/predictions/next", json=request_body())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["correct_probability"] + body["risk_probability"] == pytest.approx(1)
    assert 0 <= body["risk_probability"] <= 1
    assert "request_id" in body
    for forbidden in ["answer_text", "actions", "teacher_id", "school_id"]:
        assert forbidden not in response.text


@pytest.mark.parametrize(
    "case", ["short", "reverse", "affect", "negative", "duplicate"]
)
def test_invalid_history(client, case):
    body = request_body()
    if case == "short":
        body["history"] = body["history"][:1]
    if case == "reverse":
        body["history"].reverse()
    if case == "affect":
        body["history"][0]["affect"] = {"confused": 1.2}
    if case == "negative":
        body["history"][0]["hint_count"] = -1
    if case == "duplicate":
        body["history"][0]["event_id"] = "e1"
    response = client.post("/api/v1/predictions/next", json=body)
    assert response.status_code == 422
    assert "request_id" in response.json()


def test_missing_version_and_artifacts(client, tmp_path):
    body = request_body() | {"model_version": "incorrect-version"}
    assert client.post("/api/v1/predictions/next", json=body).status_code == 409
    assert client.get("/api/v1/students/stu_absent/risk-timeline").status_code == 404
    assert client.get("/api/v1/ena/groups/invalid/network").status_code == 422
    with TestClient(create_app(tmp_path)) as empty:
        assert (
            empty.post("/api/v1/predictions/next", json=request_body()).status_code
            == 503
        )
        assert empty.get("/api/v1/ena/groups/stable/network").status_code == 503


def test_risk_boundaries():
    thresholds = {"medium": 0.4, "high": 0.65}
    assert risk_level(0.399, thresholds) == "low"
    assert risk_level(0.4, thresholds) == "medium"
    assert risk_level(0.65, thresholds) == "high"
