"""Offline demo export using held-out students and frozen models only."""

import hashlib
import json
from pathlib import Path

import polars as pl

from zhiji.api.schemas import PredictionRequest
from zhiji.api.service import ModelBundle, predict_next


def export_demo(root: Path = Path("artifacts"), count=30):
    registry = json.loads((root / "registry.json").read_text())
    bundle = ModelBundle(root / "models" / registry["current_prediction_model"])
    test = (
        pl.read_parquet("data/splits/student_splits.parquet")
        .filter(pl.col("split") == "test")
        .select("student_id")
    )
    frame = pl.read_parquet("data/processed/interactions.parquet").join(
        test, on="student_id", how="semi"
    )
    ids = (
        frame.group_by("student_id")
        .len()
        .filter(pl.col("len") >= 30)
        .sort("student_id")
        .head(count)["student_id"]
    )
    result = {}
    for index, sid in enumerate(ids):
        public = f"stu_demo_{index + 1:03d}"
        records = (
            frame.filter(pl.col("student_id") == sid)
            .sort("event_order")
            .tail(70)
            .to_dicts()
        )
        history = []
        for record in records:
            history.append(
                {
                    "event_id": "evt_"
                    + hashlib.sha256(record["event_id"].encode()).hexdigest()[:16],
                    "occurred_at": record["event_time"].isoformat() + "Z",
                    "correct": record["correct_binary"],
                    "skill_ids": record["skill_ids"],
                    "problem_type": record["problem_type"],
                    "hint_count": record["hint_count"],
                    "attempt_count": record["attempt_count"],
                    "bottom_hint": record["bottom_hint"],
                    "ms_first_response": record["ms_first_response"],
                    "original": int(record["original"])
                    if record["original"] in ["0", "1"]
                    else None,
                    "position": float(record["position"])
                    if record["position"]
                    else None,
                    "tutor_mode": record["tutor_mode"] or "__UNK__",
                    "first_action": record["first_action"] or "__UNK__",
                    "affect": {
                        e: record[e + "_conf"]
                        for e in ["frustrated", "confused", "concentrating", "bored"]
                    },
                }
            )
        timeline = []
        for end in range(5, len(history)):
            # Tied adjacent timestamps were excluded from offline prediction examples too.
            if history[end]["occurred_at"] <= history[end - 1]["occurred_at"]:
                continue
            prediction = predict_next(
                PredictionRequest.model_validate(
                    {"student_id": public, "history": history[max(0, end - 20) : end]}
                ),
                bundle,
            )
            timeline.append(
                {
                    "anchor_event_id": history[end - 1]["event_id"],
                    "occurred_at": history[end - 1]["occurred_at"],
                    "risk_probability": prediction["risk_probability"],
                    "observed_next_correct": history[end]["correct"],
                }
            )
        current = predict_next(
            PredictionRequest.model_validate(
                {"student_id": public, "history": history[-20:]}
            ),
            bundle,
        )
        result[public] = {
            "history": history[-20:],
            "timeline": timeline[-50:],
            "current": current,
        }
    (root / "demo.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    )
    return {
        "students": len(result),
        "source": "held-out test students",
        "timestamps": "Original naive time labelled Z for transport only; not timezone-corrected",
    }
