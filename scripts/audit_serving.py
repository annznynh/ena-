"""Compare the actual frozen offline features against API history reconstruction."""

import json
from pathlib import Path

import polars as pl

from zhiji.api.schemas import PredictionRequest
from zhiji.api.service import ModelBundle, predict_next


def run():
    registry = json.loads(Path("artifacts/registry.json").read_text())
    examples = pl.read_parquet("data/features/test.parquet").sample(n=100, seed=42)
    ids = examples["student_id"].unique()
    frame = pl.read_parquet("data/processed/interactions.parquet").filter(
        pl.col("student_id").is_in(ids.implode())
    )
    errors = {}
    for known in [False, True]:
        bundle = ModelBundle(
            Path("artifacts/models")
            / registry[
                "current_prediction_model_known"
                if known
                else "current_prediction_model"
            ]
        )
        differences = []
        for row in examples.to_dicts():
            part = frame.filter(pl.col("student_id") == row["student_id"]).sort(
                "event_order"
            )
            order = part.filter(pl.col("event_id") == row["anchor_event_id"])[
                "event_order"
            ].item()
            history = []
            for e in part.filter(pl.col("event_order") <= order).tail(20).to_dicts():
                history.append(
                    {
                        "event_id": e["event_id"],
                        "occurred_at": e["event_time"].isoformat() + "Z",
                        "correct": e["correct_binary"],
                        "attempt_count": e["attempt_count"],
                        "hint_count": e["hint_count"],
                        "bottom_hint": e["bottom_hint"],
                        "skill_ids": e["skill_ids"],
                        "problem_type": e["problem_type"],
                        "original": int(e["original"])
                        if e["original"] in ["0", "1"]
                        else None,
                        "position": float(e["position"]) if e["position"] else None,
                        "ms_first_response": e["ms_first_response"],
                        "tutor_mode": e["tutor_mode"] or "__UNK__",
                        "first_action": e["first_action"] or "__UNK__",
                        "affect": {
                            k: e[k + "_conf"]
                            for k in [
                                "frustrated",
                                "confused",
                                "concentrating",
                                "bored",
                            ]
                        },
                    }
                )
            request = {"student_id": row["student_id"], "history": history}
            if known:
                target = part.filter(
                    pl.col("event_id") == row["target_event_id"]
                ).to_dicts()[0]
                request["next_context"] = {
                    "skill_ids": target["skill_ids"],
                    "problem_type": target["problem_type"],
                }
            online = predict_next(PredictionRequest.model_validate(request), bundle)[
                "correct_probability"
            ]
            offline = float(
                bundle.model.predict_proba(
                    bundle.encoder.transform(
                        pl.DataFrame([row]).to_pandas()[bundle.manifest["columns"]]
                    )
                )[0, 1]
            )
            differences.append(abs(online - offline))
        errors["known" if known else "unknown"] = {
            "examples": len(differences),
            "maximum_difference": max(differences),
        }
    passed = all(r["maximum_difference"] < 1e-6 for r in errors.values())
    report = {"passed": passed, "settings": errors}
    Path("reports/evaluation/serving_parity.json").write_text(
        json.dumps(report, indent=2)
    )
    assert passed, report
    print(report)


if __name__ == "__main__":
    run()
