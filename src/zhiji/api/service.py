import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from xgboost import DMatrix, XGBClassifier

from zhiji.api.schemas import PredictionRequest
from zhiji.data.download import sha256
from zhiji.features.tabular import history_features

WARNING = "该结果仅供教师复核，不代表确定结果或因果诊断。情感字段为系统预测置信度。"


class ModelBundle:
    def __init__(self, path: Path):
        self.manifest = json.loads((path / "manifest.json").read_text())
        if sha256(path / "model.json") != self.manifest["model_sha256"]:
            raise ValueError("Model checksum mismatch")
        self.model = XGBClassifier()
        self.model.load_model(path / "model.json")
        self.encoder = joblib.load(path / "preprocessor.joblib")
        self.thresholds = json.loads((path / "thresholds.json").read_text())


def risk_level(risk: float, thresholds: dict) -> str:
    if risk >= thresholds["high"]:
        return "high"
    return "medium" if risk >= thresholds["medium"] else "low"


def events_frame(request: PredictionRequest) -> pl.DataFrame:
    rows = []
    for i, e in enumerate(request.history[-20:]):
        rows.append(
            {
                "event_id": e.event_id,
                "student_id": request.student_id,
                "event_order": i,
                "event_time": e.occurred_at.astimezone(UTC),
                "skill_ids": sorted(set(e.skill_ids)) or ["__UNK__"],
                "problem_type": e.problem_type,
                "original": e.original,
                "position": e.position,
                "correct_binary": e.correct,
                "hint_count": e.hint_count,
                "bottom_hint": e.bottom_hint,
                "attempt_count": e.attempt_count,
                "ms_first_response": e.ms_first_response,
                "tutor_mode": e.tutor_mode,
                "first_action": e.first_action,
                **{
                    name + "_conf": value
                    for name, value in e.affect.model_dump().items()
                },
            }
        )
    return pl.DataFrame(rows).with_columns(
        *[
            pl.col(c).cast(pl.Float32)
            for c in [
                "ms_first_response",
                "frustrated_conf",
                "confused_conf",
                "bored_conf",
                "concentrating_conf",
            ]
        ]
    )


def predict_next(request: PredictionRequest, bundle: ModelBundle) -> dict:
    frame = events_frame(request)
    features = history_features(frame, bundle.thresholds).tail(1).to_pandas()
    if bundle.manifest["known_context"]:
        if request.next_context is None:
            raise ValueError("Known-context model requires next_context")
        features["N_skill"] = "|".join(
            sorted(set(request.next_context.skill_ids))
        ).replace("__UNK__", "")
        features["N_type"] = request.next_context.problem_type
    values = bundle.encoder.transform(features[bundle.manifest["columns"]])
    correct = float(bundle.model.predict_proba(values)[0, 1])
    recent = request.history[-5:]
    evidence: list[dict] = [
        {
            "kind": "recent_fact",
            "label": "近期答错",
            "value": f"最近5题中{sum(1 - e.correct for e in recent)}题答错",
        },
        {
            "kind": "recent_fact",
            "label": "提示使用",
            "value": f"最近5题共使用{sum(e.hint_count for e in recent)}次提示",
        },
        {
            "kind": "recent_fact",
            "label": "重复尝试",
            "value": f"最近5题中{sum(e.attempt_count > 1 for e in recent)}题重复尝试",
        },
    ]
    contributions = bundle.model.get_booster().predict(
        DMatrix(values),
        pred_contribs=True,
        iteration_range=(0, bundle.manifest["best_iteration"] + 1),
    )[0, :-1]
    labels = {
        "A": "历史答题与知识点",
        "B": "近期交互行为",
        "C": "情感预测置信度",
        "N": "下一题上下文",
    }
    for index in np.argsort(np.abs(contributions))[-3:][::-1]:
        name = bundle.encoder.output_names_[index]
        evidence.append(
            {
                "kind": "model_contribution",
                "label": labels[name[0]],
                "value": f"{labels[name[0]]}相关特征推动本次风险估计{'下降' if contributions[index] > 0 else '上升'}",
                "contribution_to_correct_log_odds": float(contributions[index]),
            }
        )
    return {
        "student_id": request.student_id,
        "correct_probability": correct,
        "risk_probability": 1 - correct,
        "risk_level": risk_level(1 - correct, bundle.manifest["risk_thresholds"]),
        "evidence": evidence,
        "warning": WARNING,
        "model_version": bundle.manifest["model_version"],
        "generated_at": datetime.now(UTC).isoformat(),
    }
