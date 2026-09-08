import json
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from zhiji.api.schemas import PredictionRequest
from zhiji.api.service import ModelBundle, predict_next


def create_app(root: Path | None = None) -> FastAPI:
    root = root or Path(os.environ.get("ZHIJI_ARTIFACT_ROOT", "artifacts"))
    api = FastAPI(title="知迹 · 教师复核原型", version="0.1.0")
    registry = (
        json.loads((root / "registry.json").read_text())
        if (root / "registry.json").exists()
        else {}
    )
    bundles = {}
    for key in ["current_prediction_model", "current_prediction_model_known"]:
        version = registry.get(key)
        if version:
            try:
                bundles[key] = ModelBundle(root / "models" / version)
            except (OSError, ValueError):
                pass
    demo = (
        json.loads((root / "demo.json").read_text())
        if (root / "demo.json").exists()
        else {}
    )
    ena = root / "ena" / registry.get("current_ena_analysis", "__missing__")

    @api.middleware("http")
    async def request_identity(request: Request, call_next):
        request.state.request_id = "req_" + uuid4().hex
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 -- public HTTP boundary must not expose internals
            return JSONResponse(
                {
                    "error": {"code": "INTERNAL_ERROR", "message": "服务内部错误"},
                    "request_id": request.state.request_id,
                },
                status_code=500,
            )
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    def envelope(request: Request, body: dict):
        return body | {"request_id": request.state.request_id}

    @api.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        detail = (
            exc.detail
            if isinstance(exc.detail, dict)
            else {"code": "HTTP_ERROR", "message": str(exc.detail)}
        )
        return JSONResponse(
            envelope(request, {"error": detail}), status_code=exc.status_code
        )

    @api.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        details = [{"loc": list(e["loc"]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(
            envelope(
                request,
                {
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": "请求字段不符合契约",
                        "details": details,
                    }
                },
            ),
            status_code=422,
        )

    def fail(status, code, message):
        raise HTTPException(status, detail={"code": code, "message": message})

    def bundle_for(known=False):
        bundle = bundles.get(
            "current_prediction_model_known" if known else "current_prediction_model"
        )
        if bundle is None:
            fail(503, "MODEL_UNAVAILABLE", "尚未加载模型产物")
        return bundle

    def student_record(student_id):
        if student_id not in demo:
            fail(404, "STUDENT_NOT_FOUND", "演示学生不存在")
        return demo[student_id]

    @api.get("/health")
    def health(request: Request):
        return envelope(
            request,
            {
                "status": "ok",
                "model_loaded": bool(bundles),
                "ena_artifacts_loaded": (ena / "manifest.json").exists(),
                "model_version": registry.get("current_prediction_model"),
            },
        )

    @api.get("/api/v1/models/current")
    def current(request: Request):
        m = bundle_for().manifest
        return envelope(
            request,
            {
                k: m[k]
                for k in [
                    "model_name",
                    "model_version",
                    "target",
                    "created_at",
                    "risk_thresholds",
                    "feature_set",
                    "known_context",
                ]
            },
        )

    @api.post("/api/v1/predictions/next")
    def predict(body: PredictionRequest, request: Request):
        bundle = bundle_for(body.next_context is not None)
        if (
            body.model_version
            and body.model_version != bundle.manifest["model_version"]
        ):
            fail(409, "MODEL_VERSION_CONFLICT", "请求的模型版本与当前版本不同")
        return envelope(request, predict_next(body, bundle))

    @api.get("/api/v1/students")
    def students(request: Request):
        return envelope(
            request,
            {
                "items": [
                    {"student_id": sid, **record["current"]}
                    for sid, record in demo.items()
                ],
                "scope": "匿名测试集演示样本，并非真实班级",
            },
        )

    @api.get("/api/v1/students/{student_id}/risk-timeline")
    def timeline(
        student_id: str, request: Request, limit: int = Query(50, ge=1, le=100)
    ):
        record = student_record(student_id)
        return envelope(
            request,
            {
                "student_id": student_id,
                "items": record["timeline"][-limit:],
                "model_version": record["current"]["model_version"],
            },
        )

    @api.get("/api/v1/students/{student_id}/recent-evidence")
    def recent(
        student_id: str, request: Request, window: int = Query(20, ge=1, le=100)
    ):
        record = student_record(student_id)
        history = record["history"][-window:]
        return envelope(
            request,
            {
                "student_id": student_id,
                "history": history,
                "evidence": record["current"]["evidence"],
                "summary": {
                    "correct_rate": sum(e["correct"] for e in history) / len(history),
                    "hint_rate": sum(e["hint_count"] > 0 for e in history)
                    / len(history),
                    "repeated_attempt_rate": sum(
                        e["attempt_count"] > 1 for e in history
                    )
                    / len(history),
                },
                "warning": record["current"]["warning"],
            },
        )

    def network(group):
        if not (ena / "manifest.json").exists():
            fail(503, "ENA_UNAVAILABLE", "尚未加载正式 ENA 产物")
        path = ena / f"{group}_network.json"
        if not path.exists():
            fail(404, "ARTIFACT_NOT_FOUND", "分析产物不存在")
        return json.loads(path.read_text())

    @api.get("/api/v1/ena/groups/{group}/network")
    def group_network(group: Literal["stable", "difficulty"], request: Request):
        return envelope(request, network(group))

    @api.get("/api/v1/ena/difference")
    def difference(
        request: Request,
        left: Literal["stable", "difficulty"] = "difficulty",
        right: Literal["stable", "difficulty"] = "stable",
    ):
        if left == right:
            fail(400, "INVALID_COMPARISON", "请选择两个不同群体")
        result = network("difference")
        if left == "stable":
            for edge in result["edges"]:
                edge["weight"] = -edge["weight"]
                edge["ci_low"], edge["ci_high"] = -edge["ci_high"], -edge["ci_low"]
            result["left"], result["right"] = left, right
        return envelope(request, result)

    return api


app = create_app()
