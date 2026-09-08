from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Affect(StrictModel):
    frustrated: float | None = Field(None, ge=0, le=1)
    confused: float | None = Field(None, ge=0, le=1)
    concentrating: float | None = Field(None, ge=0, le=1)
    bored: float | None = Field(None, ge=0, le=1)


class HistoryEvent(StrictModel):
    event_id: str = Field(min_length=1, max_length=100)
    occurred_at: datetime
    problem_id: str = Field(default="__UNK__", max_length=100)
    skill_ids: list[str] = Field(default_factory=lambda: ["__UNK__"], max_length=50)
    problem_type: str = Field(default="__UNK__", max_length=100)
    correct: int = Field(ge=0, le=1, strict=True)
    attempt_count: int = Field(ge=0, le=32767, strict=True)
    hint_count: int = Field(ge=0, le=32767, strict=True)
    bottom_hint: int = Field(0, ge=0, le=1, strict=True)
    ms_first_response: float | None = Field(None, ge=0, le=1e12)
    affect: Affect = Field(
        default_factory=lambda: Affect(
            frustrated=None, confused=None, concentrating=None, bored=None
        )
    )
    original: int | None = Field(None, ge=0, le=1)
    position: float | None = None
    tutor_mode: str = Field(default="__UNK__", max_length=100)
    first_action: str = Field(default="__UNK__", max_length=100)

    @field_validator("occurred_at")
    @classmethod
    def aware_time(cls, value):
        if value.tzinfo is None:
            raise ValueError("occurred_at requires an explicit timezone")
        return value


class NextContext(StrictModel):
    problem_id: str = Field(default="__UNK__", max_length=100)
    skill_ids: list[str] = Field(default_factory=lambda: ["__UNK__"], max_length=50)
    problem_type: str = Field(default="__UNK__", max_length=100)


class PredictionRequest(StrictModel):
    student_id: str = Field(pattern=r"^stu_[a-zA-Z0-9_-]{1,64}$")
    history: list[HistoryEvent] = Field(min_length=5, max_length=100)
    next_context: NextContext | None = None
    model_version: str | None = None

    @model_validator(mode="after")
    def ordered(self):
        times = [e.occurred_at for e in self.history]
        if times != sorted(times):
            raise ValueError("history times must be nondecreasing")
        if len({e.event_id for e in self.history}) != len(self.history):
            raise ValueError("history event IDs must be unique")
        return self
