"""Fail-closed raw-data inspection; never log answers or free text."""

import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

AFFECT = [
    f"Average_confidence({name})"
    for name in ("FRUSTRATED", "CONFUSED", "CONCENTRATING", "BORED")
]
REQUIRED = {
    "user_id",
    "problem_id",
    "start_time",
    "correct",
    "problem_type",
    "attempt_count",
    "hint_count",
    "bottom_hint",
    "skill_id",
    *AFFECT,
}
OPTIONAL = {
    "skill",
    "assignment_id",
    "sequence_id",
    "end_time",
    "first_action",
    "ms_first_response",
    "overlap_time",
    "original",
    "tutor_mode",
    "position",
}
MISSING = {"", "NA", "NaN", "nan", "null", "NULL"}


@dataclass(frozen=True)
class SchemaReport:
    row_count_sampled: int
    columns_found: list[str]
    required_missing: list[str]
    optional_missing: list[str]
    dtype_summary: dict[str, str]
    value_issues: dict[str, int]
    passed: bool
    statistics: dict = field(default_factory=dict)


def validate_raw_schema(csv_path: Path, *, sample_rows: int = 5000) -> SchemaReport:
    if sample_rows < 1:
        raise ValueError("sample_rows must be positive")
    issues: Counter = Counter()
    missing: Counter = Counter()
    labels: Counter = Counter()
    students: Counter = Counter()
    events: Counter = Counter()
    n = 0
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        columns = list(reader.fieldnames or [])
        absent = REQUIRED - set(columns)
        if not {"problem_log_id", "problemlogid"} & set(columns):
            absent.add("problem_log_id|problemlogid")
        try:
            for row in reader:
                n += 1
                if None in row or any(v is None for v in row.values()):
                    issues["malformed_rows"] += 1
                    continue
                for col in REQUIRED | OPTIONAL:
                    missing[col] += row.get(col, "") in MISSING
                first, second = (
                    row.get("problem_log_id", ""),
                    row.get("problemlogid", ""),
                )
                if first not in MISSING and second not in MISSING and first != second:
                    issues["log_id_mismatch"] += 1
                event = first if first not in MISSING else second
                issues["missing_event_id"] += event in MISSING
                events[event] += 1
                students[row.get("user_id", "")] += 1
                labels[row.get("correct", "")] += 1
                issues["open_response"] += row.get("problem_type") == "open_response"
                for col in [
                    "correct",
                    "attempt_count",
                    "hint_count",
                    "bottom_hint",
                    *AFFECT,
                ]:
                    value = row.get(col, "")
                    if value in MISSING:
                        continue
                    try:
                        number = float(value)
                        invalid = not math.isfinite(number) or number < 0
                        if col in ["correct", "bottom_hint", *AFFECT]:
                            invalid |= number > 1
                        if col in ["bottom_hint", "attempt_count", "hint_count"]:
                            invalid |= not number.is_integer()
                        issues[f"invalid:{col}"] += invalid
                    except ValueError:
                        issues[f"invalid:{col}"] += 1
                for col in ("start_time", "end_time"):
                    try:
                        datetime.fromisoformat(row.get(col, ""))
                    except ValueError:
                        issues[f"invalid:{col}"] += 1
                if n >= sample_rows:
                    break
        except (csv.Error, UnicodeError) as exc:
            raise ValueError(
                f"CSV parse failure near physical line {reader.line_num}"
            ) from exc
    fatal = issues["log_id_mismatch"] or issues["malformed_rows"]
    fatal |= any(issues[f"invalid:{col}"] for col in AFFECT)
    distribution = Counter(students.values())
    return SchemaReport(
        n,
        columns,
        sorted(absent),
        sorted(OPTIONAL - set(columns)),
        {c: "raw UTF-8 string; validated before conversion" for c in columns},
        dict(issues),
        bool(n and not absent and not fatal),
        {
            "encoding": "utf-8-sig",
            "column_count": len(columns),
            "correct_values": dict(labels),
            "missing_ratios": {c: v / n for c, v in missing.items()} if n else {},
            "student_interaction_distribution_sample_only": dict(distribution),
            "duplicate_row_ratio_sample_only": (n - len(events)) / n if n else 0,
            "scope": "first rows only; not full-population statistics",
        },
    )
