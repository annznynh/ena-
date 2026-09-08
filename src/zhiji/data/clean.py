"""Global event deduplication using Polars lazy queries."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from zhiji.data.schema import AFFECT, OPTIONAL


@dataclass(frozen=True)
class CleaningReport:
    input_rows: int
    output_events: int
    duplicate_rows_collapsed: int
    conflicting_events_removed: int
    invalid_events_removed: int


def build_canonical_interactions(
    part_glob: str, output_path: Path, conflict_path: Path, *, config: dict
) -> CleaningReport:
    source = pl.scan_parquet(part_glob)
    names = source.collect_schema().names()
    source = source.with_columns(
        [
            pl.lit(None, dtype=pl.String).alias(c)
            for c in (OPTIONAL | {"problem_log_id", "problemlogid"}) - set(names)
        ]
    )
    source = source.with_columns(
        pl.coalesce("problem_log_id", "problemlogid").alias("event_id")
    )
    source = source.filter(pl.col("event_id").is_not_null())
    names = source.collect_schema().names()
    non_skill = [
        c
        for c in names
        if c not in {"event_id", "skill", "skill_id", "problem_log_id", "problemlogid"}
    ]
    grouped = (
        source.group_by("event_id")
        .agg(
            [pl.col(c).first() for c in non_skill]
            + [
                pl.col("skill_id").drop_nulls().unique().sort().alias("skill_ids"),
                pl.col("skill").drop_nulls().unique().sort().alias("skill_names"),
                pl.len().alias("source_row_count"),
                pl.any_horizontal([pl.col(c).n_unique() > 1 for c in non_skill]).alias(
                    "conflict"
                ),
            ]
        )
        .collect(engine="streaming")
    )
    conflict_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.filter(pl.col("conflict")).write_parquet(conflict_path)
    conflict_count = grouped.filter(pl.col("conflict")).height
    frame = grouped.filter(~pl.col("conflict"))
    rename = dict(
        zip(
            AFFECT,
            ["frustrated_conf", "confused_conf", "concentrating_conf", "bored_conf"],
        )
    )
    frame = frame.rename(rename).with_columns(
        pl.col("start_time")
        .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
        .alias("event_time"),
        pl.col("correct").cast(pl.Float32, strict=False).alias("correct_raw"),
        *[
            pl.col(c).cast(pl.Float32, strict=False)
            for c in [
                "attempt_count",
                "hint_count",
                "bottom_hint",
                "ms_first_response",
                "overlap_time",
                *rename.values(),
            ]
        ],
    )
    valid = (
        pl.col("user_id").is_not_null()
        & pl.col("problem_id").is_not_null()
        & pl.col("event_time").is_not_null()
        & pl.col("correct_raw").is_between(0, 1)
        & ~pl.col("problem_type")
        .fill_null("__UNK__")
        .is_in(config["exclude_problem_types"])
    )
    for c in ("attempt_count", "hint_count"):
        valid &= pl.col(c).is_between(0, 32767) & (pl.col(c) == pl.col(c).floor())
    valid &= pl.col("bottom_hint").is_null() | pl.col("bottom_hint").is_in([0, 1])
    invalid = frame.filter(~valid.fill_null(False)).height
    frame = frame.filter(valid.fill_null(False))
    users = frame["user_id"].unique().to_list()
    mapping = {
        u: "stu_" + hashlib.sha256(("zhiji-research-v1:" + u).encode()).hexdigest()[:24]
        for u in users
    }
    frame = frame.with_columns(
        pl.col("user_id").replace_strict(mapping).alias("student_id"),
        (pl.col("correct_raw") == 1).cast(pl.Int8).alias("correct_binary"),
        *[pl.col(c).cast(pl.Int16) for c in ("attempt_count", "hint_count")],
        pl.col("bottom_hint").fill_null(0).cast(pl.Int8),
        pl.col("problem_type").fill_null("__UNK__"),
        (
            (pl.col("ms_first_response") < 0)
            | (pl.col("overlap_time") < 0)
            | pl.col("bottom_hint").is_null()
        )
        .fill_null(True)
        .alias("has_data_quality_issue"),
        pl.when(pl.col("ms_first_response") >= 0)
        .then(pl.col("ms_first_response"))
        .otherwise(None),
        pl.when(pl.col("overlap_time") >= 0)
        .then(pl.col("overlap_time"))
        .otherwise(None)
        .alias("overlap_time_ms"),
        pl.when(pl.col("skill_ids").list.len() == 0)
        .then(pl.lit(["__UNK__"]))
        .otherwise(pl.col("skill_ids"))
        .alias("skill_ids"),
    ).sort(["student_id", "event_time", "event_id"])
    frame = frame.with_columns(
        (pl.col("event_id").cum_count().over("student_id") - 1)
        .cast(pl.Int64)
        .alias("event_order")
    )
    keep = [
        "event_id",
        "student_id",
        "event_time",
        "event_order",
        "assignment_id",
        "sequence_id",
        "problem_id",
        "problem_type",
        "skill_ids",
        "skill_names",
        "correct_raw",
        "correct_binary",
        "attempt_count",
        "hint_count",
        "bottom_hint",
        "first_action",
        "ms_first_response",
        "overlap_time_ms",
        *rename.values(),
        "source_row_count",
        "has_data_quality_issue",
        "original",
        "position",
        "tutor_mode",
    ]
    frame = frame.select(keep)
    if not frame.height or frame["event_id"].n_unique() != frame.height:
        raise ValueError("Canonical event uniqueness/empty gate failed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    raw_rows = pl.scan_parquet(part_glob).select(pl.len()).collect().item()
    report = CleaningReport(
        raw_rows,
        frame.height,
        int(grouped["source_row_count"].sum()) - grouped.height,
        conflict_count,
        invalid,
    )
    counts = frame.group_by("student_id").len()
    details = asdict(report) | {
        "eligible_students_prediction": counts.filter(pl.col("len") >= 6).height,
        "eligible_students_ena": counts.filter(pl.col("len") >= 30).height,
        "student_interaction_quantiles": {
            str(q): counts["len"].quantile(q) for q in [0, 0.25, 0.5, 0.75, 1]
        },
    }
    Path("reports/data_quality/cleaning_report.json").write_text(
        json.dumps(details, indent=2)
    )
    return report
