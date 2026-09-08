import json
from pathlib import Path

import polars as pl

from zhiji.features.tabular import fit_history_thresholds

CODES = [
    "incorrect",
    "hint_used",
    "bottom_hint_used",
    "repeated_attempt",
    "slow_response",
    "frustrated",
    "confused",
    "bored",
    "concentrating",
]


def fit_ena_thresholds(
    train_interactions_path: Path, *, affect_threshold=0.5, slow_response_quantile=0.75
) -> dict:
    if slow_response_quantile != 0.75:
        raise ValueError("This implementation fixes the preregistered P75 threshold")
    return fit_history_thresholds(pl.read_parquet(train_interactions_path)) | {
        "affect_threshold": affect_threshold
    }


def encode(frame: pl.DataFrame, thresholds: dict) -> pl.DataFrame:
    return frame.with_columns(
        (1 - pl.col("correct_binary")).alias("incorrect"),
        (pl.col("hint_count") > 0).cast(pl.Int8).alias("hint_used"),
        pl.col("bottom_hint").alias("bottom_hint_used"),
        (pl.col("attempt_count") > 1).cast(pl.Int8).alias("repeated_attempt"),
        (
            pl.col("ms_first_response").log1p()
            > pl.col("problem_type").replace_strict(
                thresholds["slow_by_type"],
                default=thresholds["slow_global"],
                return_dtype=pl.Float64,
            )
        )
        .fill_null(False)
        .cast(pl.Int8)
        .alias("slow_response"),
        *[
            (pl.col(e + "_conf") >= thresholds["affect_threshold"])
            .fill_null(False)
            .cast(pl.Int8)
            .alias(e)
            for e in ["frustrated", "confused", "bored", "concentrating"]
        ],
    )


def export_ena_input(
    interactions_path: Path,
    splits_path: Path,
    output_csv: Path,
    *,
    thresholds: dict,
    history_fraction=0.70,
    min_events=30,
) -> dict:
    frame = pl.read_parquet(interactions_path).join(
        pl.read_parquet(splits_path).select("student_id", "split"), on="student_id"
    )
    frame = frame.with_columns(pl.len().over("student_id").alias("n")).filter(
        pl.col("n") >= min_events
    )
    frame = frame.with_columns(
        (pl.col("n") * history_fraction).floor().cast(pl.Int64).alias("cutoff")
    )
    outcomes = (
        frame.filter(pl.col("event_order") >= pl.col("cutoff"))
        .group_by("student_id", "split")
        .agg((1 - pl.col("correct_binary")).mean().alias("error_rate"))
    )
    train = outcomes.filter(pl.col("split") == "train")
    low, high = train["error_rate"].quantile(0.25), train["error_rate"].quantile(0.75)
    if low is None or high is None or low >= high:
        raise ValueError(
            "Insufficient distinct training outcome quantiles for ENA groups"
        )
    groups = outcomes.with_columns(
        pl.when(pl.col("error_rate") <= low)
        .then(pl.lit("stable"))
        .when(pl.col("error_rate") >= high)
        .then(pl.lit("difficulty"))
        .otherwise(None)
        .alias("group")
    ).drop_nulls("group")
    history = frame.filter(pl.col("event_order") < pl.col("cutoff")).join(
        groups.select("student_id", "group"), on="student_id"
    )
    history = history.with_columns(
        pl.concat_str(
            [
                pl.col("student_id"),
                pl.coalesce(
                    "assignment_id", "sequence_id", pl.lit("__UNKNOWN_SESSION__")
                ),
            ],
            separator=":",
        ).alias("conversation_id")
    )
    history = encode(history, thresholds).sort(["student_id", "event_order"])
    if (
        min(
            groups.filter(pl.col("group") == g).height for g in ["stable", "difficulty"]
        )
        < 2
    ):
        raise ValueError("ENA requires at least two independent units per group")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    history.select(
        "student_id", "group", "conversation_id", "event_order", *CODES
    ).write_csv(output_csv)
    report = {
        "rows": history.height,
        "units": groups.height,
        "low": low,
        "high": high,
        "unit_counts": {
            g: groups.filter(pl.col("group") == g).height
            for g in ["stable", "difficulty"]
        },
        "code_rates": {c: history[c].mean() for c in CODES},
        "thresholds": thresholds,
        "history_fraction": history_fraction,
        "min_events": min_events,
        "outcome_period_excluded_from_network": True,
        "quantile_fit_split": "train",
    }
    output_csv.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report
