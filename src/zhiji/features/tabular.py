"""History-only rolling features shared by training and inference."""

import json
from pathlib import Path

import polars as pl

EMOTIONS = ["frustrated", "confused", "concentrating", "bored"]


def history_features(
    frame: pl.DataFrame, thresholds: dict, windows: tuple[int, ...] = (3, 5, 10)
) -> pl.DataFrame:
    frame = frame.sort(["student_id", "event_order"])
    frame = frame.with_columns(
        pl.col("skill_ids").list.join("|").replace("__UNK__", "").alias("A_skill"),
        pl.col("problem_type").fill_null("__UNK__").alias("A_type"),
        pl.col("original").cast(pl.Float32, strict=False).alias("A_original"),
        pl.col("position").cast(pl.Float32, strict=False).alias("A_position"),
        pl.col("correct_binary").cast(pl.Float32).alias("A_last_correct"),
        pl.col("hint_count").cast(pl.Float32).alias("B_hint"),
        (pl.col("hint_count") > 0).cast(pl.Float32).alias("B_hint_used"),
        pl.col("bottom_hint").cast(pl.Float32).alias("B_bottom_hint"),
        pl.col("attempt_count").cast(pl.Float32).alias("B_attempt"),
        (pl.col("attempt_count") > 1).cast(pl.Float32).alias("B_repeated"),
        pl.col("ms_first_response").log1p().alias("B_log_response"),
        pl.col("tutor_mode").fill_null("__UNK__").alias("B_mode"),
        pl.col("first_action").fill_null("__UNK__").alias("B_action"),
        (pl.col("event_time").diff().over("student_id").dt.total_seconds()).alias(
            "B_gap_seconds"
        ),
        *[pl.col(e + "_conf").alias("C_" + e) for e in EMOTIONS],
    ).with_columns(
        (
            pl.col("B_log_response")
            > pl.col("problem_type").replace_strict(
                thresholds["slow_by_type"],
                default=thresholds["slow_global"],
                return_dtype=pl.Float64,
            )
        )
        .cast(pl.Float32)
        .alias("B_slow"),
    )
    numeric = [
        "A_last_correct",
        "B_hint",
        "B_hint_used",
        "B_bottom_hint",
        "B_attempt",
        "B_repeated",
        "B_log_response",
        "B_slow",
        *["C_" + e for e in EMOTIONS],
    ]
    expressions = [
        pl.col("A_last_correct")
        .rolling_mean(20, min_samples=1)
        .over("student_id")
        .alias("A_history_correct")
    ]
    for col in numeric:
        for window in windows:
            expressions.append(
                pl.col(col)
                .rolling_mean(window, min_samples=1)
                .over("student_id")
                .alias(f"{col}_mean{window}")
            )
        expressions.append(
            pl.col(col).is_null().cast(pl.Float32).alias(col + "_missing")
        )
    return frame.with_columns(expressions).select(
        "event_id", "student_id", pl.selectors.matches("^[ABC]_")
    )


def fit_history_thresholds(frame: pl.DataFrame) -> dict:
    times = frame.filter(pl.col("ms_first_response").is_not_null()).with_columns(
        pl.col("ms_first_response").log1p().alias("time")
    )
    if not times.height:
        raise ValueError("Training response times unavailable")
    by_type = times.group_by("problem_type").agg(pl.col("time").quantile(0.75))
    return {
        "slow_by_type": dict(zip(by_type["problem_type"], by_type["time"])),
        "slow_global": times["time"].quantile(0.75),
    }


def build_tabular_features(
    interactions_path: Path,
    examples_path: Path,
    output_dir: Path,
    *,
    windows: tuple[int, ...] = (3, 5, 10),
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(interactions_path)
    examples = pl.read_parquet(examples_path)
    train_ids = (
        examples.filter(pl.col("split") == "train").select("student_id").unique()
    )
    thresholds = fit_history_thresholds(
        frame.join(train_ids, on="student_id", how="semi")
    )
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    features = history_features(frame, thresholds, windows)
    output = examples.join(
        features,
        left_on=["anchor_event_id", "student_id"],
        right_on=["event_id", "student_id"],
        validate="1:1",
    )
    context = frame.select(
        pl.col("event_id").alias("target_event_id"),
        pl.col("skill_ids").list.join("|").replace("__UNK__", "").alias("N_skill"),
        pl.col("problem_type").alias("N_type"),
    )
    output = output.join(context, on="target_event_id", validate="m:1")
    columns = [c for c in output.columns if c.startswith(("A_", "B_", "C_", "N_"))]
    schema = {"columns": columns, "history_limit": 20, "threshold_fit_split": "train"}
    (output_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2))
    for split in ["train", "validation", "test"]:
        subset = output.filter(pl.col("split") == split)
        subset.write_parquet(output_dir / f"{split}.parquet")
        print(f"Features {split}: {subset.height:,}", flush=True)
    return schema
