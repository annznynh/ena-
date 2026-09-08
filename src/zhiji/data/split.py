import json
from pathlib import Path

import numpy as np
import polars as pl

from zhiji.data.download import sha256


def create_student_splits(
    interactions_path: Path,
    output_path: Path,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    ratios = [train_ratio, validation_ratio, test_ratio]
    if min(ratios) <= 0 or not np.isclose(sum(ratios), 1):
        raise ValueError("Split ratios must be positive and sum to 1")
    if output_path.exists():
        manifest = json.loads(output_path.with_suffix(".json").read_text())
        if (
            manifest["input_sha256"] != sha256(interactions_path)
            or manifest["seed"] != seed
            or manifest["ratios"] != ratios
        ):
            raise ValueError("Saved split belongs to different data/configuration")
        return manifest
    ids = (
        pl.scan_parquet(interactions_path)
        .select("student_id")
        .unique()
        .sort("student_id")
        .collect()["student_id"]
        .to_numpy()
    )
    if len(ids) < 7:
        raise ValueError("At least seven students required for nonempty 70/15/15 split")
    ids = np.random.default_rng(seed).permutation(ids)
    a, b = int(len(ids) * train_ratio), int(len(ids) * (train_ratio + validation_ratio))
    frame = pl.DataFrame(
        {
            "student_id": ids,
            "split": ["train"] * a
            + ["validation"] * (b - a)
            + ["test"] * (len(ids) - b),
            "seed": seed,
            "split_version": "split-v1",
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    report = {
        "input_sha256": sha256(interactions_path),
        "seed": seed,
        "ratios": ratios,
        "counts": {"train": a, "validation": b - a, "test": len(ids) - b},
        "split_sha256": sha256(output_path),
        "student_overlap": 0,
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def build_prediction_examples(
    interactions_path: Path,
    splits_path: Path,
    output_path: Path,
    *,
    min_history: int = 5,
    max_history: int = 20,
    future_horizon: int = 3,
) -> dict:
    if future_horizon != 3 or min_history < 1 or max_history < min_history:
        raise ValueError("Invalid history / future horizon")
    frame = pl.read_parquet(interactions_path).sort(["student_id", "event_order"])
    frame = (
        frame.with_columns(
            pl.col("event_id").alias("anchor_event_id"),
            pl.col("event_id").shift(-1).over("student_id").alias("target_event_id"),
            pl.col("event_time").shift(-1).over("student_id").alias("target_time"),
            pl.col("correct_binary")
            .shift(-1)
            .over("student_id")
            .alias("target_correct"),
            (pl.col("event_order") + 1)
            .clip(upper_bound=max_history)
            .alias("history_length"),
            *[
                (1 - pl.col("correct_binary"))
                .shift(-k)
                .over("student_id")
                .alias(f"future_{k}")
                for k in range(1, 4)
            ],
        )
        .with_columns(
            (pl.col("future_1") + pl.col("future_2") + pl.col("future_3")).alias(
                "future3_incorrect_count"
            ),
        )
        .with_columns(
            (pl.col("future3_incorrect_count") >= 2)
            .cast(pl.Int8)
            .alias("future3_difficulty"),
            (1 - pl.col("target_correct")).alias("target_risk"),
            pl.concat_str(["student_id", "anchor_event_id"], separator=":").alias(
                "example_id"
            ),
        )
    )
    # Adjacent tied timestamps cannot satisfy the strict temporal quality gate.
    ties = frame.filter(pl.col("target_time") == pl.col("event_time")).height
    frame = frame.filter(
        (pl.col("history_length") >= min_history)
        & (pl.col("target_time") > pl.col("event_time"))
    )
    frame = frame.select(
        "example_id",
        "student_id",
        "anchor_event_id",
        "target_event_id",
        "history_length",
        "target_correct",
        "target_risk",
        "future3_incorrect_count",
        "future3_difficulty",
    )
    frame = frame.join(
        pl.read_parquet(splits_path).select("student_id", "split"),
        on="student_id",
        validate="m:1",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    report = {
        "examples": frame.height,
        "tied_adjacent_targets_excluded": ties,
        "sha256": sha256(output_path),
        "min_history": min_history,
        "max_history": max_history,
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report
