from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from zhiji.data.split import build_prediction_examples, create_student_splits


def fixture(path):
    rows = []
    for student in range(20):
        for event in range(12):
            rows.append(
                {
                    "student_id": f"stu_synthetic_{student}",
                    "event_id": f"e{student}_{event}",
                    "event_order": event,
                    "event_time": datetime(2012, 1, 1, tzinfo=UTC)
                    + timedelta(minutes=event),
                    "correct_binary": event % 2,
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def test_split_reproducible_and_immutable(tmp_path):
    source = tmp_path / "events.parquet"
    fixture(source)
    first, second = tmp_path / "split1.parquet", tmp_path / "split2.parquet"
    create_student_splits(source, first)
    create_student_splits(source, second)
    a, b = pl.read_parquet(first), pl.read_parquet(second)
    assert a.equals(b)
    assert a["student_id"].n_unique() == a.height
    assert set(a["split"]) == {"train", "validation", "test"}
    create_student_splits(source, first)
    with pytest.raises(ValueError):
        create_student_splits(source, first, seed=43)


def test_examples_future_labels_and_ties(tmp_path):
    source = tmp_path / "events.parquet"
    fixture(source)
    splits = tmp_path / "splits.parquet"
    create_student_splits(source, splits)
    dest = tmp_path / "examples.parquet"
    build_prediction_examples(source, splits, dest)
    frame = pl.read_parquet(dest)
    assert frame.height == 20 * 7
    assert frame["history_length"].min() == 5
    assert (
        frame.filter(pl.col("anchor_event_id") == "e0_4")["target_correct"].item() == 1
    )
    assert (
        frame.filter(pl.col("anchor_event_id") == "e0_10")["future3_difficulty"].item()
        is None
    )
