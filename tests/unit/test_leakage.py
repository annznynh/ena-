from datetime import UTC, datetime, timedelta

import polars as pl

from zhiji.features.tabular import history_features


def test_future_changes_do_not_change_anchor_features():
    frame = pl.DataFrame(
        [
            {
                "event_id": f"synthetic_{i}",
                "student_id": "stu_synthetic",
                "event_order": i,
                "event_time": datetime(2012, 1, 1, tzinfo=UTC) + timedelta(minutes=i),
                "skill_ids": ["s"],
                "problem_type": "algebra",
                "original": "1",
                "position": str(i),
                "correct_binary": i % 2,
                "hint_count": i % 3,
                "bottom_hint": 0,
                "attempt_count": 1,
                "ms_first_response": 1000.0,
                "tutor_mode": "tutor",
                "first_action": "answer",
                "frustrated_conf": 0.1,
                "confused_conf": 0.2,
                "concentrating_conf": 0.8,
                "bored_conf": 0.1,
            }
            for i in range(30)
        ]
    )
    frame = frame.with_columns(
        pl.lit("stu_synthetic").alias("student_id"),
        pl.int_range(pl.len()).alias("event_order"),
    )
    thresholds = {"slow_by_type": {}, "slow_global": 9.0}
    expected = history_features(frame, thresholds).head(10)
    changed = frame.with_columns(
        pl.when(pl.col("event_order") >= 10)
        .then(999)
        .otherwise(pl.col("hint_count"))
        .alias("hint_count"),
        pl.when(pl.col("event_order") >= 10)
        .then(1.0)
        .otherwise(pl.col("confused_conf"))
        .alias("confused_conf"),
    )
    assert expected.equals(history_features(changed, thresholds).head(10))
    assert expected.equals(history_features(frame.head(10), thresholds))
    assert not any(c.startswith(("target_", "future")) for c in expected.columns)


def test_missing_skill_representation_is_equivalent():
    import pandas as pd

    from zhiji.features.encoders import TabularEncoder

    encoder = TabularEncoder().fit(pd.DataFrame({"A_skill": ["", "s"]}))
    empty = pl.DataFrame({"skill_ids": [[]]}, schema={"skill_ids": pl.List(pl.String)})
    token = pl.DataFrame({"skill_ids": [["__UNK__"]]})
    for frame in [empty, token]:
        key = frame.select(
            pl.col("skill_ids").list.join("|").replace("__UNK__", "").alias("A_skill")
        ).to_pandas()
        assert encoder.transform(key)[0, 0] == 0.5
