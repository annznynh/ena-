import polars as pl

from zhiji.data.clean import build_canonical_interactions
from zhiji.data.schema import REQUIRED


def test_global_dedup_and_conflict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports/data_quality").mkdir(parents=True)
    row = {c: "0" for c in REQUIRED}
    row.update(
        problem_log_id="1",
        problemlogid="1",
        user_id="synthetic",
        start_time="2012-01-01 00:00:00",
        problem_type="algebra",
        skill_id="a",
        skill="A",
        correct="0.25",
    )
    rows = [
        row,
        row | {"skill_id": "b", "skill": "B"},
        row | {"problem_log_id": "2", "problemlogid": "2"},
        row | {"problem_log_id": "2", "problemlogid": "2", "correct": "1"},
    ]
    pl.DataFrame(rows[:1]).write_parquet(tmp_path / "part-0.parquet")
    pl.DataFrame(rows[1:]).write_parquet(tmp_path / "part-1.parquet")
    report = build_canonical_interactions(
        str(tmp_path / "part-*.parquet"),
        tmp_path / "out.parquet",
        tmp_path / "conflict.parquet",
        config={"exclude_problem_types": ["open_response"]},
    )
    out = pl.read_parquet(tmp_path / "out.parquet")
    assert report.conflicting_events_removed == 1
    assert out.height == 1
    assert out["skill_ids"][0].to_list() == ["a", "b"]
    assert out["correct_binary"][0] == 0
    assert "user_id" not in out.columns
