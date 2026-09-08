import csv
from pathlib import Path

from zhiji.data.schema import AFFECT, REQUIRED, validate_raw_schema


def write_sample(path, **changes):
    row = {c: "0" for c in REQUIRED}
    row.update(
        problem_log_id="1",
        problemlogid="1",
        start_time="2012-01-01 00:00:00",
        problem_type="algebra",
        user_id="synthetic",
        skill_id="skill",
    )
    row.update(changes)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_valid_partial_credit(tmp_path: Path):
    path = tmp_path / "input.csv"
    write_sample(path, correct="0.25")
    assert validate_raw_schema(path).passed


def test_id_mismatch_stops(tmp_path: Path):
    path = tmp_path / "input.csv"
    write_sample(path, problemlogid="2")
    assert not validate_raw_schema(path).passed


def test_affect_range_stops(tmp_path: Path):
    path = tmp_path / "input.csv"
    write_sample(path, **{AFFECT[0]: "1.1"})
    assert not validate_raw_schema(path).passed


def test_missing_columns_stops(tmp_path: Path):
    path = tmp_path / "input.csv"
    path.write_text("user_id\nsynthetic\n")
    assert not validate_raw_schema(path).passed
