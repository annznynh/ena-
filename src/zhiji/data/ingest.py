"""Bounded-memory CSV import with strict parsing and an audited completion gate."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from zhiji.data.schema import AFFECT, OPTIONAL, REQUIRED, validate_raw_schema

USE_COLUMNS = sorted(REQUIRED | OPTIONAL | {"problem_log_id", "problemlogid"})


@dataclass(frozen=True)
class IngestionReport:
    raw_rows: int
    chunks_written: int
    parse_failures: int
    output_paths: list[Path]


def ingest_csv_to_parquet(
    csv_path: Path, output_dir: Path, *, columns: list[str], chunk_size: int = 250000
) -> IngestionReport:
    if not validate_raw_schema(csv_path).passed:
        raise ValueError("Sample schema gate failed")
    if output_dir.exists() and any(output_dir.glob("*.parquet")):
        raise FileExistsError(
            "Use a fresh import directory to avoid mixing input versions"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    paths = []
    issues: Counter = Counter()
    correct: Counter = Counter()
    missing: Counter = Counter()
    affect_ranges = {c: [1.0, 0.0] for c in AFFECT}
    report_path = Path("reports/data_quality/schema_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for i, chunk in enumerate(
            pd.read_csv(
                csv_path,
                usecols=lambda c: c in columns,
                dtype="string",
                chunksize=chunk_size,
                encoding="utf-8-sig",
                on_bad_lines="error",
            )
        ):
            total += len(chunk)
            for col in chunk:
                missing[col] += int(chunk[col].isna().sum())
            if {"problem_log_id", "problemlogid"} <= set(chunk):
                both = chunk.problem_log_id.notna() & chunk.problemlogid.notna()
                count = int((both & chunk.problem_log_id.ne(chunk.problemlogid)).sum())
                if count:
                    raise ValueError(
                        f"Log ID mismatch: {count} rows in chunk {i}; manual review required"
                    )
            correct.update(chunk.correct.fillna("__MISSING__").value_counts().to_dict())
            issues["open_response"] += int(chunk.problem_type.eq("open_response").sum())
            for col in ("start_time", "end_time"):
                if col in chunk:
                    issues[f"invalid:{col}"] += int(
                        pd.to_datetime(chunk[col], format="mixed", errors="coerce")
                        .isna()
                        .sum()
                    )
            for col in AFFECT:
                values = pd.to_numeric(chunk[col], errors="coerce")
                invalid = values.notna() & ~values.between(0, 1)
                invalid |= chunk[col].notna() & values.isna()
                if invalid.any():
                    raise ValueError(f"Affect contract failed in chunk {i}: {col}")
                affect_ranges[col][0] = min(affect_ranges[col][0], float(values.min()))
                affect_ranges[col][1] = max(affect_ranges[col][1], float(values.max()))
            path = output_dir / f"part-{i:05d}.parquet"
            chunk.to_parquet(path, index=False)
            paths.append(path)
            print(
                f"Imported chunk {i}: {len(chunk):,} rows; cumulative {total:,}",
                flush=True,
            )
    except Exception as exc:
        report_path.write_text(
            json.dumps(
                {
                    "passed": False,
                    "rows_read": total,
                    "error": str(exc),
                    "completed_chunks": len(paths),
                },
                indent=2,
            )
        )
        raise
    report = {
        "passed": True,
        "raw_rows": total,
        "chunks": len(paths),
        "encoding": "utf-8-sig",
        "correct_values": dict(correct),
        "missing_ratios": {c: v / total for c, v in missing.items()},
        "value_issues": dict(issues),
        "affect_ranges": affect_ranges,
        "parse_failures": 0,
        "log_id_mismatches": 0,
    }
    report_path.write_text(json.dumps(report, indent=2))
    (output_dir / "_SUCCESS.json").write_text(json.dumps(report, indent=2))
    return IngestionReport(total, len(paths), 0, paths)
