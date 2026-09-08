"""Validate and register artifacts, and publish only aggregate provenance."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

from zhiji.data.download import sha256
from zhiji.ena.contract import export_results


def main():
    registry_path = Path("artifacts/registry.json")
    registry = json.loads(registry_path.read_text())
    analyses = []
    for path in sorted(Path("artifacts/ena").glob("ena-*/manifest.json")):
        export_results(path.parent)
        analyses.append(json.loads(path.read_text()))
        if "-w4-a0.5-" in path.parent.name:
            registry["current_ena_analysis"] = path.parent.name
    registry_path.write_text(json.dumps(registry, indent=2))
    source_hashes = {str(p): sha256(p) for p in sorted(Path("src").rglob("*.py"))}
    source_digest = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()
    ).hexdigest()
    test_path = Path("reports/evaluation/test_metrics.json")
    test = json.loads(test_path.read_text()) if test_path.exists() else {}
    runs = json.loads(Path("reports/evaluation/validation_ablations.json").read_text())
    for run in runs:
        path = Path("artifacts/models") / run["model_version"]
        manifest = json.loads((path / "manifest.json").read_text())
        manifest.update(
            data_manifest_sha256=sha256(Path("data/raw/manifest.json")),
            split_version="split-v1",
            split_sha256=sha256(Path("data/splits/student_splits.parquet")),
            config_sha256=hashlib.sha256(
                json.dumps(manifest["config"], sort_keys=True).encode()
            ).hexdigest(),
            python_version=platform.python_version(),
            validated_source_tree_sha256=source_digest,
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            working_tree_dirty=bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True
                ).strip()
            ),
            preprocessor_sha256=sha256(path / "preprocessor.joblib"),
        )
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (path / "feature_schema.json").write_text(
            json.dumps(
                {
                    "columns": manifest["columns"],
                    "encoded_columns": manifest["encoded_columns"],
                    "missing_skill_key": "empty-string; __UNK__ normalized equivalently",
                },
                indent=2,
            )
        )
        (path / "metrics.json").write_text(
            json.dumps(
                {
                    "validation": manifest["validation"],
                    "test": test.get(run["model_version"]),
                },
                indent=2,
            )
        )
        (path / "calibration.json").write_text(
            json.dumps(
                {
                    "method": "raw probabilities",
                    "validation_curve": manifest["validation"]["calibration_curve"],
                    "validation_ece": manifest["validation"]["ece"],
                },
                indent=2,
            )
        )
    Path("reports/artifact_manifest.json").write_text(
        json.dumps(
            {
                "registry": registry,
                "validated_source_tree_sha256": source_digest,
                "source_hashes": source_hashes,
                "raw_file_manifest": json.loads(
                    Path("data/raw/manifest.json").read_text()
                ),
                "python_lock_sha256": sha256(Path("uv.lock")),
                "r_lock_sha256": sha256(Path("renv.lock")),
                "ena_analyses": analyses,
            },
            indent=2,
        )
    )
    print(
        "Validated and registered",
        len(runs),
        "models and",
        len(analyses),
        "ENA analyses",
    )


if __name__ == "__main__":
    main()
