"""Run the preregistered one-factor-at-a-time ENA sensitivities."""

import json
import os
import subprocess
from pathlib import Path

from zhiji.ena.codes import export_ena_input
from zhiji.ena.contract import export_results


def main():
    reports = []
    for back, affect in [(4, 0.5), (2, 0.5), (6, 0.5), (4, 0.33), (4, 0.67)]:
        path = (
            Path("data/ena/ena_input.csv")
            if affect == 0.5
            else Path(f"data/ena/ena_input_{affect}.csv")
        )
        if not path.exists():
            thresholds = json.loads(
                Path("data/features/thresholds.json").read_text()
            ) | {"affect_threshold": affect}
            export_ena_input(
                Path("data/processed/interactions.parquet"),
                Path("data/splits/student_splits.parquet"),
                path,
                thresholds=thresholds,
            )
        existing = list(
            Path("artifacts/ena").glob(f"ena-w{back}-a{affect}-*/manifest.json")
        )
        if not existing:
            subprocess.run(
                [
                    "Rscript",
                    "--vanilla",
                    "analysis/ena/run_ena.R",
                    str(path),
                    str(back),
                    str(affect),
                ],
                check=True,
                env=os.environ.copy(),
            )
        for manifest in Path("artifacts/ena").glob(
            f"ena-w{back}-a{affect}-*/manifest.json"
        ):
            export_results(manifest.parent)
            network = json.loads(
                (manifest.parent / "difference_network.json").read_text()
            )
            selected = [
                e
                for e in network["edges"]
                if (e["source"], e["target"])
                in [
                    ("incorrect", "hint_used"),
                    ("repeated_attempt", "slow_response"),
                    ("frustrated", "confused"),
                ]
            ]
            reports.append(
                {
                    "version": network["analysis_version"],
                    "window": back,
                    "affect": affect,
                    "permutation_p": network["permutation_p"],
                    "key_edges": selected,
                }
            )
    Path("reports/evaluation/ena_sensitivity.json").write_text(
        json.dumps(reports, indent=2)
    )


if __name__ == "__main__":
    main()
