"""Documented pipeline commands; each stage enforces its prerequisites."""

import json
from dataclasses import asdict
from pathlib import Path

import typer

from zhiji.data.clean import build_canonical_interactions
from zhiji.data.download import download_dataset
from zhiji.data.ingest import USE_COLUMNS, ingest_csv_to_parquet
from zhiji.data.schema import validate_raw_schema
from zhiji.data.split import build_prediction_examples, create_student_splits
from zhiji.settings import load_config

app = typer.Typer(no_args_is_help=True)
data = typer.Typer(no_args_is_help=True)
app.add_typer(data, name="data")
train = typer.Typer(no_args_is_help=True)
ena = typer.Typer(no_args_is_help=True)
app.add_typer(train, name="train")
app.add_typer(ena, name="ena")


def settings(config):
    return load_config(config)


@data.command()
def download(config: Path = Path("configs/debug.yaml")):
    c = settings(config)["data"]
    typer.echo(
        download_dataset(c["dataset_handle"], c["file_name"], Path(c["raw_dir"]))
    )


@data.command()
def validate(config: Path = Path("configs/debug.yaml")):
    c = settings(config)["data"]
    result = validate_raw_schema(
        Path(c["raw_dir"]) / c["file_name"], sample_rows=c["sample_rows"]
    )
    out = Path("reports/data_quality/sample_schema.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2))
    typer.echo(f"Sample schema passed: {result.passed}")
    if not result.passed:
        raise typer.Exit(4)


@data.command()
def ingest(config: Path = Path("configs/full.yaml")):
    c = settings(config)["data"]
    result = ingest_csv_to_parquet(
        Path(c["raw_dir"]) / c["file_name"],
        Path(c["interim_dir"]) / "raw_parts",
        columns=USE_COLUMNS,
        chunk_size=c["chunk_size"],
    )
    typer.echo(f"Imported {result.raw_rows} rows")


@data.command()
def clean(config: Path = Path("configs/full.yaml")):
    c = settings(config)["data"]
    parts = Path(c["interim_dir"]) / "raw_parts"
    if not (parts / "_SUCCESS.json").exists():
        raise typer.BadParameter("Import completion gate missing")
    typer.echo(
        build_canonical_interactions(
            str(parts / "*.parquet"),
            Path(c["processed_path"]),
            Path("data/processed/conflicting_events.parquet"),
            config=c,
        )
    )


@data.command()
def split(config: Path = Path("configs/full.yaml")):
    c = settings(config)
    options = {k: v for k, v in c["split"].items() if k != "version"}
    typer.echo(
        create_student_splits(
            Path(c["data"]["processed_path"]),
            Path("data/splits/student_splits.parquet"),
            **options,
        )
    )


@data.command("build-examples")
def examples(config: Path = Path("configs/full.yaml")):
    c = settings(config)
    typer.echo(
        build_prediction_examples(
            Path(c["data"]["processed_path"]),
            Path("data/splits/student_splits.parquet"),
            Path("data/processed/prediction_examples.parquet"),
            min_history=c["data"]["min_history_prediction"],
            max_history=c["features"]["sequence_max_length"],
        )
    )


@data.command("features")
def features(config: Path = Path("configs/full.yaml")):
    from zhiji.features.tabular import build_tabular_features

    c = settings(config)
    typer.echo(
        build_tabular_features(
            Path(c["data"]["processed_path"]),
            Path("data/processed/prediction_examples.parquet"),
            Path("data/features"),
            windows=tuple(c["features"]["rolling_windows"]),
        )
    )


@train.command("baseline")
def baseline(config: Path = Path("configs/full.yaml")):
    from zhiji.models.baseline import train_baselines

    settings(config)
    train_baselines(Path("data/features"), Path("artifacts/models/baselines-v2"))


@train.command("xgb")
def xgb(config: Path = Path("configs/full.yaml")):
    from zhiji.models.xgb import train_ablations

    train_ablations(
        Path("data/features"),
        Path("artifacts/models"),
        config=settings(config)["xgboost"],
    )


@app.command("evaluate")
def evaluate(
    model: str = "xgb", split: str = "test", config: Path = Path("configs/full.yaml")
):
    from zhiji.evaluation.report import evaluate_frozen_models

    settings(config)
    if model != "xgb" or split != "test":
        raise typer.BadParameter(
            "Final evaluation freezes all baselines and XGBoost variants on test"
        )
    evaluate_frozen_models(Path("data/features"), Path("artifacts"))


@ena.command("export")
def ena_export(config: Path = Path("configs/full.yaml")):
    from zhiji.ena.codes import export_ena_input

    c = settings(config)
    thresholds = json.loads(Path("data/features/thresholds.json").read_text()) | {
        "affect_threshold": c["ena"]["affect_threshold"]
    }
    typer.echo(
        export_ena_input(
            Path(c["data"]["processed_path"]),
            Path("data/splits/student_splits.parquet"),
            Path("data/ena/ena_input.csv"),
            thresholds=thresholds,
            history_fraction=c["ena"]["history_fraction"],
            min_events=c["data"]["min_events_ena"],
        )
    )


@ena.command("validate-artifacts")
def ena_validate(config: Path = Path("configs/full.yaml")):
    from zhiji.ena.contract import export_results

    settings(config)
    paths = list(Path("artifacts/ena").glob("ena-*/manifest.json"))
    if not paths:
        raise typer.BadParameter("No frozen ENA analyses")
    for path in paths:
        typer.echo(export_results(path.parent))


@app.command("build-demo")
def build_demo():
    from zhiji.data.demo import export_demo

    typer.echo(export_demo())


@app.command("demo")
def demo():
    import subprocess
    import sys

    api = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "zhiji.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
    )
    ui = None
    try:
        ui = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app/Home.py",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                "8501",
                "--server.headless",
                "true",
            ]
        )
        ui.wait()
    finally:
        if ui is not None and ui.poll() is None:
            ui.terminate()
            ui.wait()
        if api.poll() is None:
            api.terminate()
            api.wait()


def main():
    """Map domain failures to the documented process exit codes."""
    import sys

    import yaml

    try:
        app()
    except FileNotFoundError as exc:
        typer.echo(f"Required input is missing: {exc}", err=True)
        sys.exit(3)
    except (FileExistsError, yaml.YAMLError) as exc:
        typer.echo(f"Configuration or existing artifact conflict: {exc}", err=True)
        sys.exit(2)
    except ValueError as exc:
        typer.echo(f"Data contract failed: {exc}", err=True)
        sys.exit(4)
    except RuntimeError as exc:
        typer.echo(f"Training/evaluation failed: {exc}", err=True)
        sys.exit(5)


if __name__ == "__main__":
    main()
