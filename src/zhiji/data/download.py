"""Acquire the documented dataset, preserving existing local bytes."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


class DatasetIntegrityError(ValueError):
    pass


class DatasetFileNotFoundError(FileNotFoundError):
    pass


class AuthenticationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(
    dataset_handle: str, file_name: str, output_dir: Path, *, force: bool = False
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / file_name
    existing = Path(file_name)
    source = "kagglehub"
    if not destination.exists() and existing.is_file() and not force:
        destination.symlink_to(existing.resolve())
        source = "user-provided local file; upstream version unverified"
    elif force or not destination.exists():
        import kagglehub

        downloaded = Path(
            kagglehub.dataset_download(
                dataset_handle,
                path=file_name,
                output_dir=str(output_dir),
                force_download=force,
            )
        )
        destination = downloaded if downloaded.is_file() else downloaded / file_name
    else:
        source = "existing local file; upstream version unverified"
    if not destination.is_file():
        raise DatasetFileNotFoundError(str(destination))
    if not destination.stat().st_size:
        raise DatasetIntegrityError("Dataset is empty")
    manifest = {
        "dataset_handle": dataset_handle,
        "file_name": file_name,
        "sha256": sha256(destination),
        "size_bytes": destination.stat().st_size,
        "registered_at": datetime.now(UTC).isoformat(),
        "dataset_version": None,
        "source": source,
        "publication_ready": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return destination.resolve()
