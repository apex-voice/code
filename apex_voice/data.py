"""Locate the APEX-Voice task packages on disk, downloading them from the Hugging Face Hub if needed.

Resolution order for :func:`resolve_tasks_dir`:

1. an explicit path (a dataset root containing ``tasks/``, a directory of task packages, or a
   single task package);
2. the ``APEX_VOICE_DATA`` environment variable;
3. a previously downloaded copy in the Hugging Face cache (``download=True`` fetches it).
"""

from __future__ import annotations

import os
from pathlib import Path

HF_DATASET_ID = "puneetUMD/APEX-Voice"
HF_REVISION = "main"


def hf_dataset_id() -> str:
    return os.environ.get("APEX_VOICE_HF_REPO", HF_DATASET_ID)


def download_dataset(revision: str | None = None, local_dir: str | Path | None = None) -> Path:
    """Download the dataset snapshot from the Hugging Face Hub and return its root directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install 'apex-voice[hf]' to download the dataset") from e
    path = snapshot_download(
        repo_id=hf_dataset_id(),
        repo_type="dataset",
        revision=revision or HF_REVISION,
        local_dir=str(local_dir) if local_dir else None,
    )
    return Path(path)


def _as_tasks_dir(p: Path) -> Path:
    if (p / "task.yaml").exists():
        return p
    if (p / "tasks").is_dir():
        return p / "tasks"
    if p.is_dir() and any((d / "task.yaml").exists() for d in p.iterdir()):
        return p
    raise FileNotFoundError(f"no APEX-Voice task packages found at {p}")


def resolve_tasks_dir(path: str | Path | None = None, download: bool = False) -> Path:
    """Return a directory of task packages (or a single task package)."""
    if path:
        return _as_tasks_dir(Path(path).expanduser())
    env = os.environ.get("APEX_VOICE_DATA")
    if env:
        return _as_tasks_dir(Path(env).expanduser())
    if download:
        return _as_tasks_dir(download_dataset())
    raise FileNotFoundError(
        "dataset not found: pass --data PATH, set APEX_VOICE_DATA, or run `apex-voice download`."
    )


def list_task_dirs(path: str | Path, only: set[str] | None = None) -> list[Path]:
    """Sorted task-package directories under ``path`` (or ``[path]`` for a single package)."""
    path = _as_tasks_dir(Path(path).expanduser())
    if (path / "task.yaml").exists():
        return [path]
    return sorted(
        d for d in path.iterdir() if (d / "task.yaml").exists() and (only is None or d.name in only)
    )


__all__ = ["HF_DATASET_ID", "hf_dataset_id", "download_dataset", "resolve_tasks_dir", "list_task_dirs"]
