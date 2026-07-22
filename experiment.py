"""Small, dependency-free experiment recording helpers."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def environment_metadata() -> dict[str, Any]:
    git_commit = _command_output(["git", "rev-parse", "HEAD"])
    git_status = _command_output(["git", "status", "--short"])
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return {
        "command": sys.argv,
        "cwd": str(Path.cwd()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": gpu_names,
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "git_status": git_status,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "nvidia_smi_topology": _command_output(["nvidia-smi", "topo", "-m"]),
    }


def snapshot_sources(destination: Path) -> list[str]:
    """Copy small source/config files, including untracked files, into a run directory."""
    listing = _command_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"])
    if not listing:
        return []
    included = []
    allowed_suffixes = {".py", ".toml", ".md", ".txt", ".yml", ".yaml"}
    for name in listing.splitlines():
        source = Path(name)
        if not source.is_file() or source.suffix not in allowed_suffixes:
            continue
        if source.stat().st_size > 1_000_000:
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        included.append(name)
    return included
