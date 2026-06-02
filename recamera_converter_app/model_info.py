from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_classes(path: Path | None = None) -> list[str]:
    if path is None:
        path = REPO_ROOT / "data" / "coco80.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model_info(
    *,
    model_name: str,
    task: str = "detection",
    classes: list[str] | None = None,
    description: str | None = None,
    author: str = "local reCamera converter",
    model_id: str = "0",
    model_file: Path | None = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "model_id": str(model_id),
        "model_name": model_name,
        "model_format": "cvimodel",
        "ai_framwork": 6,
        "description": description or f"{model_name} converted locally for Seeed reCamera CV181x",
        "author": author,
        "classes": classes if classes is not None else read_classes(),
        "task": task,
        "target": "cv181x",
    }
    if model_file is not None and model_file.exists():
        info["model_size"] = model_file.stat().st_size
        info["model_md5"] = file_md5(model_file)
    return info


def write_model_info(path: Path, **kwargs: Any) -> dict[str, Any]:
    info = build_model_info(**kwargs)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info
