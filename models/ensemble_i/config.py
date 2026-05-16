"""Загрузка ensemble_config.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "ensemble_config.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_CONFIG
    if not p.is_file():
        raise FileNotFoundError(f"Нет конфига ансамбля: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_path(root: Path, rel: str | Path | None) -> Path | None:
    if rel is None or rel == "":
        return None
    q = Path(rel)
    return q if q.is_absolute() else (root / q).resolve()


def save_config(cfg: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
