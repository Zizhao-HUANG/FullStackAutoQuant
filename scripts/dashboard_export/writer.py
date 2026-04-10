"""File I/O helpers for JSON output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(data: Any, path: Path, pretty: bool = False) -> None:
    """Write JSON data to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if pretty else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
