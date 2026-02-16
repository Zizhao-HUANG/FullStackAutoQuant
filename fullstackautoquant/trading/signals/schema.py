from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import jsonschema


def _resolve_schema_path() -> Path:
    script_root = Path(__file__).resolve().parent
    search_roots = [script_root]
    search_roots.extend(script_root.parents[:5])
    seen: set[Path] = set()
    for base in search_roots:
        if base in seen:
            continue
        seen.add(base)
        candidate = base / "configs" / "schema" / "trading.schema.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
                "Cannot locate trading.schema.json. Ensure configs/schema/trading.schema.json exists in the repository."
    )


SCHEMA_PATH = _resolve_schema_path()


def load_schema() -> Dict[str, Any]:
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    return json.loads(text)


def validate_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    schema = load_schema()
    jsonschema.validate(payload, schema)
    return payload


__all__ = ["validate_config"]

