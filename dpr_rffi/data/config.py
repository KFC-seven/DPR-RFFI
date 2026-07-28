from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{config_path} must contain a JSON-compatible YAML object."
        ) from exc
