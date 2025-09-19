"""Case dictionary helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def read_yaml_file(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data
