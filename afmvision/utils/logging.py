from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .io import ensure_dir


class JSONLLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        ensure_dir(self.path.parent)

    def log(self, event: str, **fields: Any) -> None:
        row = {"wall_time": time.time(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
