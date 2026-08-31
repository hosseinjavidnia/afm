from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_IMAGE_RE = re.compile(r"C_(?P<session>\d+)_(?P<object>\d+)_(?P<frame>\d+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class Core50Item:
    path: Path
    session: int
    object_id: int
    frame: int
    category: int


def find_core50_root(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    candidates = [root, root / "core50_128x128", root / "CORe50", root / "core50"]
    for candidate in candidates:
        if any(candidate.glob("s*/o*/C_*.png")):
            return candidate
    for candidate in root.rglob("s1"):
        parent = candidate.parent
        if any(parent.glob("s*/o*/C_*.png")):
            return parent
    raise FileNotFoundError(f"Could not locate CORe50 image tree under {root}")


def scan_core50(root: str | Path) -> list[Core50Item]:
    data_root = find_core50_root(root)
    items: list[Core50Item] = []
    for path in sorted(data_root.glob("s*/o*/C_*.png")):
        match = _IMAGE_RE.match(path.name)
        if match is None:
            continue
        session = int(match.group("session"))
        object_id = int(match.group("object"))
        frame = int(match.group("frame"))
        category = (object_id - 1) // 5
        items.append(Core50Item(path=path.resolve(), session=session, object_id=object_id, frame=frame, category=category))
    if not items:
        raise RuntimeError(f"No CORe50 images found under {data_root}")
    return items


def group_by_session(items: Iterable[Core50Item]) -> dict[int, list[Core50Item]]:
    grouped: dict[int, list[Core50Item]] = {}
    for item in items:
        grouped.setdefault(item.session, []).append(item)
    return grouped
