#!/usr/bin/env python3
"""Lightweight static checks for a public AFM source release."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "pyproject.toml",
    "CITATION.cff",
    "LICENSE",
    "afmvision/afm/persistent_assimilation.py",
    "afmvision/afm/functional_shield.py",
    "afmvision/afm/trainer.py",
    "examples/quickstart.sh",
    "tests/test_core_math.py",
]

FORBIDDEN_TEXT = [
    "/home/",
    "~/afm_compatibility",
    "#SBATCH --nodelist=",
    "#SBATCH --mail-user=",
]

FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".pyc"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    failures: list[str] = []

    for rel in REQUIRED:
        if not (ROOT / rel).is_file():
            failures.append(f"missing required file: {rel}")

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if any(part in {".git", ".pytest_cache", "__pycache__"} for part in path.parts):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden generated/binary artifact: {path.relative_to(ROOT)}")
        if path.stat().st_size > 5 * 1024 * 1024:
            failures.append(f"unexpected >5 MiB source-release file: {path.relative_to(ROOT)}")
        if path.suffix.lower() in {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".cff", ".sh"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for needle in FORBIDDEN_TEXT:
                if needle in text:
                    failures.append(f"private/HPC-specific text {needle!r} in {path.relative_to(ROOT)}")

    if failures:
        print("PUBLIC RELEASE CHECK: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("PUBLIC RELEASE CHECK: PASS")
    print(f"root: {ROOT}")
    print(f"README sha256: {sha256(ROOT / 'README.md')}")


if __name__ == "__main__":
    main()
