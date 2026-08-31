from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Clear10Item:
    path: Path
    label: int
    bucket: int
    relative_path: str


def _old_split_names(split: str | None) -> tuple[str, ...]:
    if split == "train":
        return ("training_folder", "train_folder", "train")
    if split == "test":
        return ("test_folder", "testing_folder", "test")
    return ("training_folder", "train_folder", "train", "test_folder", "testing_folder", "test")


def _new_split_aliases(split: str) -> tuple[str, ...]:
    if split == "train":
        return ("train", "train_image_only", "training", "training_image_only")
    if split == "test":
        return ("test", "test_image_only", "testing", "testing_image_only")
    raise ValueError(f"Unknown CLEAR-10 split: {split}")


def _is_new_split_folder(folder: Path) -> bool:
    return (folder / "labeled_metadata.json").is_file() and (folder / "class_names.txt").is_file()


def _has_new_split(candidate: Path, split: str) -> bool:
    return any(_is_new_split_folder(candidate / alias) for alias in _new_split_aliases(split))


def _find_dataset_root(root: str | Path, split: str | None = None) -> Path:
    """Locate the legacy CLEAR root. New-format splits are found separately."""
    root = Path(root).expanduser().resolve()
    required_names = _old_split_names(split)
    candidates = [root, root / "raw", root / "clear10", root / "CLEAR10"]

    # Retain compatibility with the older NeurIPS/Avalanche filelist layout.
    candidates.extend(sorted(path.parent for path in root.rglob("class_names.txt")))
    for name in required_names:
        candidates.extend(sorted(path.parent for path in root.rglob(name) if path.is_dir()))

    seen: set[Path] = set()
    valid: list[tuple[int, Path]] = []
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except FileNotFoundError:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if any((candidate / name).is_dir() for name in required_names):
            old_priority = 0 if (candidate / "class_names.txt").is_file() else 1
            valid.append((old_priority, candidate))

    if valid:
        return sorted(valid, key=lambda item: (item[0], len(item[1].parts), str(item[1])))[0][1]
    raise FileNotFoundError(
        f"Could not locate an extracted legacy CLEAR-10 {split or ''} tree under {root}. "
        "Expected training_folder|test_folder with filelists."
    )


def _find_new_split_folder(root: Path, split: str) -> Path | None:
    """Find a current CLEAR split, including the official train_image_only name."""
    root = root.expanduser().resolve()
    aliases = _new_split_aliases(split)
    alias_set = {name.casefold() for name in aliases}

    candidates: list[tuple[int, int, str, Path]] = []
    for alias in aliases:
        direct = root / alias
        if _is_new_split_folder(direct):
            candidates.append((0, len(direct.parts), str(direct), direct))

    for metadata in sorted(root.rglob("labeled_metadata.json")):
        folder = metadata.parent
        if not _is_new_split_folder(folder):
            continue
        name = folder.name.casefold()
        parts = {part.casefold() for part in folder.parts}
        if name in alias_set:
            priority = 0
        elif parts.intersection(alias_set):
            priority = 1
        else:
            # Do not guess between train and test when both archives are present.
            continue
        candidates.append((priority, len(folder.parts), str(folder), folder))

    if not candidates:
        return None
    return sorted(candidates)[0][3]


def _find_split_folder(dataset_root: Path, split: str) -> Path:
    if split == "train":
        names = ("training_folder", "train_folder", "train")
    elif split == "test":
        names = ("test_folder", "testing_folder", "test")
    else:
        raise ValueError(f"Unknown CLEAR-10 split: {split}")
    for name in names:
        candidate = dataset_root / name
        if candidate.is_dir():
            return candidate
    # The official archives may introduce a single wrapper directory.
    for name in names:
        matches = sorted(path for path in dataset_root.rglob(name) if path.is_dir())
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not locate CLEAR-10 {split} folder below {dataset_root}")


def _bucket_indices(split_folder: Path) -> list[int]:
    candidates = [
        split_folder / "bucket_indices.json",
        split_folder.parent / "bucket_indices.json",
    ]
    for path in candidates:
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw = raw.get("bucket_indices", raw.get("buckets", []))
            return [int(value) for value in raw]
    filelist_root = split_folder / "filelists"
    if filelist_root.is_dir():
        indices = sorted(int(path.name) for path in filelist_root.iterdir() if path.is_dir() and path.name.isdigit())
        if indices:
            return indices
    raise FileNotFoundError(f"No bucket_indices.json or numeric filelist directories below {split_folder}")


def _filelist_path(split_folder: Path, bucket: int) -> Path:
    candidates = [
        split_folder / "filelists" / str(bucket) / "all.txt",
        split_folder / str(bucket) / "all.txt",
        split_folder / "filelists" / f"bucket_{bucket}" / "all.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(split_folder.rglob(f"*/{bucket}/all.txt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No CLEAR-10 filelist for bucket {bucket} below {split_folder}")


def _read_filelist(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            relative, target = line.rsplit(maxsplit=1)
            rows.append((relative, int(target)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid CLEAR filelist row {path}:{line_number}: {raw!r}") from exc
    if not rows:
        raise RuntimeError(f"Empty CLEAR filelist: {path}")
    return rows


def _read_class_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if not names:
        raise RuntimeError(f"Empty CLEAR class_names.txt: {path}")
    return names


def class_names(root: str | Path) -> list[str]:
    root = Path(root).expanduser().resolve()
    candidates = [
        root / "class_names.txt",
        root / "raw" / "class_names.txt",
        root / "train" / "class_names.txt",
        root / "test" / "class_names.txt",
    ]
    candidates.extend(sorted(root.rglob("class_names.txt")))
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    return [] if path is None else _read_class_names(path)


def _numeric_bucket_key(value: Any) -> tuple[int, str]:
    try:
        return int(value), str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CLEAR bucket key is not an integer: {value!r}") from exc


def _resolve_new_image(
    input_root: Path,
    dataset_root: Path,
    split_folder: Path,
    split: str,
    image_path: str,
) -> Path:
    relative = Path(image_path)
    candidates = [
        split_folder / relative,
        dataset_root / split / relative,
        dataset_root / relative,
        input_root / split / relative,
        input_root / "raw" / split / relative,
        input_root / relative,
    ]
    image = next((candidate for candidate in candidates if candidate.is_file()), None)
    if image is not None:
        return image.resolve()
    basename_matches = list(dataset_root.rglob(relative.name))
    if len(basename_matches) == 1:
        return basename_matches[0].resolve()
    raise FileNotFoundError(
        f"CLEAR-10 image does not exist for split={split}: {image_path}; tried {candidates}"
    )


def _scan_new_format(input_root: Path, dataset_root: Path, split: str, split_folder: Path) -> list[Clear10Item]:
    metadata_path = split_folder / "labeled_metadata.json"
    class_path = split_folder / "class_names.txt"
    if not class_path.is_file():
        candidates = sorted(dataset_root.rglob("class_names.txt"))
        class_path = candidates[0] if candidates else class_path
    if not metadata_path.is_file() or not class_path.is_file():
        raise FileNotFoundError(
            f"Incomplete new-format CLEAR-10 split at {split_folder}: "
            f"expected labeled_metadata.json and class_names.txt"
        )

    names = _read_class_names(class_path)
    labeled_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(labeled_metadata, dict) or not labeled_metadata:
        raise RuntimeError(f"Invalid or empty CLEAR labeled metadata: {metadata_path}")

    items: list[Clear10Item] = []
    for bucket_key, class_map in sorted(labeled_metadata.items(), key=lambda pair: _numeric_bucket_key(pair[0])):
        bucket = int(bucket_key)
        if not isinstance(class_map, dict):
            raise ValueError(f"CLEAR bucket {bucket} metadata must be a class mapping")
        for label, class_name in enumerate(names):
            metadata_reference = class_map.get(class_name)
            if metadata_reference is None:
                # Some releases omit empty class/bucket cells. Absence therefore
                # means zero examples for this class in this period, not a label shift.
                continue
            class_metadata_path = split_folder / str(metadata_reference)
            if not class_metadata_path.is_file():
                raise FileNotFoundError(
                    f"CLEAR class metadata does not exist: {class_metadata_path} "
                    f"(bucket={bucket}, class={class_name})"
                )
            class_metadata = json.loads(class_metadata_path.read_text(encoding="utf-8"))
            if not isinstance(class_metadata, dict):
                raise ValueError(f"CLEAR class metadata must be an object: {class_metadata_path}")
            for sample_key, sample in sorted(class_metadata.items(), key=lambda pair: str(pair[0])):
                if not isinstance(sample, dict) or "IMG_PATH" not in sample:
                    raise ValueError(
                        f"CLEAR sample metadata lacks IMG_PATH: {class_metadata_path}:{sample_key}"
                    )
                archive_relative = str(sample["IMG_PATH"])
                image = _resolve_new_image(
                    input_root=input_root,
                    dataset_root=dataset_root,
                    split_folder=split_folder,
                    split=split,
                    image_path=archive_relative,
                )
                try:
                    relative_path = f"{split}/{image.relative_to(split_folder).as_posix()}"
                except ValueError:
                    relative_path = f"{split}/{archive_relative}"
                items.append(
                    Clear10Item(
                        path=image,
                        label=int(label),
                        bucket=bucket,
                        relative_path=relative_path,
                    )
                )
    if not items:
        raise RuntimeError(f"No CLEAR-10 {split} images found from {metadata_path}")
    return items


def _scan_legacy_format(input_root: Path, dataset_root: Path, split: str) -> list[Clear10Item]:
    split_folder = _find_split_folder(dataset_root, split)
    items: list[Clear10Item] = []
    for bucket in _bucket_indices(split_folder):
        filelist = _filelist_path(split_folder, bucket)
        for relative, label in _read_filelist(filelist):
            relative_path = Path(relative)
            candidates = [
                dataset_root / relative_path,
                split_folder / relative_path,
                filelist.parent / relative_path,
                input_root / relative_path,
                input_root / "raw" / relative_path,
            ]
            image = next((candidate for candidate in candidates if candidate.is_file()), None)
            if image is None:
                # Some filelists are relative to the archive wrapper rather than the metadata folder.
                basename_matches = list(dataset_root.rglob(relative_path.name))
                image = basename_matches[0] if len(basename_matches) == 1 else None
            if image is None:
                raise FileNotFoundError(
                    f"CLEAR-10 image from {filelist} does not exist: {relative}; tried {candidates}"
                )
            items.append(
                Clear10Item(
                    path=image.resolve(),
                    label=int(label),
                    bucket=int(bucket),
                    relative_path=relative,
                )
            )
    if not items:
        raise RuntimeError(f"No CLEAR-10 {split} images found below {dataset_root}")
    return items


def scan_clear10(root: str | Path, split: str) -> list[Clear10Item]:
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown CLEAR-10 split: {split}")
    input_root = Path(root).expanduser().resolve()

    # Current official archives are independent split roots. In particular the
    # train image-only archive extracts as train_image_only/, while test uses
    # test/. They need not share one common dataset wrapper.
    new_split_folder = _find_new_split_folder(input_root, split)
    if new_split_folder is not None:
        return _scan_new_format(
            input_root=input_root,
            dataset_root=new_split_folder.parent,
            split=split,
            split_folder=new_split_folder,
        )

    dataset_root = _find_dataset_root(input_root, split)
    return _scan_legacy_format(input_root, dataset_root, split)


def group_by_bucket(items: Iterable[Clear10Item]) -> dict[int, list[Clear10Item]]:
    grouped: dict[int, list[Clear10Item]] = {}
    for item in items:
        grouped.setdefault(item.bucket, []).append(item)
    return grouped
