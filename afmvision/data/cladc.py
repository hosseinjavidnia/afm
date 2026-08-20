from __future__ import annotations

import importlib
import json
import shutil
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class CLADCSample:
    image: Image.Image
    original_label: int
    source_index: int
    segment_index: int | None = None


def _find_repository_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    candidates = [root]
    if root.is_dir():
        candidates.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    for candidate in candidates:
        if (candidate / "clad").is_dir() and (candidate / "README.md").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find the official CLAD repository below {root}. "
        "Expected a directory containing clad/ and README.md."
    )


def import_official_clad(repository_root: str | Path):
    """Import the official VerwimpEli/CLAD module without accepting a PyPI namesake."""
    repository = _find_repository_root(repository_root)
    sys.path.insert(0, str(repository))
    try:
        for name in [key for key in sys.modules if key == "clad" or key.startswith("clad.")]:
            sys.modules.pop(name, None)
        module = importlib.import_module("clad")
    finally:
        try:
            sys.path.remove(str(repository))
        except ValueError:
            pass
    module_path = Path(getattr(module, "__file__", "")).resolve()
    if repository not in module_path.parents:
        raise ImportError(
            "Imported a module named 'clad' that is not the official checked-out repository: "
            f"{module_path}. Remove the unrelated package or pass the correct repository root."
        )
    for name in ("get_cladc_train", "get_cladc_val", "get_cladc_test"):
        if not callable(getattr(module, name, None)):
            raise AttributeError(f"Official CLAD module does not expose required function {name}")
    return module


_ANNOTATION_FILENAMES = (
    "instance_train.json",
    "instance_val.json",
    "instance_test.json",
)


def _candidate_data_roots(data_root: str | Path) -> list[Path]:
    root = Path(data_root).expanduser().resolve()
    candidates = [root, root / "raw", root / "SODA10M", root / "raw" / "SODA10M"]
    for base in (root, root / "raw"):
        if not base.is_dir():
            continue
        for match in sorted(base.rglob("SSLAD-2D")):
            if match.is_dir():
                candidates.extend((match.parent, match))
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _archive_preview(path: Path, limit: int = 160) -> str:
    try:
        return path.read_bytes()[:limit].decode("utf-8", errors="replace").replace("\n", "\\n")
    except OSError as exc:
        return f"<unreadable: {exc}>"


def ensure_official_annotations_archive(repository_root: str | Path) -> Path:
    """Return the supplemental CLAD ``annotations.zip`` shipped by the repository."""
    repository = _find_repository_root(repository_root)
    archive = repository / "annotations.zip"
    if archive.is_file() and zipfile.is_zipfile(archive):
        return archive
    state = "missing"
    if archive.exists():
        state = f"not a ZIP; first bytes={_archive_preview(archive)!r}"
    raise RuntimeError(
        "The official CLAD supplemental annotation archive is invalid. "
        f"Expected a readable ZIP at {archive}, but it is {state}. "
        "Re-extract the official VerwimpEli/CLAD source archive."
    )


def _annotation_member_map(archive: Path) -> dict[str, str]:
    members: dict[str, str] = {}
    with zipfile.ZipFile(archive, "r") as handle:
        for info in handle.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if name in _ANNOTATION_FILENAMES:
                if name in members:
                    raise RuntimeError(f"Duplicate {name} in {archive}")
                members[name] = info.filename
    missing = [name for name in _ANNOTATION_FILENAMES if name not in members]
    if missing:
        raise RuntimeError(
            f"Official CLAD {archive.name} is missing required members: {missing}; "
            f"found={sorted(members)}"
        )
    return members


def _validate_annotation_payload(payload: Any, *, split: str, source: str) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{source} is not a COCO-style JSON object")
    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not isinstance(images, list) or not images:
        raise RuntimeError(f"{source} has no images")
    if not isinstance(annotations, list):
        raise RuntimeError(f"{source} has no annotations list")
    if split == "test" and not annotations:
        raise RuntimeError(
            f"{source} has zero test annotations. The raw SODA10M test JSON is not sufficient for CLAD-C; "
            "the repository supplemental annotations.zip must be installed."
        )
    if not isinstance(categories, list) or len(categories) != 6:
        raise RuntimeError(f"{source} must contain the six CLAD-C categories")
    image_required = {"id", "file_name", "file_name_old", "date", "time", "period"}
    missing_image = image_required.difference(images[0])
    if missing_image:
        raise RuntimeError(f"{source} image records miss required CLAD-C fields: {sorted(missing_image)}")
    if annotations:
        annotation_required = {"id", "image_id", "category_id", "bbox", "area", "occluded", "truncated"}
        missing_annotation = annotation_required.difference(annotations[0])
        if missing_annotation:
            raise RuntimeError(
                f"{source} annotation records miss required CLAD-C fields: {sorted(missing_annotation)}"
            )


def _find_soda_layout_root(data_root: str | Path) -> Path:
    errors: list[str] = []
    for candidate in _candidate_data_roots(data_root):
        annotation_dir = candidate / "SSLAD-2D" / "labeled" / "annotations"
        missing = [name for name in _ANNOTATION_FILENAMES if not (annotation_dir / name).is_file()]
        if not missing:
            return candidate
        errors.append(f"{candidate}: missing {missing}")
    raise FileNotFoundError(
        "Could not locate the extracted SODA10M labelled layout. Expected "
        "<root>/SSLAD-2D/labeled/annotations/instance_{train,val,test}.json. Tried:\n- "
        + "\n- ".join(errors)
    )


def install_official_cladc_annotations(
    repository_root: str | Path,
    data_root: str | Path,
) -> tuple[Path, dict[str, str]]:
    """Install the repository's corrected CLAD-C JSON files at the exact SODA paths.

    The repository archive contains members such as ``annotations/instance_val.json``.
    They must be extracted into ``SSLAD-2D/labeled`` so that they land at
    ``SSLAD-2D/labeled/annotations/instance_val.json``. Extracting the archive into
    the already-existing ``annotations`` directory creates a wrong nested
    ``annotations/annotations`` tree and leaves the timestamp-free SODA files active.
    """
    archive = ensure_official_annotations_archive(repository_root)
    layout_root = _find_soda_layout_root(data_root)
    destination_dir = layout_root / "SSLAD-2D" / "labeled" / "annotations"
    destination_dir.mkdir(parents=True, exist_ok=True)
    member_map = _annotation_member_map(archive)
    installed_hashes: dict[str, str] = {}

    import hashlib

    with zipfile.ZipFile(archive, "r") as handle:
        for filename in _ANNOTATION_FILENAMES:
            raw = handle.read(member_map[filename])
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid JSON in {archive}:{member_map[filename]}: {exc}") from exc
            split = filename.removeprefix("instance_").removesuffix(".json")
            _validate_annotation_payload(payload, split=split, source=f"{archive}:{member_map[filename]}")

            destination = destination_dir / filename
            if destination.is_file() and destination.read_bytes() != raw:
                backup = destination.with_name(destination.name + ".soda_original")
                if not backup.exists():
                    shutil.copy2(destination, backup)
            temporary = destination.with_name(destination.name + ".clad_tmp")
            temporary.write_bytes(raw)
            temporary.replace(destination)
            installed_hashes[filename] = hashlib.sha256(raw).hexdigest()

    # Validate the files at the paths the upstream loader actually opens.
    for filename in _ANNOTATION_FILENAMES:
        destination = destination_dir / filename
        payload = json.loads(destination.read_text(encoding="utf-8"))
        split = filename.removeprefix("instance_").removesuffix(".json")
        _validate_annotation_payload(payload, split=split, source=str(destination))

    # Stale files created by the broken upstream timestamp URL are never used after
    # the corrected JSONs are installed. Keep forensic backups, but remove an active
    # payload so the loader cannot accidentally consume it in a future partial state.
    stale = destination_dir / "time_stamps.zip"
    if stale.exists():
        backup = stale.with_name("time_stamps.zip.obsolete")
        if backup.exists():
            backup.unlink()
        stale.replace(backup)

    return layout_root, installed_hashes


@contextmanager
def _forbid_upstream_timestamp_downloads(repository_root: Path) -> Iterator[None]:
    """Fail loudly if the official loader still attempts its dead timestamp URL."""
    try:
        utility = importlib.import_module("clad.utils.utils")
    except ModuleNotFoundError:
        yield
        return
    requests_module = getattr(utility, "requests", None)
    if requests_module is None or not hasattr(requests_module, "get"):
        yield
        return
    original = requests_module.get

    def forbidden(*args, **kwargs):
        raise RuntimeError(
            "Official CLAD attempted a timestamp network download after corrected annotations were installed. "
            "This indicates the active JSON path is wrong or incomplete."
        )

    requests_module.get = forbidden
    try:
        yield
    finally:
        requests_module.get = original


def _call_loader(loader, layout_root: Path):
    kwargs: dict[str, Any] = {}
    import inspect

    parameters = inspect.signature(loader).parameters
    if "transform" in parameters:
        kwargs["transform"] = lambda image: image
    if "img_size" in parameters:
        kwargs["img_size"] = 64
    return loader(str(layout_root), **kwargs)


def load_official_datasets(repository_root: str | Path, data_root: str | Path):
    repository = _find_repository_root(repository_root)
    layout_root, _ = install_official_cladc_annotations(repository, data_root)
    clad = import_official_clad(repository)

    # Clear the official lru_cache in case a caller used this process before repair.
    try:
        utility = importlib.import_module("clad.utils.utils")
    except ModuleNotFoundError:
        utility = None
    cache_clear = getattr(getattr(utility, "load_obj_img_dic", None), "cache_clear", None)
    if callable(cache_clear):
        cache_clear()

    with _forbid_upstream_timestamp_downloads(repository):
        train_sets = _call_loader(clad.get_cladc_train, layout_root)
        validation = _call_loader(clad.get_cladc_val, layout_root)
        test = _call_loader(clad.get_cladc_test, layout_root)

    if isinstance(train_sets, Dataset):
        train_sets = [train_sets]
    elif not isinstance(train_sets, Sequence):
        train_sets = list(train_sets)
    if not train_sets:
        raise RuntimeError("Official get_cladc_train returned no chronological segments")
    return list(train_sets), validation, test


def _extract_image_label(item: Any) -> tuple[Any, int]:
    if isinstance(item, dict):
        image = next((item[key] for key in ("image", "images", "input", "x") if key in item), None)
        label = next((item[key] for key in ("label", "labels", "target", "y") if key in item), None)
        if image is None or label is None:
            raise TypeError(f"Unsupported dictionary sample keys: {sorted(item)}")
        return image, int(torch.as_tensor(label).item())
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return item[0], int(torch.as_tensor(item[1]).item())
    raise TypeError(f"Unsupported CLAD-C sample type: {type(item).__name__}")


def to_rgb_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
            array = np.transpose(array, (1, 2, 0))
        if np.issubdtype(array.dtype, np.floating):
            minimum = float(np.nanmin(array))
            maximum = float(np.nanmax(array))
            if minimum < -1e-6 or maximum > 1.0 + 1e-6:
                raise ValueError(
                    "Official CLAD loader returned a floating image outside [0,1]. "
                    "Refusing to guess an inverse normalization."
                )
            array = np.clip(array * 255.0, 0, 255).round().astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    if torch.is_tensor(image):
        tensor = image.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(f"Expected CHW/HWC image tensor, got shape {tuple(tensor.shape)}")
        if tensor.shape[0] not in {1, 3, 4} and tensor.shape[-1] in {1, 3, 4}:
            tensor = tensor.permute(2, 0, 1)
        if tensor.dtype.is_floating_point:
            minimum = float(tensor.min().item())
            maximum = float(tensor.max().item())
            if minimum < -1e-6 or maximum > 1.0 + 1e-6:
                raise ValueError(
                    "Official CLAD loader returned a floating tensor outside [0,1]. "
                    "Refusing to silently invert an unknown normalization."
                )
            tensor = tensor.clamp(0.0, 1.0)
        else:
            tensor = tensor.to(torch.float32).div(255.0).clamp(0.0, 1.0)
        return TF.to_pil_image(tensor).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def iter_dataset(dataset: Dataset, segment_index: int | None = None) -> Iterator[CLADCSample]:
    for index in range(len(dataset)):
        image, label = _extract_image_label(dataset[index])
        yield CLADCSample(
            image=to_rgb_pil(image),
            original_label=int(label),
            source_index=index,
            segment_index=segment_index,
        )


def discover_soda_category_names(data_root: str | Path) -> dict[int, str]:
    """Best-effort category names from COCO-format SODA annotation files."""
    root = Path(data_root).expanduser().resolve()
    names: dict[int, str] = {}
    for path in sorted(root.rglob("*.json")):
        if "annot" not in str(path).lower():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        categories = payload.get("categories") if isinstance(payload, dict) else None
        if not isinstance(categories, list):
            continue
        for category in categories:
            if isinstance(category, dict) and "id" in category and "name" in category:
                names[int(category["id"])] = str(category["name"])
        if names:
            break
    return names
