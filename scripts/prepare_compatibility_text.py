from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_huggingface_wikitext(output: Path) -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The default WikiText-2 preparation requires the 'datasets' package. "
            "Install it or pass --input-file /path/to/corpus.txt."
        ) from exc

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    with output.open("w", encoding="utf-8") as handle:
        for row in ds:
            text = str(row["text"])
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
    return "huggingface:Salesforce/wikitext:wikitext-2-v1:train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data_compatibility/text/input.txt")
    parser.add_argument(
        "--input-file",
        default=None,
        help="Use an existing UTF-8 text corpus instead of downloading WikiText-2.",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing text corpus: {output}")

    if args.input_file:
        shutil.copy2(args.input_file, output)
        source = str(Path(args.input_file).resolve())
    else:
        source = _write_huggingface_wikitext(output)

    payload = {
        "source": source,
        "path": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }
    (output.parent / "source.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
