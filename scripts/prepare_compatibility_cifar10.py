from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data_compatibility/cifar10")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    datasets.CIFAR10(root=str(root), train=True, download=True)
    datasets.CIFAR10(root=str(root), train=False, download=True)
    print(root.resolve())


if __name__ == "__main__":
    main()
