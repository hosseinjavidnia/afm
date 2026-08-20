#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import os
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from afmvision.afm.trainer import AFMTrainer, SGDTrainer
from afmvision.baselines import AGEMTrainer, OnlineEWCTrainer, ReplayTrainer
from afmvision.config import load_config, set_by_dotted_key
from afmvision.data.stream import ManifestDataset, PrecomputedBatchSampler, collate_manifest
from afmvision.models.factory import build_model
from afmvision.utils.io import ensure_dir
from afmvision.utils.seed import seed_everything


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AFM-U vision potential experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=[
            "afm", "afm_no_protection", "matched_sgd", "sgd",
            "replay", "agem", "online_ewc", "oracle_ewc",
        ],
        default="afm",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        set_by_dotted_key(cfg, key, parse_value(value))
    cfg["method"] = args.method
    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["training"].get("deterministic", False)))

    if torch.cuda.is_available() and cfg["training"].get("device", "auto") in {"auto", "cuda"}:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    run_dir = ensure_dir(args.run_dir)
    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "runtime.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    dataset = ManifestDataset(
        cfg["data"]["train_manifest"],
        image_size=int(cfg["data"]["image_size"]),
        normalise=True,
    )
    batch_index_file = cfg.get("data", {}).get("batch_index_file")
    loader_kwargs = {
        "num_workers": int(cfg["training"]["num_workers"]),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(cfg["training"]["num_workers"]) > 0,
        "collate_fn": collate_manifest,
    }
    if batch_index_file:
        loader = DataLoader(
            dataset,
            batch_sampler=PrecomputedBatchSampler(batch_index_file, len(dataset)),
            **loader_kwargs,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
    model = build_model(cfg)
    if args.method in {"afm", "afm_no_protection"}:
        if args.method == "afm_no_protection":
            cfg.setdefault("afm", {}).setdefault("protection", {})["enabled"] = False
            cfg.setdefault("afm", {}).setdefault("run_requirements", {})["min_commits"] = 0
            cfg["afm"]["run_requirements"]["min_protected_nonzero_steps"] = 0
            (run_dir / "resolved_config.json").write_text(
                json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8"
            )
        trainer = AFMTrainer(model, cfg, device, run_dir)
    elif args.method == "replay":
        trainer = ReplayTrainer(model, cfg, device, run_dir)
    elif args.method == "agem":
        trainer = AGEMTrainer(model, cfg, device, run_dir)
    elif args.method in {"online_ewc", "oracle_ewc"}:
        trainer = OnlineEWCTrainer(
            model, cfg, device, run_dir, oracle_boundaries=args.method == "oracle_ewc"
        )
    else:
        trainer = SGDTrainer(model, cfg, device, run_dir, matched=args.method == "matched_sgd")
    try:
        summary = trainer.fit(loader)
    except BaseException as exc:
        failure = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "device": str(device),
            "torch": torch.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }
        (run_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
