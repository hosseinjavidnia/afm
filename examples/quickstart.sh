#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/smoke_output}"

rm -rf "${OUT}"
mkdir -p "${OUT}"
cd "${ROOT}"

python scripts/make_synthetic_smoke.py --output "${OUT}/data"

python scripts/run_experiment.py \
  --config configs/smoke.yaml \
  --method afm \
  --run-dir "${OUT}/run" \
  --set "data.train_manifest=\"${OUT}/data/train.jsonl\"" \
  --set "data.evaluator_manifest=\"${OUT}/data/eval.jsonl\""

python scripts/evaluate_run.py \
  --checkpoint "${OUT}/run/final.pt" \
  --manifest "${OUT}/data/eval.jsonl" \
  --output "${OUT}/run/evaluation.json" \
  --batch-size 4 \
  --num-workers 0

python scripts/check_run_validity.py --run-dir "${OUT}/run"
python scripts/analyze_afm_run.py --run-dir "${OUT}/run"

echo "AFM smoke run complete: ${OUT}/run"
