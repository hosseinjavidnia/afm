from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
import torch
from afmvision.compatibility.boundary_audit import BoundaryAuditReplayRunner

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--matrix',default=os.environ.get('AFM_KAPPA_AUDIT_MATRIX','runs_kappa_zero_boundary_audit_v1/job_matrix.json')); ap.add_argument('--index',type=int,default=None); args=ap.parse_args(); idx=args.index
    if idx is None:
        raw=os.environ.get('SLURM_ARRAY_TASK_ID');
        if raw is None: raise SystemExit('--index or SLURM_ARRAY_TASK_ID required')
        idx=int(raw)
    rows=json.loads(Path(args.matrix).read_text()); row=next((r for r in rows if int(r['index'])==idx),None)
    if row is None: raise SystemExit(f'job {idx} not found')
    if not torch.cuda.is_available(): raise SystemExit('exact boundary replay audit requires CUDA')
    runner=BoundaryAuditReplayRunner(source_run_dir=row['source_run_dir'],run_dir=row['run_dir'],system=row['system'],device=torch.device('cuda'),target_state_indices=row['target_state_indices']); print(json.dumps(runner.run(),indent=2,sort_keys=True))
if __name__=='__main__': main()
