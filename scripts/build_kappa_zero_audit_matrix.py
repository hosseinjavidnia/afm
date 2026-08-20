from __future__ import annotations
import argparse,json
from pathlib import Path

def read_jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source-suite-root',default='runs_compatibility_causal_v1'); ap.add_argument('--output-root',default='runs_kappa_zero_boundary_audit_v1'); args=ap.parse_args()
    source=Path(args.source_suite_root).resolve(); out=Path(args.output_root).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Refusing to overwrite {out}')
    (out/'runs').mkdir(parents=True)
    matrix=json.loads((source/'job_matrix.json').read_text()); jobs=[]; negative=[]
    for row in matrix:
        native=Path(row['run_dir'])/'afm_native_points.jsonl'; states=[]
        for r in read_jsonl(native):
            if bool(r.get('accepted')) and abs(float(r['requested_kappa']))<=1e-15 and r.get('afm_lambda_hat') is not None:
                margin=float(r['persistent_ratio'])-float(r['afm_lambda_hat'])*float(r['measured_kappa'])/3.0
                if margin<0:
                    states.append(int(r['state_index'])); negative.append({'system':row['system'],'seed':int(row['seed']),'state_index':int(r['state_index']),'stored_kappa':float(r['measured_kappa']),'stored_rho':float(r['persistent_ratio']),'stored_lambda_hat':float(r['afm_lambda_hat']),'stored_margin':margin})
        if states:
            name=f"{row['system']}_seed{row['seed']}"; jobs.append({'index':len(jobs),'system':row['system'],'seed':int(row['seed']),'source_run_dir':str(Path(row['run_dir']).resolve()),'run_dir':str((out/'runs'/name).resolve()),'target_state_indices':sorted(states)})
    (out/'job_matrix.json').write_text(json.dumps(jobs,indent=2,sort_keys=True)); (out/'stored_negative_rows.json').write_text(json.dumps(negative,indent=2,sort_keys=True)); print(f'negative stored rows: {len(negative)}'); print(f'GPU replay audit jobs: {len(jobs)}')

if __name__=='__main__': main()
