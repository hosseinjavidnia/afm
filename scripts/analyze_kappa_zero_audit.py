from __future__ import annotations
import argparse,csv,json,sys
from collections import Counter,defaultdict
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from scripts.extension_analysis_utils import read_jsonl,write_csv

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--suite-root',default='runs_kappa_zero_boundary_audit_v1'); args=ap.parse_args(); suite=Path(args.suite_root).resolve(); out=suite/'analysis'; out.mkdir(parents=True,exist_ok=True); matrix=json.loads((suite/'job_matrix.json').read_text())
    rows=[]; failures=[]
    for m in matrix:
        p=Path(m['run_dir'])/'kappa_zero_audit_rows.jsonl'; s=Path(m['run_dir'])/'summary.json'
        if not p.is_file() or not s.is_file(): failures.append({'system':m['system'],'seed':m['seed'],'reason':'missing audit output'}); continue
        rows.extend(read_jsonl(p))
    write_csv(out/'kappa_zero_replay_audit_rows.csv',rows)
    counts=Counter(r['classification'] for r in rows); bysys=defaultdict(Counter)
    for r in rows: bysys[r['system']][r['classification']]+=1
    summary=[{'scope':'all','classification':k,'count':v} for k,v in sorted(counts.items())]
    for sysn,c in sorted(bysys.items()): summary.extend({'scope':sysn,'classification':k,'count':v} for k,v in sorted(c.items()))
    write_csv(out/'classification_summary.csv',summary)
    expected=sum(len(m['target_state_indices']) for m in matrix)
    validation={'pass':not failures and len(rows)==expected,'audit_jobs_expected':len(matrix),'failures':failures,'negative_rows_expected':expected,'audit_rows_observed':len(rows),'classification_counts':dict(counts),'max_abs_stored_vs_recomputed_rho':max((abs(float(r['stored_rho'])-float(r['recomputed_rho'])) for r in rows),default=0.0),'max_abs_stored_vs_recomputed_kappa':max((abs(float(r['stored_measured_kappa'])-float(r['recomputed_kappa_float64'])) for r in rows),default=0.0)}
    (out/'validation.json').write_text(json.dumps(validation,indent=2,sort_keys=True)); print(json.dumps(validation,indent=2,sort_keys=True))
if __name__=='__main__': main()
