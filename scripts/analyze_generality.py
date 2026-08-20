from __future__ import annotations
import argparse,csv,json,math,sys,hashlib
from collections import defaultdict
from pathlib import Path
from statistics import mean
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from scripts.extension_analysis_utils import read_jsonl,write_csv,mc_bootstrap_ci,slope

def stable_offset(obj):
    raw=repr(obj).encode('utf-8')
    return int(hashlib.sha256(raw).hexdigest()[:8],16)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--suite-root',default='runs_compatibility_generality_v1'); ap.add_argument('--bootstrap-resamples',type=int,default=100000); ap.add_argument('--bootstrap-seed',type=int,default=20260818); args=ap.parse_args()
    suite=Path(args.suite_root).resolve(); out=suite/'analysis'; out.mkdir(parents=True,exist_ok=True); matrix=json.loads((suite/'job_matrix.json').read_text())
    failures=[]; summaries=[]; state=defaultdict(list); observed=0
    for mr in matrix:
        run=Path(mr['run_dir']); sp=run/'summary.json'; fp=run/'retention_frontier_points.jsonl'
        if not sp.is_file() or not fp.is_file(): failures.append({'system':mr['system'],'seed':mr['seed'],'reason':'missing output'}); continue
        s=json.loads(sp.read_text()); summaries.append({**s,'system':mr['system'],'seed':int(mr['seed'])})
        if s.get('status')!='complete' or int(s.get('causal_states',-1))!=50: failures.append({'system':mr['system'],'seed':mr['seed'],'reason':f"status={s.get('status')} states={s.get('causal_states')}"})
        for p in read_jsonl(fp):
            observed+=1; key=(mr['system'],int(mr['seed']),int(p['state_index']),p['method'],float(p['retention_beta'])); state[key].append((float(p['measured_kappa']),float(p['persistent_ratio'])))
    if not summaries: raise RuntimeError('no runs found')
    methods=sorted({x for s in summaries for x in s['methods']}); betas=sorted({float(x) for s in summaries for x in s['retention_budget_betas']}); kappas=sorted({float(x) for s in summaries for x in s['requested_kappas']}); systems=sorted({s['system'] for s in summaries})
    expected=len(matrix)*50*len(methods)*len(betas)*len(kappas)
    state_rows=[]
    for key,vals in sorted(state.items()):
        vals=sorted(vals); state_rows.append({'system':key[0],'seed':key[1],'state_index':key[2],'method':key[3],'retention_beta':key[4],'kappa_levels':len(vals),'matched_slope':slope([v[0] for v in vals],[v[1] for v in vals])})
    write_csv(out/'state_level_matched_slopes.csv',state_rows)
    sg=defaultdict(list)
    for r in state_rows:
        if math.isfinite(float(r['matched_slope'])): sg[(r['system'],r['seed'],r['method'],r['retention_beta'])].append(float(r['matched_slope']))
    seed_rows=[]
    for key,vals in sorted(sg.items()): seed_rows.append({'system':key[0],'seed':key[1],'method':key[2],'retention_beta':key[3],'states':len(vals),'mean_matched_slope':mean(vals)})
    write_csv(out/'seed_level_matched_slopes.csv',seed_rows)
    gg=defaultdict(list)
    for r in seed_rows: gg[(r['system'],r['method'],r['retention_beta'])].append(float(r['mean_matched_slope']))
    summary=[]
    for key,vals in sorted(gg.items()):
        lo,hi=mc_bootstrap_ci(vals,resamples=args.bootstrap_resamples,seed=args.bootstrap_seed+stable_offset(key)%1000000)
        summary.append({'system':key[0],'method':key[1],'retention_beta':key[2],'seeds':len(vals),'mean_matched_slope':mean(vals),'ci95_low':lo,'ci95_high':hi,'ci_strictly_positive':lo>0,'bootstrap':'deterministic Monte Carlo seed bootstrap','bootstrap_resamples':args.bootstrap_resamples})
    write_csv(out/'generality_matched_slopes.csv',summary)
    # Equal-weight pooled system estimate within seed, only across systems sharing the same seed.
    idx={(r['system'],r['seed'],r['method'],r['retention_beta']):float(r['mean_matched_slope']) for r in seed_rows}
    pooled=[]
    for method in methods:
        for beta in betas:
            common=sorted(set.intersection(*[set(r['seed'] for r in seed_rows if r['system']==sysn and r['method']==method and float(r['retention_beta'])==beta) for sysn in systems])) if systems else []
            vals=[]
            for seed in common: vals.append(mean(idx[(sysn,seed,method,beta)] for sysn in systems))
            if vals:
                lo,hi=mc_bootstrap_ci(vals,resamples=args.bootstrap_resamples,seed=args.bootstrap_seed+17+stable_offset((method,beta))%1000000); pooled.append({'method':method,'retention_beta':beta,'systems':len(systems),'common_seeds':len(vals),'pooled_equal_weight_slope':mean(vals),'ci95_low':lo,'ci95_high':hi})
    write_csv(out/'generality_pooled_slopes.csv',pooled)
    try:
        import matplotlib.pyplot as plt
        rows=[r for r in summary if r['method']=='projection' and abs(float(r['retention_beta'])-.1)<1e-12]
        if rows:
            fig,ax=plt.subplots(figsize=(7,4.8)); xs=list(range(len(rows))); ys=[r['mean_matched_slope'] for r in rows]; yerr=[[r['mean_matched_slope']-r['ci95_low'] for r in rows],[r['ci95_high']-r['mean_matched_slope'] for r in rows]]; ax.errorbar(xs,ys,yerr=yerr,fmt='o'); ax.set_xticks(xs,[r['system'] for r in rows],rotation=20,ha='right'); ax.axhline(0,linewidth=1); ax.set_ylabel('Matched κ slope of persistent-progress ratio'); ax.set_title('10-seed / stronger-model compatibility generality, β=0.1'); fig.tight_layout(); fig.savefig(out/'generality_projection_beta0p1.png',dpi=220); plt.close(fig)
    except Exception as exc: (out/'plot_warning.txt').write_text(str(exc))
    seed_counts={sysn:len({int(s['seed']) for s in summaries if s['system']==sysn}) for sysn in systems}
    validation={'pass':not failures and observed==expected and all(v==10 for v in seed_counts.values()),'runs_expected':len(matrix),'runs_complete':len(summaries),'failures':failures,'frontier_rows_expected':expected,'frontier_rows_observed':observed,'seed_counts_by_system':seed_counts,'bootstrap_resamples':args.bootstrap_resamples,'systems':systems}
    (out/'validation.json').write_text(json.dumps(validation,indent=2,sort_keys=True)); print(json.dumps(validation,indent=2,sort_keys=True))

if __name__=='__main__': main()
