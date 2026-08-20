from __future__ import annotations
import argparse,csv,json,math,sys
from collections import defaultdict
from pathlib import Path
from statistics import mean,pstdev
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from scripts.extension_analysis_utils import read_jsonl,write_csv,exact_bootstrap_ci,slope


def rr(vals):
    m=mean(vals)
    return (max(vals)-min(vals))/abs(m) if vals and m!=0 else float('nan')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suite-root',default='runs_compatibility_independent_directions_v1')
    ap.add_argument('--plot-method',default='projection')
    ap.add_argument('--plot-beta',type=float,default=.10)
    args=ap.parse_args()
    suite=Path(args.suite_root).resolve(); out=suite/'analysis'; out.mkdir(parents=True,exist_ok=True)
    matrix=json.loads((suite/'job_matrix.json').read_text())
    failures=[]; summaries=[]; observed=0
    # Fixed-kappa groups across direction IDs.
    fixed=defaultdict(list) # sys seed state method beta req -> dict values
    kappa_state=defaultdict(list) # sys seed state method beta -> kappa-level mean later
    long_path=out/'independent_direction_frontier_rows.csv'; h=long_path.open('w',newline='',encoding='utf-8'); w=None
    for mr in matrix:
        run=Path(mr['run_dir']); sp=run/'summary.json'; fp=run/'independent_direction_frontier_points.jsonl'
        if not sp.is_file() or not fp.is_file(): failures.append({'system':mr['system'],'seed':mr['seed'],'reason':'missing output'}); continue
        s=json.loads(sp.read_text()); summaries.append(s)
        if s.get('status')!='complete' or int(s.get('causal_states',-1))!=50: failures.append({'system':mr['system'],'seed':mr['seed'],'reason':f"status={s.get('status')} states={s.get('causal_states')}"})
        for p in read_jsonl(fp):
            if w is None: w=csv.DictWriter(h,fieldnames=list(p.keys())); w.writeheader()
            w.writerow(p); observed+=1
            key=(p['system'],int(p['seed']),int(p['state_index']),p['method'],float(p['retention_beta']),float(p['requested_kappa']))
            fixed[key].append({'direction_id':int(p['direction_id']),'measured_kappa':float(p['measured_kappa']),'rho':float(p['persistent_ratio']),'delta0':float(p['delta0']),'update_norm':float(p['unrestricted_update_norm'])})
    h.close()
    if not summaries: raise RuntimeError('no completed independent-direction runs found')
    systems=sorted({s['system'] for s in summaries}); kappas=sorted({float(x) for s in summaries for x in s['requested_kappas']}); methods=sorted({x for s in summaries for x in s['methods']}); betas=sorted({float(x) for s in summaries for x in s['retention_budget_betas']}); dirs=sorted({int(s['directions_per_kappa']) for s in summaries})
    if len(dirs)!=1: raise RuntimeError(f'inconsistent directions_per_kappa: {dirs}')
    d=dirs[0]; expected=len(matrix)*50*len(kappas)*d*len(methods)*len(betas)

    fixed_rows=[]
    for key,vals in sorted(fixed.items()):
        rhos=[v['rho'] for v in vals]; ks=[v['measured_kappa'] for v in vals]; ds=[v['delta0'] for v in vals]; uns=[v['update_norm'] for v in vals]
        fixed_rows.append({'system':key[0],'seed':key[1],'state_index':key[2],'method':key[3],'retention_beta':key[4],'requested_kappa':key[5],'directions':len(vals),'mean_measured_kappa':mean(ks),'kappa_range':max(ks)-min(ks),'mean_rho':mean(rhos),'direction_sd_rho':pstdev(rhos) if len(rhos)>1 else 0.0,'direction_range_rho':max(rhos)-min(rhos),'delta0_cv':pstdev(ds)/mean(ds) if mean(ds)>0 else float('nan'),'update_norm_relative_range':rr(uns)})
        kappa_state[(key[0],key[1],key[2],key[3],key[4])].append((mean(ks),mean(rhos),pstdev(rhos) if len(rhos)>1 else 0.0))
    write_csv(out/'fixed_kappa_direction_variability.csv',fixed_rows)

    state_rows=[]
    for key,vals in sorted(kappa_state.items()):
        vals=sorted(vals)
        means=[v[1] for v in vals]; within=[v[2] for v in vals]
        between=pstdev(means) if len(means)>1 else 0.0
        within_mean=mean(within)
        state_rows.append({'system':key[0],'seed':key[1],'state_index':key[2],'method':key[3],'retention_beta':key[4],'kappa_levels':len(vals),'matched_kappa_slope':slope([v[0] for v in vals],means),'mean_within_kappa_direction_sd':within_mean,'between_kappa_sd_of_direction_means':between,'direction_to_kappa_sd_ratio':within_mean/between if between>0 else float('nan')})
    write_csv(out/'state_level_direction_vs_kappa.csv',state_rows)

    seedg=defaultdict(list)
    for r in state_rows: seedg[(r['system'],r['seed'],r['method'],r['retention_beta'])].append(r)
    seed_rows=[]
    for key,rows in sorted(seedg.items()):
        ratios=[float(r['direction_to_kappa_sd_ratio']) for r in rows if math.isfinite(float(r['direction_to_kappa_sd_ratio']))]
        slopes=[float(r['matched_kappa_slope']) for r in rows if math.isfinite(float(r['matched_kappa_slope']))]
        below=[1.0 if float(r['direction_to_kappa_sd_ratio'])<1 else 0.0
               for r in rows if math.isfinite(float(r['direction_to_kappa_sd_ratio']))]
        seed_rows.append({
            'system':key[0],
            'seed':key[1],
            'method':key[2],
            'retention_beta':key[3],
            'states':len(rows),
            'defined_ratio_states':len(ratios),
            'defined_slope_states':len(slopes),
            'mean_direction_to_kappa_sd_ratio':mean(ratios) if ratios else float('nan'),
            'mean_matched_kappa_slope':mean(slopes) if slopes else float('nan'),
            'fraction_states_direction_sd_below_between_kappa_sd':mean(below) if below else float('nan'),
        })
    write_csv(out/'seed_level_direction_independence.csv',seed_rows)
    agg=defaultdict(list)
    for r in seed_rows: agg[(r['system'],r['method'],r['retention_beta'])].append(r)
    summary=[]
    for key,rows in sorted(agg.items()):
        ratios=[float(r['mean_direction_to_kappa_sd_ratio']) for r in rows if math.isfinite(float(r['mean_direction_to_kappa_sd_ratio']))]
        slopes=[float(r['mean_matched_kappa_slope']) for r in rows if math.isfinite(float(r['mean_matched_kappa_slope']))]
        fracs=[float(r['fraction_states_direction_sd_below_between_kappa_sd'])
               for r in rows if math.isfinite(float(r['fraction_states_direction_sd_below_between_kappa_sd']))]
        rlo,rhi=exact_bootstrap_ci(ratios)
        slo,shi=exact_bootstrap_ci(slopes)
        summary.append({
            'system':key[0],
            'method':key[1],
            'retention_beta':key[2],
            'seeds':len(rows),
            'defined_ratio_seeds':len(ratios),
            'defined_slope_seeds':len(slopes),
            'mean_direction_to_kappa_sd_ratio':mean(ratios) if ratios else float('nan'),
            'ratio_ci95_low':rlo,
            'ratio_ci95_high':rhi,
            'mean_matched_kappa_slope':mean(slopes) if slopes else float('nan'),
            'slope_ci95_low':slo,
            'slope_ci95_high':shi,
            'slope_ci_strictly_positive':slo>0 if math.isfinite(slo) else False,
            'mean_fraction_states_direction_sd_below_between_kappa_sd':mean(fracs) if fracs else float('nan'),
        })
    write_csv(out/'direction_independence_summary.csv',summary)

    # Aggregate rho by requested kappa for publication plots.
    fg=defaultdict(list)
    for r in fixed_rows: fg[(r['system'],r['seed'],r['method'],r['retention_beta'],r['requested_kappa'])].append(float(r['mean_rho']))
    seedk=[]
    for key,vals in sorted(fg.items()): seedk.append({'system':key[0],'seed':key[1],'method':key[2],'retention_beta':key[3],'requested_kappa':key[4],'mean_rho':mean(vals)})
    ag=defaultdict(list)
    for r in seedk: ag[(r['system'],r['method'],r['retention_beta'],r['requested_kappa'])].append(float(r['mean_rho']))
    plotrows=[]
    for key,vals in sorted(ag.items()):
        lo,hi=exact_bootstrap_ci(vals); plotrows.append({'system':key[0],'method':key[1],'retention_beta':key[2],'requested_kappa':key[3],'mean_rho':mean(vals),'ci95_low':lo,'ci95_high':hi})
    write_csv(out/'aggregate_fixed_kappa_rho.csv',plotrows)
    try:
        import matplotlib.pyplot as plt
        for system in systems:
            rows=sorted([r for r in plotrows if r['system']==system and r['method']==args.plot_method and abs(float(r['retention_beta'])-args.plot_beta)<1e-12],key=lambda r:r['requested_kappa'])
            if not rows: continue
            fig,ax=plt.subplots(figsize=(6.5,4.8)); ax.errorbar([r['requested_kappa'] for r in rows],[r['mean_rho'] for r in rows],yerr=[[r['mean_rho']-r['ci95_low'] for r in rows],[r['ci95_high']-r['mean_rho'] for r in rows]],marker='o')
            ax.set_xlabel('Requested compatibility κ'); ax.set_ylabel('Mean persistent-progress ratio across independent directions'); ax.set_title(f'{system}: fixed-κ direction test, {args.plot_method}, β={args.plot_beta:g}'); fig.tight_layout(); fig.savefig(out/f"independent_directions_{system}_{args.plot_method}_beta{str(args.plot_beta).replace('.','p')}.png",dpi=220); plt.close(fig)
    except Exception as exc: (out/'plot_warning.txt').write_text(str(exc))

    validation={'pass':not failures and observed==expected and all(int(r['directions'])==d for r in fixed_rows),'runs_expected':len(matrix),'runs_complete':len(summaries),'failures':failures,'frontier_rows_expected':expected,'frontier_rows_observed':observed,'directions_per_kappa':d,'requested_kappas':kappas,'systems':systems,'methods':methods,'retention_betas':betas,'fixed_groups':len(fixed_rows),'all_fixed_groups_complete':all(int(r['directions'])==d for r in fixed_rows)}
    (out/'validation.json').write_text(json.dumps(validation,indent=2,sort_keys=True)); print(json.dumps(validation,indent=2,sort_keys=True))

if __name__=='__main__': main()
