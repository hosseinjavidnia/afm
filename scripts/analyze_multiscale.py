from __future__ import annotations
import argparse, csv, json, math, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from scripts.extension_analysis_utils import read_jsonl, write_csv, exact_bootstrap_ci, slope


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--suite-root",default="runs_compatibility_multiscale_v1")
    ap.add_argument("--natural-rows",default=None,help="optional natural_state_validation_rows.csv for update-scale calibration")
    ap.add_argument("--plot-method",default="projection")
    ap.add_argument("--plot-beta",type=float,default=.10)
    args=ap.parse_args()
    suite=Path(args.suite_root).resolve(); out=suite/"analysis"; out.mkdir(parents=True,exist_ok=True)
    matrix=json.loads((suite/"job_matrix.json").read_text())
    summaries=[]; failures=[]; native_rows=[]
    # Aggregation sufficient for table + matched-state causal slopes.
    cell=defaultdict(list)  # system seed method beta scale req -> rows
    state=defaultdict(list) # system seed state method beta scale -> (kappa,rho)
    long_path=out/"multiscale_frontier_rows.csv"
    writer=None; handle=long_path.open("w",newline="",encoding="utf-8")
    observed=0
    for mr in matrix:
        run=Path(mr["run_dir"]); sp=run/"summary.json"; fp=run/"multiscale_frontier_points.jsonl"
        if not sp.is_file() or not fp.is_file():
            failures.append({"system":mr["system"],"seed":mr["seed"],"reason":"missing output"}); continue
        s=json.loads(sp.read_text()); summaries.append(s)
        if s.get("status")!="complete" or int(s.get("causal_states",-1))!=50:
            failures.append({"system":mr["system"],"seed":mr["seed"],"reason":f"status={s.get('status')} states={s.get('causal_states')}"})
        for p in read_jsonl(fp):
            if writer is None:
                writer=csv.DictWriter(handle,fieldnames=list(p.keys())); writer.writeheader()
            writer.writerow(p); observed += 1
            key=(p["system"],int(p["seed"]),p["method"],float(p["retention_beta"]),float(p["scale_fraction"]),float(p["requested_kappa"]))
            cell[key].append((float(p["measured_kappa"]),float(p["persistent_ratio"]),float(p["delta0"]),float(p["unrestricted_update_norm"]),float(p.get("target_fraction_of_common_peak", float("nan"))),float(p.get("target_multiple_of_v15_local", float("nan")))))
            sk=(p["system"],int(p["seed"]),int(p["state_index"]),p["method"],float(p["retention_beta"]),float(p["scale_fraction"]))
            state[sk].append((float(p["measured_kappa"]),float(p["persistent_ratio"])))
        np=run/"multiscale_afm_native_points.jsonl"
        if np.is_file():
            native_rows.extend(read_jsonl(np))
    handle.close()
    if not summaries:
        raise RuntimeError("no completed multi-scale runs found")
    systems=sorted({s["system"] for s in summaries}); scales=sorted({float(x) for s in summaries for x in s["scale_fractions"]})
    kappas=sorted({float(x) for s in summaries for x in s["requested_kappas"]}); methods=sorted({x for s in summaries for x in s["methods"]}); betas=sorted({float(x) for s in summaries for x in s["retention_budget_betas"]})
    expected=len(matrix)*50*len(scales)*len(kappas)*len(methods)*len(betas)

    seed_rows=[]
    for key, vals in sorted(cell.items()):
        system,seed,method,beta,scale,req=key
        seed_rows.append({"system":system,"seed":seed,"method":method,"retention_beta":beta,"scale_fraction":scale,"requested_kappa":req,"states":len(vals),"mean_measured_kappa":mean(v[0] for v in vals),"mean_persistent_ratio":mean(v[1] for v in vals),"median_persistent_ratio":median(v[1] for v in vals),"mean_delta0":mean(v[2] for v in vals),"mean_unrestricted_update_norm":mean(v[3] for v in vals),"mean_target_fraction_of_common_peak":mean(v[4] for v in vals if math.isfinite(v[4])) if any(math.isfinite(v[4]) for v in vals) else float("nan"),"mean_target_multiple_of_v15_local":mean(v[5] for v in vals if math.isfinite(v[5])) if any(math.isfinite(v[5]) for v in vals) else float("nan")})
    write_csv(out/"seed_level_multiscale.csv",seed_rows)
    ag=defaultdict(list)
    for r in seed_rows: ag[(r["system"],r["method"],r["retention_beta"],r["scale_fraction"],r["requested_kappa"])].append(r)
    agg=[]
    for key,rows in sorted(ag.items()):
        vals=[float(r["mean_persistent_ratio"]) for r in rows]; lo,hi=exact_bootstrap_ci(vals)
        agg.append({"system":key[0],"method":key[1],"retention_beta":key[2],"scale_fraction":key[3],"requested_kappa":key[4],"seeds":len(rows),"mean_measured_kappa":mean(float(r["mean_measured_kappa"]) for r in rows),"mean_persistent_ratio":mean(vals),"ci95_low":lo,"ci95_high":hi,"mean_delta0":mean(float(r["mean_delta0"]) for r in rows),"mean_unrestricted_update_norm":mean(float(r["mean_unrestricted_update_norm"]) for r in rows),"mean_target_fraction_of_common_peak":mean(float(r["mean_target_fraction_of_common_peak"]) for r in rows if math.isfinite(float(r["mean_target_fraction_of_common_peak"]))),"mean_target_multiple_of_v15_local":mean(float(r["mean_target_multiple_of_v15_local"]) for r in rows if math.isfinite(float(r["mean_target_multiple_of_v15_local"])) )})
    write_csv(out/"aggregate_multiscale.csv",agg)

    state_slopes=[]
    for key,vals in sorted(state.items()):
        vals=sorted(vals)
        state_slopes.append({"system":key[0],"seed":key[1],"state_index":key[2],"method":key[3],"retention_beta":key[4],"scale_fraction":key[5],"kappa_levels":len(vals),"matched_slope":slope([v[0] for v in vals],[v[1] for v in vals])})
    write_csv(out/"state_level_matched_slopes.csv",state_slopes)
    sg=defaultdict(list)
    for r in state_slopes:
        if math.isfinite(float(r["matched_slope"])): sg[(r["system"],r["seed"],r["method"],r["retention_beta"],r["scale_fraction"])].append(float(r["matched_slope"]))
    seed_s=[]
    for key,vals in sorted(sg.items()): seed_s.append({"system":key[0],"seed":key[1],"method":key[2],"retention_beta":key[3],"scale_fraction":key[4],"states":len(vals),"mean_matched_slope":mean(vals)})
    write_csv(out/"seed_level_matched_slopes.csv",seed_s)
    gg=defaultdict(list)
    for r in seed_s: gg[(r["system"],r["method"],r["retention_beta"],r["scale_fraction"])].append(float(r["mean_matched_slope"]))
    slopes=[]
    for key,vals in sorted(gg.items()):
        lo,hi=exact_bootstrap_ci(vals); slopes.append({"system":key[0],"method":key[1],"retention_beta":key[2],"scale_fraction":key[3],"seeds":len(vals),"matched_slope":mean(vals),"ci95_low":lo,"ci95_high":hi,"ci_strictly_positive":lo>0})
    write_csv(out/"matched_slopes_by_scale.csv",slopes)

    # Native AFM mechanism across update scales (separate from common projection frontier).
    ng=defaultdict(list)
    for r in native_rows:
        ng[(r['system'],int(r['seed']),float(r['scale_fraction']),float(r['requested_kappa']))].append(r)
    native_seed=[]
    for key,rows in sorted(ng.items()):
        native_seed.append({'system':key[0],'seed':key[1],'scale_fraction':key[2],'requested_kappa':key[3],'states':len(rows),'mean_measured_kappa':mean(float(r['measured_kappa']) for r in rows),'mean_persistent_ratio':mean(float(r['persistent_ratio']) for r in rows)})
    write_csv(out/'seed_level_multiscale_native_afm.csv',native_seed)
    nag=defaultdict(list)
    for r in native_seed: nag[(r['system'],r['scale_fraction'],r['requested_kappa'])].append(r)
    native_agg=[]
    for key,rows in sorted(nag.items()):
        vals=[float(r['mean_persistent_ratio']) for r in rows]; lo,hi=exact_bootstrap_ci(vals); native_agg.append({'system':key[0],'scale_fraction':key[1],'requested_kappa':key[2],'seeds':len(rows),'mean_measured_kappa':mean(float(r['mean_measured_kappa']) for r in rows),'mean_persistent_ratio':mean(vals),'ci95_low':lo,'ci95_high':hi})
    write_csv(out/'aggregate_multiscale_native_afm.csv',native_agg)

    # Optional comparison to ordinary natural-update scale.
    calibration=[]
    if args.natural_rows:
        nat_norm=defaultdict(list); nat_delta0=defaultdict(list)
        with Path(args.natural_rows).open(newline="",encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["method"]=="unrestricted":
                    nat_norm[r["system"]].append(float(r["update_norm"]))
                    nat_delta0[r["system"]].append(float(r["delta0"]))
        ms_norm=defaultdict(list); ms_delta0=defaultdict(list)
        for r in seed_rows:
            ms_norm[(r["system"],r["scale_fraction"])].append(float(r["mean_unrestricted_update_norm"]))
            ms_delta0[(r["system"],r["scale_fraction"])].append(float(r["mean_delta0"]))
        for (system,scale),vals in sorted(ms_norm.items()):
            nv=nat_norm.get(system,[]); nd=nat_delta0.get(system,[]); dv=ms_delta0[(system,scale)]
            calibration.append({"system":system,"scale_fraction":scale,"mean_causal_unrestricted_update_norm":mean(vals),"median_natural_unrestricted_update_norm":median(nv) if nv else float('nan'),"causal_to_natural_update_norm_ratio":mean(vals)/median(nv) if nv and median(nv)>0 else float('nan'),"mean_causal_delta0":mean(dv),"median_natural_delta0":median(nd) if nd else float('nan'),"causal_to_natural_delta0_ratio_note_cross_objective_units":mean(dv)/median(nd) if nd and median(nd)>0 else float('nan')})
        write_csv(out/"natural_update_scale_calibration.csv",calibration)

    # Requested plots: rho versus realised kappa at every Delta0 scale, one panel/file per system.
    try:
        import matplotlib.pyplot as plt
        for system in systems:
            rows=[r for r in agg if r["system"]==system and r["method"]==args.plot_method and abs(float(r["retention_beta"])-args.plot_beta)<1e-12]
            if not rows: continue
            fig,ax=plt.subplots(figsize=(7.2,5.0))
            for sc in scales:
                rr=sorted([r for r in rows if abs(float(r["scale_fraction"])-sc)<1e-12],key=lambda r:r["mean_measured_kappa"])
                if rr:
                    ax.errorbar([r["mean_measured_kappa"] for r in rr],[r["mean_persistent_ratio"] for r in rr],yerr=[[r["mean_persistent_ratio"]-r["ci95_low"] for r in rr],[r["ci95_high"]-r["mean_persistent_ratio"] for r in rr]],marker="o",label=f"scale={sc:g}")
            ax.set_xlabel("Realised compatibility κ"); ax.set_ylabel("Persistent-progress ratio ρ")
            ax.set_title(f"{system}: {args.plot_method}, β={args.plot_beta:g}"); ax.legend(); fig.tight_layout()
            fig.savefig(out/f"multiscale_{system}_{args.plot_method}_beta{str(args.plot_beta).replace('.','p')}.png",dpi=220); plt.close(fig)
        for system in systems:
            rows=[r for r in native_agg if r['system']==system]
            if not rows: continue
            fig,ax=plt.subplots(figsize=(7.2,5.0))
            for sc in scales:
                rr=sorted([r for r in rows if abs(float(r['scale_fraction'])-sc)<1e-12],key=lambda r:r['mean_measured_kappa'])
                if rr:
                    ax.errorbar([r['mean_measured_kappa'] for r in rr],[r['mean_persistent_ratio'] for r in rr],yerr=[[r['mean_persistent_ratio']-r['ci95_low'] for r in rr],[r['ci95_high']-r['mean_persistent_ratio'] for r in rr]],marker='o',label=f'scale={sc:g}')
            ax.set_xlabel('Realised compatibility κ'); ax.set_ylabel('Native AFM persistent-base progress ratio'); ax.set_title(f'{system}: native AFM across Δ0 scales'); ax.legend(); fig.tight_layout(); fig.savefig(out/f'multiscale_native_afm_{system}.png',dpi=220); plt.close(fig)
    except Exception as exc:
        (out/"plot_warning.txt").write_text(str(exc),encoding="utf-8")

    validation={"pass":not failures and observed==expected,"scale_fraction_semantics":"log_expansion_from_v15_local_to_common_peak","runs_expected":len(matrix),"runs_complete":len(summaries),"failures":failures,"frontier_rows_expected":expected,"frontier_rows_observed":observed,"systems":systems,"scale_fractions":scales,"requested_kappas":kappas,"methods":methods,"retention_betas":betas}
    (out/"validation.json").write_text(json.dumps(validation,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(validation,indent=2,sort_keys=True))

if __name__=="__main__": main()
