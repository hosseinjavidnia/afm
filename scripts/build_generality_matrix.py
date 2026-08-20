from __future__ import annotations
import argparse,json
from copy import deepcopy
from pathlib import Path
import yaml

BASE_SYSTEMS=[('cifar10_cnn','configs/compatibility/causal_sweep_cifar_cnn.yaml'),('cifar10_vit','configs/compatibility/causal_sweep_cifar_vit.yaml'),('text_transformer','configs/compatibility/causal_sweep_text_transformer.yaml')]
ORIGINAL=[11,29,47,71,101]
EXTRA=[131,149,167,191,223]
TEN=ORIGINAL+EXTRA

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-suite-root',default='runs_compatibility_causal_v1')
    ap.add_argument('--output-root',default='runs_compatibility_generality_v1')
    ap.add_argument('--extra-seeds',nargs='+',type=int,default=EXTRA)
    ap.add_argument('--strong-vit-seeds',nargs='+',type=int,default=TEN)
    ap.add_argument('--strong-vit-dim',type=int,default=96)
    ap.add_argument('--strong-vit-depth',type=int,default=6)
    ap.add_argument('--strong-vit-heads',type=int,default=6)
    args=ap.parse_args()
    root=Path.cwd().resolve(); source=Path(args.source_suite_root).resolve(); out=Path(args.output_root).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Refusing to overwrite {out}')
    (out/'configs').mkdir(parents=True); (out/'runs').mkdir(parents=True); (out/'analysis').mkdir(parents=True)
    source_matrix=json.loads((source/'job_matrix.json').read_text())
    combined=[]
    for row in source_matrix:
        combined.append({'index':len(combined),'system':row['system'],'seed':int(row['seed']),'config':str(Path(row['config']).resolve()),'run_dir':str(Path(row['run_dir']).resolve()),'provenance':'existing_v15'})
    new=[]
    for system,rel in BASE_SYSTEMS:
        base=yaml.safe_load((root/rel).read_text())
        for seed in args.extra_seeds:
            cfg=deepcopy(base); cfg['seed']=int(seed); name=f'{system}_seed{seed}'
            cp=out/'configs'/f'{name}.yaml'; rd=out/'runs'/name; cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
            row={'index':len(new),'system':system,'seed':int(seed),'config':str(cp.resolve()),'run_dir':str(rd.resolve()),'provenance':'new_extra_seed'}; new.append(row)
            combined.append({**row,'index':len(combined)})
    vit_base=yaml.safe_load((root/'configs/compatibility/causal_sweep_cifar_vit.yaml').read_text())
    vit_base['compatibility_sweep']['model']['dim']=int(args.strong_vit_dim)
    vit_base['compatibility_sweep']['model']['depth']=int(args.strong_vit_depth)
    vit_base['compatibility_sweep']['model']['heads']=int(args.strong_vit_heads)
    for seed in args.strong_vit_seeds:
        cfg=deepcopy(vit_base); cfg['seed']=int(seed); name=f'cifar10_vit_strong_seed{seed}'
        cp=out/'configs'/f'{name}.yaml'; rd=out/'runs'/name; cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
        row={'index':len(new),'system':'cifar10_vit_strong','seed':int(seed),'config':str(cp.resolve()),'run_dir':str(rd.resolve()),'provenance':'new_strong_vit'}; new.append(row)
        combined.append({**row,'index':len(combined)})
    (out/'new_job_matrix.json').write_text(json.dumps(new,indent=2,sort_keys=True))
    (out/'job_matrix.json').write_text(json.dumps(combined,indent=2,sort_keys=True))
    design={'original_seeds':ORIGINAL,'extra_seeds':[int(x) for x in args.extra_seeds],'strong_vit_seeds':[int(x) for x in args.strong_vit_seeds],'strong_vit':{'architecture':'vit','dim':args.strong_vit_dim,'depth':args.strong_vit_depth,'heads':args.strong_vit_heads},'new_jobs':len(new),'combined_runs':len(combined)}
    (out/'design.json').write_text(json.dumps(design,indent=2,sort_keys=True))
    print(f'WROTE: {out / "new_job_matrix.json"}'); print(f'new GPU jobs: {len(new)}'); print(f'combined runs after completion: {len(combined)}')

if __name__=='__main__': main()
