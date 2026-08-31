from __future__ import annotations

import json, math, time
from pathlib import Path
from typing import Any
import torch

from afmvision.compatibility.data import Batch, make_causal_current_batch
from afmvision.compatibility.extension_common import JsonlWriter, SavedParentProbeBase, coefficient_of_variation
from afmvision.compatibility.geometry import (
    _solve_protected_gram,
    build_protected_geometry,
    compatible_projection,
    functional_jacobian,
    make_controlled_targets,
)
from afmvision.compatibility.methods import calibrate_comparator_to_decrease, run_method, unrestricted_comparator


def compatibility_fraction_float64(vector: torch.Tensor, geometry) -> float:
    q=vector.to(dtype=torch.float64)
    den=float(torch.dot(q,q).item())
    if den<=1e-30: return 0.0
    J=geometry.jacobian.to(device=q.device,dtype=torch.float64)
    if J.numel()==0: return 1.0
    coeff=_solve_protected_gram(geometry,J@q,device=q.device)
    allowed=q-J.T@coeff
    return float(torch.dot(allowed,allowed).item())/den


def classify_boundary_row(*, stored_kappa:float, recomputed_kappa:float, recomputed_margin:float, efficiency_ratio:float, kappa_tol:float=1e-4, margin_tol:float=1e-8) -> str:
    if abs(float(stored_kappa)-float(recomputed_kappa))>float(kappa_tol):
        return 'diagnostic_kappa_mismatch'
    if float(recomputed_margin)>=-float(margin_tol):
        return 'stored_roundoff_or_reporting_precision'
    if math.isfinite(float(efficiency_ratio)) and float(efficiency_ratio)<(1.0/3.0):
        return 'finite_curvature_or_step_condition_not_certified'
    return 'unresolved_requires_deeper_theorem_condition_audit'


class BoundaryAuditReplayRunner(SavedParentProbeBase):
    schema='kappa_zero_boundary_audit_v1'
    def __init__(self,*,source_run_dir:str|Path,run_dir:str|Path,system:str,device:torch.device,target_state_indices:list[int]):
        super().__init__(source_run_dir=source_run_dir,run_dir=run_dir,system=system,device=device,max_states=50,max_probe_steps=int(5000))
        self.targets=set(int(x) for x in target_state_indices)
        self.audit=JsonlWriter(self.run_dir/'kappa_zero_audit_rows.jsonl')
        stored={}
        p=self.source_run_dir/'afm_native_points.jsonl'
        with p.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                r=json.loads(line)
                if abs(float(r['requested_kappa']))<=1e-15:
                    stored[int(r['state_index'])]=r
        self.stored=stored
        missing=sorted(self.targets-set(stored))
        if missing: raise RuntimeError(f'target audit states absent from source native rows: {missing}')
        (self.run_dir/'audit_config.json').write_text(json.dumps({'schema':self.schema,'source_run_dir':str(self.source_run_dir.resolve()),'system':self.system,'seed':self.seed,'target_state_indices':sorted(self.targets)},indent=2,sort_keys=True))

    def probe_state(self,*,state_index:int,parent_step:int,protected:Batch,novel:Batch,guards:Batch|None)->int:
        requested=[float(x) for x in self.causal['requested_kappas']]
        current=make_causal_current_batch(modality=self.modality,protected=protected,novel=novel,near_count=int(self.causal['near_protected_count']),seed=self.seed*1000003+state_index,vocab_size=self.vocab_size)
        current_inputs=current.inputs.to(self.device); protected_inputs=protected.inputs.to(self.device); protected_labels=protected.labels.to(self.device); guard_inputs=None if guards is None else guards.inputs.to(self.device)
        self.model.eval(); pre=self.vectoriser.flatten(detach=True)
        with torch.no_grad(): protected_before=self.model(protected_inputs).detach().clone(); replay_logits=protected_before.clone()
        _,Jp=functional_jacobian(self.model,protected_inputs); cm,Jc=functional_jacobian(self.model,current_inputs); geometry=build_protected_geometry(Jp,ridge=float(self.causal.get('geometry_ridge',1e-7)))
        targets=make_controlled_targets(model=self.model,current_inputs=current_inputs,current_measurements=cm.detach(),current_jacobian=Jc.detach(),geometry=geometry,requested_kappas=requested,gradient_norm=float(self.causal.get('gradient_norm',1.0)))
        if float(targets[0].achievable_max-targets[0].achievable_min)<float(self.causal.get('minimum_achievable_span',.5)):
            self.vectoriser.assign(pre); return 0
        raw=[]
        try:
            for t in targets:
                self.vectoriser.assign(pre); c=unrestricted_comparator(model=self.model,vectoriser=self.vectoriser,current_inputs=current_inputs,current_targets=t.teacher_measurements,current_gradient=t.gradient,initial_alpha=float(self.causal.get('comparator_lr',1e-2)),max_backtracks=int(self.causal.get('comparator_max_backtracks',20)),backtrack_factor=float(self.causal.get('comparator_backtrack_factor',.5))); raw.append((t,c))
        except RuntimeError:
            self.vectoriser.assign(pre); return 0
        target_delta=float(self.causal.get('comparator_match_fraction',.5))*min(float(c.decrease) for _,c in raw)
        matched=[]
        try:
            for t,c in raw:
                self.vectoriser.assign(pre); m=calibrate_comparator_to_decrease(model=self.model,vectoriser=self.vectoriser,current_inputs=current_inputs,current_targets=t.teacher_measurements,base=c,target_decrease=target_delta,relative_tolerance=float(self.causal.get('comparator_match_rtol',2e-3)),max_bisections=int(self.causal.get('comparator_match_bisections',40))); matched.append((t,m))
        except RuntimeError:
            self.vectoriser.assign(pre); return 0
        cv=coefficient_of_variation([float(c.decrease) for _,c in matched])
        if cv>float(self.causal.get('maximum_comparator_cv',.15)):
            self.vectoriser.assign(pre); return 0
        # The state is now accepted under the original v1.5 rule.
        if state_index in self.targets:
            pair=min(matched,key=lambda tc:abs(float(tc[0].requested_kappa)))
            t,c=pair; self.vectoriser.assign(pre)
            native=run_method(method='afm',model=self.model,vectoriser=self.vectoriser,comparator=c,current_gradient=t.gradient,geometry=geometry,current_inputs=current_inputs,current_targets=t.teacher_measurements,protected_inputs=protected_inputs,protected_labels=protected_labels,protected_logits_before=protected_before,replay_stored_logits=replay_logits,guard_inputs=guard_inputs,retention_tolerance=float(self.causal.get('retention_tolerance',.005)),method_config=self.causal,address_encoder=self.address_encoder)
            lam=float(native.afm_lambda_hat) if native.afm_lambda_hat is not None else float('nan')
            k_native=float(t.measured_kappa); k64=compatibility_fraction_float64(t.gradient,geometry)
            q=t.gradient; allowed=compatible_projection(q,geometry)
            q2=float(torch.dot(q,q).item()); a2=float(torch.dot(allowed,allowed).item())
            linear_unres=float(c.alpha)*q2
            linear_persist=float(c.alpha)*lam*a2 if math.isfinite(lam) else float('nan')
            unres_eff=float(c.decrease)/linear_unres if linear_unres>0 else float('nan')
            persist_eff=float(native.persistent_decrease)/linear_persist if linear_persist>0 else float('nan')
            eff_ratio=persist_eff/unres_eff if unres_eff>0 and math.isfinite(persist_eff) else float('nan')
            coarse=lam*k64/3.0 if math.isfinite(lam) else float('nan'); margin=float(native.persistent_ratio)-coarse
            allowed2=compatible_projection(allowed,geometry); idem=float(torch.linalg.vector_norm(allowed2-allowed).item())/max(float(torch.linalg.vector_norm(allowed).item()),1e-30)
            stored=self.stored[state_index]; stored_lam=float(stored['afm_lambda_hat']) if stored.get('afm_lambda_hat') is not None else float('nan'); stored_margin=float(stored['persistent_ratio'])-stored_lam*float(stored['measured_kappa'])/3.0
            classification=classify_boundary_row(stored_kappa=float(stored['measured_kappa']),recomputed_kappa=k64,recomputed_margin=margin,efficiency_ratio=eff_ratio)
            self.audit.write({'system':self.system,'seed':self.seed,'state_index':state_index,'parent_step':parent_step,'stored_measured_kappa':float(stored['measured_kappa']),'recomputed_measured_kappa_native':k_native,'recomputed_kappa_float64':k64,'kappa_native_minus_float64':k_native-k64,'stored_lambda_hat':stored_lam,'recomputed_lambda_hat':lam,'stored_rho':float(stored['persistent_ratio']),'recomputed_rho':float(native.persistent_ratio),'stored_margin':stored_margin,'recomputed_coarse_reference':coarse,'recomputed_margin':margin,'delta0':float(c.decrease),'persistent_decrease':float(native.persistent_decrease),'comparator_alpha':float(c.alpha),'gradient_norm_sq':q2,'compatible_gradient_norm_sq':a2,'linear_unrestricted_decrease':linear_unres,'linear_persistent_decrease':linear_persist,'unrestricted_finite_efficiency':unres_eff,'persistent_finite_efficiency':persist_eff,'finite_efficiency_ratio':eff_ratio,'coarse_reference_requires_efficiency_ratio_ge_one_third':eff_ratio>=1/3 if math.isfinite(eff_ratio) else False,'projection_idempotence_relative_error':idem,'classification':classification})
        self.vectoriser.assign(pre); return 1

    def run(self)->dict[str,Any]:
        start=time.time(); traj=self.run_probe_trajectory(); rows=sum(1 for _ in (self.run_dir/'kappa_zero_audit_rows.jsonl').open()) if (self.run_dir/'kappa_zero_audit_rows.jsonl').is_file() else 0
        summary={'schema':self.schema,'status':'complete' if self.targets.issubset(self.stored.keys()) and rows==len(self.targets) else 'incomplete','system':self.system,'seed':self.seed,'target_rows':len(self.targets),'audit_rows':rows,'causal_states_replayed':traj['accepted_states'],'probe_attempts':traj['probe_attempts'],'elapsed_seconds':time.time()-start}
        (self.run_dir/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)); self.events.write({'event':'finished',**summary}); self.events.close(); self.audit.close()
        if rows!=len(self.targets): raise RuntimeError(f'audit reconstructed {rows}/{len(self.targets)} target rows')
        return summary
