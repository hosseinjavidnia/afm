from __future__ import annotations
import argparse, subprocess, sys

def run(cmd):
    print('+',' '.join(cmd),flush=True); subprocess.run(cmd,check=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--natural-rows',default=None); args=ap.parse_args()
    cmd=[sys.executable,'scripts/analyze_multiscale.py','--suite-root','runs_compatibility_multiscale_v1']
    if args.natural_rows: cmd += ['--natural-rows',args.natural_rows]
    run(cmd)
    run([sys.executable,'scripts/analyze_independent_directions.py','--suite-root','runs_compatibility_independent_directions_v1'])
    run([sys.executable,'scripts/analyze_generality.py','--suite-root','runs_compatibility_generality_v1'])
    run([sys.executable,'scripts/analyze_kappa_zero_audit.py','--suite-root','runs_kappa_zero_boundary_audit_v1'])
if __name__=='__main__': main()
