"""
Single-run sandbox for the walltime/support scan.

Runs ONE configuration -- one dimension D, one trial, one TT-PI tolerance
(EPS16 by default) -- and prints the result to the terminal. Nothing is
plotted and no result file is written, so this cannot clobber the outputs of
walltime_support_vs_D.py; it exists to isolate a single expensive run instead
of re-running the whole D scan.

Everything except the driver is imported from walltime_support_vs_D, so the
problem setup (Lorenz-96, library, test function, threshold grids, metrics)
is by construction identical to the scan.

Usage
-----
    python sandbox_run.py                    # D=10, M from the scan's schedule, eps=1e-16
    python sandbox_run.py --D 12 --M 20000
    python sandbox_run.py --D 8 --noise 0    # clean data
    python sandbox_run.py --D 8 --flat       # also run flat WSINDy for comparison
    python sandbox_run.py --D 8 --eps 0      # exact TT-PI (no SVD truncation)
"""
import os, sys, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, 'TT-WSINDy'))
sys.path.insert(0, os.path.join(_ROOT, 'WSINDy'))
import matplotlib
matplotlib.use('Agg')          # never open a window; this script does not plot
import numpy as np
import walltime_support_vs_D as scan
from ttwsindy import TT_WSINDy


def coarse_report(coarse_supps, supp, D, J):
    """Print how many candidate terms the coarse TT-MSTLS pass eliminates.

    The coarse pass sparsifies one dimension at a time, so what survives it is
    the PRODUCT over the D modes of the surviving per-mode feature counts --
    that product is the column count handed to the fine matrix MSTLS. Counts
    are per equation and then summed over the D equations.
    """
    per_eqn = J**D
    rows = []
    for d in range(D):
        survivors = [max(len(s), 1) for s in coarse_supps[d]]
        coarse = int(np.prod(survivors))
        fine = 0 if supp[d] is None else len(supp[d])
        rows.append((d, survivors, coarse, fine))

    tot_init = D*per_eqn
    tot_coarse = sum(r[2] for r in rows)
    tot_fine = sum(r[3] for r in rows)

    print(f"\n--- coarse-pass elimination (J={J}, D={D}, J^D={per_eqn} per equation) ---")
    print("eqn   per-mode survivors        coarse   eliminated (coarse)      fine")
    for d, survivors, coarse, fine in rows:
        elim = per_eqn - coarse
        pct = 100.0*elim/per_eqn
        s = ''.join(str(v) for v in survivors)          # J is small: one digit per mode
        print(f"{d:3d}   {s:<22s} {coarse:8d}   {elim:8d} ({pct:5.1f}%)  {fine:8d}")
    print(f"total  initial {tot_init:9d} -> coarse {tot_coarse:8d} -> fine {tot_fine:6d}")
    print(f"       coarse pass eliminated {tot_init - tot_coarse} of {tot_init} terms "
          f"({100.0*(tot_init - tot_coarse)/tot_init:.2f}%)")
    print(f"       fine pass eliminated a further {tot_coarse - tot_fine} terms "
          f"({100.0*(tot_coarse - tot_fine)/max(tot_coarse, 1):.2f}% of what the coarse pass left)")
    return tot_init, tot_coarse, tot_fine


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--D', type=int, default=10, help='system dimension (default 10)')
    ap.add_argument('--M', type=int, default=None,
                    help='number of time points; default = the scan schedule for this D')
    ap.add_argument('--eps', type=float, default=scan.EPS16,
                    help='TT-PI SVD truncation tolerance (default 1e-16; 0 = exact)')
    ap.add_argument('--noise', type=float, default=None,
                    help='noise level as a fraction (default: the scan setting, %(default)s)')
    ap.add_argument('--seed', type=int, default=0, help='trial seed / initial condition')
    ap.add_argument('--flat', action='store_true',
                    help='also run flat WSINDy on the same data for comparison')
    ap.add_argument('--verbosity', type=int, default=0,
                    help='TT_WSINDy verbosity (0 quiet, 1 verbose, 2 debug)')
    args = ap.parse_args()

    D, J = args.D, scan.J
    if args.noise is not None:
        scan.NOISE_LEVEL = args.noise
    M = scan.set_sampling(D, args.M)          # sets scan.M, scan.tM, scan.R_FRAC

    noise_lab = ('clean data' if not scan.NOISE_LEVEL
                 else f'{scan.NOISE_LEVEL*100:g}% noise')
    print(f"=== sandbox: D={D}  J^D={J**D} candidates/eqn  M={M}  "
          f"t in [{scan.t0:g},{scan.tM:g}]  dt={scan.DT}  {noise_lab}  "
          f"eps={args.eps:g}  seed={args.seed} ===", flush=True)

    true = scan.l96_true(D)
    X = scan.l96_data(D, seed=args.seed)

    # --- TT-WSINDy (call TT_WSINDy directly so the timing split is visible) ---
    tim = {}
    W, supp, fmap, tt_time, tt_mstls_time, mstls_time, coarse_supps = TT_WSINDy(
        X, scan.t0, scan.tM, scan.f, scan.TTlambs, scan.flatlambs,
        testfn=('piecewise_polynomial', scan.R_FRAC, scan.DEGREE, 1),
        threshold=args.eps, verbosity=args.verbosity,
        low_rank=True, one_pass=True, timings=tim)

    recall, spur = scan.support_error(supp, fmap, D, true)
    cerr = scan.coeff_error(W, supp, fmap, D, true)

    print(f"\nTT-WSINDy  walltime {tt_time:8.2f}s   recall {recall:5.1f}%   "
          f"spurious/eqn {spur:6.2f}   coeff err {cerr:6.2f}%")
    # Stage breakdown. NOTE: the returned tt_mstls_time/mstls_time cover only the
    # two sparsification loops -- with one_pass=True the single global TT-PI SVD
    # is taken up front, OUTSIDE both, and is usually the dominant stage. Do not
    # read "total minus the two loops" as feature-tensor construction time.
    stages = [('feature tensor construction', 'feature_tensor'),
              ('global TT-PI SVD (one_pass)', 'pi_factors'),
              ('coarse TT-MSTLS loop',        'tt_mstls'),
              ('flat library rebuild',        'library_rebuild'),
              ('fine MSTLS loop',             'mstls')]
    print("  stage breakdown:")
    for lab, k in stages:
        v = tim.get(k, 0.0)
        print(f"    {lab:<30s} {v:8.2f}s  ({100.0*v/tt_time:5.1f}%)")
    other = tt_time - sum(tim.get(k, 0.0) for _, k in stages)
    print(f"    {'other (test fn, Y, overhead)':<30s} {other:8.2f}s  "
          f"({100.0*other/tt_time:5.1f}%)")
    # Cost is driven by the bond ranks: equal to the maximal ranks means the
    # carry never collapsed, so every interior SVD is as large as it can be.
    print(f"  TT bond ranks {tim.get('ranks')}")
    print(f"  maximal ranks {[min(J**k, M) for k in range(D + 1)]}")

    coarse_report(coarse_supps, supp, D, J)

    # --- optional flat WSINDy on the same trajectory ---
    if args.flat:
        tf, sf, fmf, Wf = scan.run_flat(X, D)
        rf, spf = scan.support_error(sf, fmf, D, true)
        cef = scan.coeff_error(Wf, sf, fmf, D, true)
        print(f"\nflat WSINDy  walltime {tf:8.2f}s   recall {rf:5.1f}%   "
              f"spurious/eqn {spf:8.2f}   coeff err {cef:6.2f}%")
        print(f"speedup TT vs flat: {tf/tt_time:.2f}x")

    print("\ndone.", flush=True)


if __name__ == '__main__':
    main()
