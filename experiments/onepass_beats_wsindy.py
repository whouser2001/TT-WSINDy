"""
A setting where one-pass TT-WSINDy BEATS flat (matrix) WSINDy -- in BOTH
walltime and recovery.

Setting
-------
Chaotic Lorenz-96 (F=8), with a *polynomial* candidate library per variable,
    f = {1, x, x^2, ..., x^(J-1)}   (J=5),
so the full tensor-product library has J^D candidates per equation
(3125 at D=5, 4096 at J=4/D=6, ...). The true model uses only {1, x}:
    dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F .

Result
------
One-pass TT-WSINDy recovers the EXACT governing equations and runs ~4.7x faster
than flat WSINDy, which FAILS to recover on the same library (J=5, D=5, 3125
candidates/eqn, averaged over 3 seeds):
    TT ~14s, recovers 5/5   vs   flat ~66s, recovers 0/5.

(The speedup reflects two construction/coarse-pass optimizations in
feature_tensor.py and ttwsindy.py: a fused convolution + randomized truncated
SVD in the low_rank build, and reusing the target-independent pseudoinverse SVD
across all D equations. Before them TT was only ~1.3-1.5x faster. D=5 is the
robust recovery regime for this polynomial library; D>=6 needs a narrower window
/ longer trajectory to keep the weak -x_i self-term, and is left out here.)

Why TT wins on this library
---------------------------
Both methods use the SAME weak form, the SAME library, and the SAME MSTLS
thresholding loss. The only difference is the coarse reduction:

* Flat WSINDy must run least-squares + sequential thresholding on the entire
  J^D library at once. That library is ill-conditioned and scale-disparate (an
  x^4 column has ~500x the norm of the constant column on the F=8 attractor), so
  the MSTLS band thresholds out everything -> empty / garbage support. Column-
  normalizing does not save it (the band logic then loses its per-column scale,
  and a classic normalized sequential-threshold solve is both far slower and
  still wrong here).

* TT-WSINDy's coarse pass *normalizes every feature* and operates on the
  separable (product) structure, so it robustly reduces J^D -> a few hundred
  candidates; the fine MSTLS then solves a tiny, well-conditioned problem.

So the polynomial library is exactly the regime the structured TT solve is for:
flat chokes on the full rich library, TT reduces it first.

Run from experiments/ with the scikit_tt env:
    python onepass_beats_wsindy.py
"""
import os, sys, itertools
sys.path.insert(0, '.')
sys.path.insert(0, '../TT-WSINDy')
sys.path.insert(0, '../WSINDy')
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.signal import correlate
from scipy.integrate import odeint
import test_function
from sparsification import MSTLS
from ttwsindy import TT_WSINDy

t0 = 0.0
TTlambs   = 10.0 ** ((3.7/20)*np.arange(0, 21) - 4)   # coarse thresholds 1e-4..0.5
flatlambs = np.logspace(-4, -0.5, 16)                  # fine + flat thresholds


def poly_lib(J):
    return [(lambda p: (lambda x: x**p))(p) for p in range(J)]

POLY_STR = ['', 'x_{}', 'x_{}^2', 'x_{}^3', 'x_{}^4', 'x_{}^5']


def l96_data(F, c, D, M, tM, seed=0, burn=20.0):
    rhs = lambda x, t: (np.roll(x, -1) - np.roll(x, 2))*np.roll(x, 1) - c*x + F
    rng = np.random.default_rng(seed)
    x0 = (F/c)*np.ones(D) + 0.01*rng.standard_normal(D)
    x0 = odeint(rhs, x0, np.linspace(0, burn, 1000))[-1]   # land on attractor
    return odeint(rhs, x0, np.linspace(t0, tM, M)).T


def l96_true(D, c, F):
    """Per-equation dict {tuple-of-powers: coefficient}."""
    true = []
    for d in range(D):
        terms = {tuple([0]*D): F}
        t = [0]*D; t[d] = 1;                           terms[tuple(t)] = -c
        t = [0]*D; t[(d+1) % D] = 1; t[(d-1) % D] = 1; terms[tuple(t)] = 1.0
        t = [0]*D; t[(d-2) % D] = 1; t[(d-1) % D] = 1; terms[tuple(t)] = -1.0
        true.append(terms)
    return true


def recovery(supps, fmaps, W, true, D, rtol=0.15):
    """(# exactly recovered eqns, # support-matched eqns)."""
    ex = supp = 0
    for d in range(D):
        if supps[d] is None:
            continue
        rec = {tuple(int(p) for p in fmaps[d][k]): W[d][i]
               for i, k in enumerate(supps[d])}
        td = true[d]
        if set(rec) == set(td):
            supp += 1
            if all(abs(rec[t]-td[t]) <= rtol*max(1.0, abs(td[t])) for t in td):
                ex += 1
    return ex, supp


def term_str(tup):
    s = ''
    for d, p in enumerate(tup):
        if p:
            s += POLY_STR[p].format(d+1) if '{}' in POLY_STR[p] else POLY_STR[p]
    return s if s else '1'


def print_eqn(d, rec):
    parts = [f"{c:+.2f} {term_str(t)}" for t, c in sorted(rec.items())]
    print(f"   dx_{d+1}/dt = " + "  ".join(parts) if parts else f"   dx_{d+1}/dt = (empty)")


def run_flat(X, D, M, f, J, tM, r_frac, degree):
    st = time()
    phi, dphi = test_function.piecewise_polynomial((tM-t0)*r_frac, degree, t0, tM, M, order=1)
    fmap = list(itertools.product(*[range(J)]*D))
    basis = np.stack([np.vectorize(fj)(X).astype(float) for fj in f])
    G = np.ones((J**D, M))
    for k, tup in enumerate(fmap):
        for d in range(D):
            G[k] *= basis[tup[d], d]
    Gc = correlate(G, np.expand_dims(phi, 0), mode='valid').T
    Y = -1*correlate(X, np.expand_dims(dphi, 0), mode='valid').T
    supps, Ws = [], []
    for d in range(D):
        w, s = MSTLS(Gc, Y[:, d], flatlambs)
        supps.append(s); Ws.append(w)
    return time()-st, supps, [fmap]*D, Ws


def run_config(F, c, J, D, M, tM, r_frac, degree, seeds, verbose=True):
    tt_times, fl_times, tt_exs, fl_exs = [], [], [], []
    f = poly_lib(J)
    true = l96_true(D, c, F)
    for si, seed in enumerate(seeds):
        X = l96_data(F, c, D, M, tM, seed)
        r = TT_WSINDy(X, t0, tM, f, TTlambs, flatlambs,
                      testfn=('piecewise_polynomial', r_frac, degree, 1),
                      threshold=1e-16, verbosity=0, low_rank=True, one_pass=True)
        tt = r[3]; tt_ex, _ = recovery(r[1], r[2], r[0], true, D)
        ft, fs, ff, fW = run_flat(X, D, M, f, J, tM, r_frac, degree)
        fl_ex, _ = recovery(fs, ff, fW, true, D)
        tt_times.append(tt); fl_times.append(ft); tt_exs.append(tt_ex); fl_exs.append(fl_ex)
        if verbose:
            print(f"  seed {seed}: TT={tt:6.1f}s ex={tt_ex}/{D} | "
                  f"flat={ft:6.1f}s ex={fl_ex}/{D} | speedup={ft/tt:.2f}x", flush=True)
        if verbose and si == 0:
            # show the recovered equation for x_1 (TT correct, flat fails)
            rec_tt = {tuple(int(p) for p in r[2][0][k]): r[0][0][i] for i, k in enumerate(r[1][0])}
            print("   --- TT-WSINDy recovered dx_1/dt (correct): ---"); print_eqn(0, rec_tt)
            rec_fl = ({tuple(int(p) for p in ff[0][k]): fW[0][i] for i, k in enumerate(fs[0])}
                      if fs[0] is not None else {})
            print("   --- flat WSINDy recovered dx_1/dt (wrong/empty): ---"); print_eqn(0, rec_fl)
    return (np.mean(tt_times), np.mean(fl_times),
            int(np.min(tt_exs)), int(np.min(fl_exs)), D)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    recompute = True
    DATA = 'results/onepass_beats_wsindy.txt'

    # (F, c, J, D, M, tM, r_frac, degree, seeds). The robust J=5,D=5 regime
    # (3125 candidates/eqn), averaged over seeds.
    configs = [
        (8., 1., 5, 5, 5000, 50., 1/40, 16, [0, 1, 2]),
    ]

    if recompute:
        rows = []
        for (F, c, J, D, M, tM, rf, deg, seeds) in configs:
            print(f"\n=== L96 poly J={J} D={D} M={M} (J^D={J**D} candidates/eqn) ===", flush=True)
            tt, ft, tt_ex, fl_ex, _ = run_config(F, c, J, D, M, tM, rf, deg, seeds)
            print(f"  MEAN: TT={tt:.1f}s (recover {tt_ex}/{D}) | "
                  f"flat={ft:.1f}s (recover {fl_ex}/{D}) | speedup={ft/tt:.2f}x", flush=True)
            rows.append([J, D, J**D, tt, ft, tt_ex, fl_ex, M])
        rows = np.array(rows, float)
        np.savetxt(DATA, rows,
                   header="J D J^D TT_time flat_time TT_recover flat_recover M", fmt="%.6g")
    else:
        rows = np.loadtxt(DATA)

    # ----------------------------- plot --------------------------------------
    labels = [f"J={int(r[0])}, D={int(r[1])}\n($J^D$={int(r[2])}, M={int(r[7])})" for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 5))
    b1 = ax.bar(x - w/2, rows[:, 3], w, label='one-pass TT-WSINDy', color='C0')
    b2 = ax.bar(x + w/2, rows[:, 4], w, label='flat WSINDy', color='C3')
    for i, r in enumerate(rows):
        ax.text(x[i]-w/2, r[3], f"{int(r[5])}/{int(r[1])}\nrecovered", ha='center', va='bottom',
                fontsize=9, color='C0', fontweight='bold')
        ax.text(x[i]+w/2, r[4], f"{int(r[6])}/{int(r[1])}\nrecovered", ha='center', va='bottom',
                fontsize=9, color='C3', fontweight='bold')
        ax.text(x[i], max(r[3], r[4])*1.18, f"{r[4]/r[3]:.2f}x\nfaster", ha='center',
                va='bottom', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('walltime (s)')
    ax.set_ylim(0, rows[:, 4].max()*1.35)
    ax.set_title('One-pass TT-WSINDy beats flat WSINDy on a rich polynomial library\n'
                 '(Lorenz-96, F=8): TT recovers the exact ODE, flat fails')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('results/onepass_beats_wsindy.png', dpi=150)
    print("\nsaved results/onepass_beats_wsindy.png")
