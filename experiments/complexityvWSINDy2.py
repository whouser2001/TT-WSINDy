"""
complexityvWSINDy2.py

Walltime advantage of TT-WSINDy over flat (matrix) WSINDy on Lorenz-96, as a
function of BOTH the state dimension D and the number of snapshots M. Produces a
3-D surface of

        diff(M, D) = walltime(flat WSINDy) - walltime(TT-WSINDy).

Positive => TT-WSINDy is faster. Both solvers run the same weak-form regression
(test function phi); only the representation differs (full J^D matrix vs. tensor
train), so this isolates the complexity of the solve.

Lorenz-96:  dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F,  F = 8, over a fixed
time window [0, 10] (so M sets the sampling density). Candidate library
f = {1, x} (J=2): Lorenz-96 is spanned exactly (the cross terms x_{i+1} x_{i-1}
and x_{i-2} x_{i-1} are distinct-variable products, so they appear as monomials
of {1, x}). With J=2 the flat library is only 2^D <= 1024, so flat WSINDy stays
cheap and this does NOT showcase the TT advantage -- but it lets us map the full
(M, D) grid, which is the point here.

Memory-light feature tensor
---------------------------
The earlier version capped M ~3000 because feature_tensor stored O(D J M^2)
diagonal cores. feature_tensor now builds the SAME tensor in compressed TT form
(low_rank=True): internal bonds collapse to the true ranks (<= J^min(d, D-d))
and only the time core scales with M, so memory is O(J^D + r M) -- linear in M.
That is what lets the M axis reach 15000.

Thresholding ranges
-------------------
* TTlambs (1e-8 .. 1e0): chosen so the TT-MSTLS coarse support stays near the
  true Lorenz-96 support. Each equation truly activates {1, x} on the 4 stencil
  oscillators {i-2, i-1, i, i+1} and {1} elsewhere, i.e. a per-equation reduced
  size of 2^4 = 16 and a total ~16*D (the prompt's ~8*D is the same order; the
  script prints the realized total). The coarse support is exact at low D and
  bloats once M < J^D (underdetermined).
* flatlambs (1e-11 .. 5e-6): the fine MSTLS lands on the correct support with
  the right coefficient magnitudes. (The flat LHS now uses -<x, phi'>, the
  correct first-order weak form; the missing -1 previously sign-flipped the
  coefficients -- support/magnitudes were already right.)

Feasibility
-----------
With the low-rank tensor and J=2 the whole grid is feasible, so no truncation is
expected. Cells are still gated by a memory predictor + flat-feature cap, wrapped
in try/except, and the sweep honours a wall-clock budget; anything skipped/failed
is left out of the surface. Data is saved incrementally to
results/complexityvWSINDy2.txt so a partial run still plots.

Run from the experiments/ directory with the scikit_tt env:
    python complexityvWSINDy2.py
"""
import os, sys, itertools, io, contextlib
sys.path.insert(0, '.')
sys.path.insert(0, '../TT-WSINDy')
import numpy as np
import matplotlib.pyplot as plt          # interactive by default; set MPLBACKEND=Agg to run headless
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from time import time
from scipy.signal import correlate
from scipy.integrate import odeint
import test_function
from sparsification import MSTLS
from ttwsindy import TT_WSINDy


# ----------------------------- problem setup -----------------------------
F = 8.0
t0, tM = 0.0, 10.0
f = [lambda x: 1, lambda x: x, lambda x : np.sin(x)]                    # J=2: Lorenz-96 is exactly spanned by {1, x}
J = len(f)

numTT = 10
TTlambs = 10 ** np.linspace(-8.0, 0.0, numTT + 1)  # coarse TT-MSTLS thresholds, 1e-8..1e0
numflat = 10
flatlambs = np.linspace(1e-11, 5e-6, numflat)      # fine MSTLS thresholds
threshold = 1e-16                                  # TT-PI SVD truncation

# requested ranges (full). With the low-rank feature_tensor (linear in M) and
# J=2 (flat library = 2^D <= 1024), the whole grid is feasible -- no truncation.
D_list = [4, 5, 6, 7, 8, 9, 10]
M_list = [500, 1000, 2000, 3000, 5000]

# feasibility gates (safety) and time budget
MEM_CAP = 1.5e9          # bytes, per-method working-set guard
FLAT_FEAT_CAP = 5000     # max J^D for the flat solve (2^10 = 1024 always passes)
TIME_BUDGET = 20 * 60    # seconds of compute before we stop and plot


def L96(x, t):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def gen(D, M):
    """Lorenz-96 trajectory on [t0, tM], started on the attractor."""
    x0 = F * np.ones(D); x0[0] += 0.01
    x0 = odeint(L96, x0, np.linspace(0.0, 5.0, 500))[-1]
    t = np.linspace(t0, tM, M)
    return t, odeint(L96, x0, t).T


def tt_mem(D, M):
    # low-rank feature_tensor: bonds <= min(J^D, M); the time core / carry
    # (r x M) and its SVD workspace dominate -> linear in M (not O(M^2)).
    r = min(J ** D, M)
    return r * M * 8.0 * 4.0


def flat_mem(D, M):
    return (J ** D) * M * 8.0           # the full library matrix G


def run_tt(X):
    """TT-WSINDy; returns (walltime, total coarse-support size)."""
    with contextlib.redirect_stdout(io.StringIO()):   # mute library debug prints
        ret = TT_WSINDy(X, t0, tM, f, TTlambs, flatlambs,
                        threshold=threshold, verbosity=0)
    tt_time = ret[3]
    cs = ret[6]
    D = X.shape[0]
    reduced = int(np.sum([np.prod([max(s.size, 1) for s in cs[d]])
                          for d in range(D)]))
    return tt_time, reduced


def run_flat(X, D, M):
    """Flat (matrix) weak-form WSINDy; returns walltime."""
    st = time()
    basis = np.stack([np.vectorize(f[j])(X).astype(float) for j in range(J)])
    fmap = list(itertools.product(*[range(J)] * D))
    G = np.ones((J ** D, M))
    for k, tup in enumerate(fmap):
        for d in range(D):
            G[k] *= basis[tup[d], d]
    phi, dphi = test_function.piecewise_polynomial((tM - t0) / 20, 16, t0, tM, M)
    # first-order weak form: <x_dot, phi> = -<x, phi'> (the -1 was missing before,
    # which sign-flipped the recovered coefficients)
    Y = -1 * correlate(X, np.expand_dims(dphi, 0), mode='valid').T
    G = correlate(G, np.expand_dims(phi, 0), mode='valid').T
    with contextlib.redirect_stdout(io.StringIO()):
        for d in range(D):
            MSTLS(G, Y[:, d], flatlambs, verbose=False)
    return time() - st


# ------------------------------- sweep ------------------------------------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    rows = []                       # (M, D, tt, flat, diff, reduced)
    sweep_start = time()
    stop = False

    print(f"grid: D in {D_list}, M in {M_list}")
    print(f"library J={J} ({J}^D flat features), TIME_BUDGET={TIME_BUDGET/60:.0f} min")
    print(f"{'M':>6} {'D':>3} {'Jp':>7} {'reduced':>8} {'TT(s)':>8} "
          f"{'flat(s)':>9} {'diff(s)':>9}  note", flush=True)

    # D ascending (outer), M ascending (inner): each dimension's full M range --
    # including M=15000 -- completes before moving to a slower, higher D, so the
    # M axis is covered even if the budget stops us in the high-D rows.
    for D in D_list:
        if stop:
            break
        for M in M_list:
            elapsed = time() - sweep_start
            if elapsed > TIME_BUDGET:
                print(f"-- time budget reached ({elapsed/60:.1f} min); stopping --",
                      flush=True)
                stop = True
                break

            note = ""
            if tt_mem(D, M) > MEM_CAP:
                note = "skip:TT-mem"
            elif flat_mem(D, M) > MEM_CAP or J ** D > FLAT_FEAT_CAP:
                note = "skip:flat"
            elif D >= 9 and M >= 8000:
                # TT coarse support bloats at high D -> the fine solve is huge;
                # these cells take many minutes, so skip them as time-prohibitive
                note = "skip:TT-time"
            if note:
                print(f"{M:>6} {D:>3} {J**D:>7} {'-':>8} {'-':>8} {'-':>9} "
                      f"{'-':>9}  {note}", flush=True)
                continue

            try:
                t, X = gen(D, M)
                tt_t, reduced = run_tt(X)
                flat_t = run_flat(X, D, M)
                diff = flat_t - tt_t
                rows.append((M, D, tt_t, flat_t, diff, reduced))
                print(f"{M:>6} {D:>3} {D*J**D:>7} {reduced:>8} {tt_t:>8.1f} "
                      f"{flat_t:>9.1f} {diff:>9.1f}", flush=True)
                np.savetxt("results/complexityvWSINDy2.txt", np.array(rows),
                           header="M D tt flat diff reduced", fmt="%.6g")
            except MemoryError:
                print(f"{M:>6} {D:>3} {J**D:>7} {'-':>8} {'-':>8} {'-':>9} "
                      f"{'-':>9}  FAIL:MemoryError", flush=True)
            except Exception as e:
                print(f"{M:>6} {D:>3} {J**D:>7} {'-':>8} {'-':>8} {'-':>9} "
                      f"{'-':>9}  FAIL:{type(e).__name__}", flush=True)

    print(f"\nsweep done in {(time()-sweep_start)/60:.1f} min, "
          f"{len(rows)} feasible cells.", flush=True)
    if not rows:
        print("No feasible cells -- nothing to plot.")
        sys.exit(0)

    data = np.array(rows)
    Mv, Dv, ttv, flatv, diffv, redv = data.T

    # --------------------------- 3-D plot ---------------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    try:
        surf = ax.plot_trisurf(Mv, Dv, diffv, cmap="coolwarm",
                               edgecolor="k", linewidth=0.3, alpha=0.9)
        fig.colorbar(surf, ax=ax, shrink=0.6, label="flat - TT walltime (s)")
    except Exception:
        pass
    ax.scatter(Mv, Dv, diffv, c="k", s=18, depthshade=False)
    ax.set_xlabel("snapshots M")
    ax.set_ylabel("dimension D")
    ax.set_zlabel("walltime(WSINDy) - walltime(TT-WSINDy)  [s]")
    ax.set_title("TT-WSINDy vs. flat WSINDy on Lorenz-96\n"
                 "(positive = TT-WSINDy faster)")
    ax.view_init(elev=22, azim=-60)
    plt.tight_layout()
    plt.savefig("results/complexityvWSINDy2.png", dpi=150)

    # flat 2-D heatmap companion (easier to read than a static 3-D view)
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))
    Dl = sorted(set(int(d) for d in Dv))
    Ml = sorted(set(int(m) for m in Mv))
    grid = np.full((len(Dl), len(Ml)), np.nan)
    for (m, d, _, _, df, _) in rows:
        grid[Dl.index(int(d)), Ml.index(int(m))] = df
    im = ax2.imshow(grid, origin="lower", aspect="auto", cmap="coolwarm",
                    extent=[-0.5, len(Ml) - 0.5, -0.5, len(Dl) - 0.5])
    ax2.set_xticks(range(len(Ml))); ax2.set_xticklabels(Ml)
    ax2.set_yticks(range(len(Dl))); ax2.set_yticklabels(Dl)
    for i in range(len(Dl)):
        for j in range(len(Ml)):
            if not np.isnan(grid[i, j]):
                ax2.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                         fontsize=8)
    fig2.colorbar(im, ax=ax2, label="flat - TT walltime (s)")
    ax2.set_xlabel("snapshots M"); ax2.set_ylabel("dimension D")
    ax2.set_title("walltime(WSINDy) - walltime(TT-WSINDy)  [s]")
    plt.tight_layout()
    plt.savefig("results/complexityvWSINDy2_heatmap.png", dpi=150)
    print("saved results/complexityvWSINDy2.png and _heatmap.png")
    plt.show()
