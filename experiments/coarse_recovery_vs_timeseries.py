"""
Coarse-support recovery vs. time-series length on Lorenz-96 (D=9, J=2).

Claim demonstrated
------------------
The TT-WSINDy coarse pass (TT-MSTLS) recovers the *correct* coarse support for
Lorenz-96 from a LONG trajectory (t in [0, 30]) but NOT a short one (t in
[0, 10]): the short trajectory drops the weak self-term -x_i in every equation.

Why this happens
----------------
Lorenz-96,  dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F,  is spanned exactly
by the separable library f = {1, x} (J=2), so the flat candidate count is only
2^9 = 512 per equation. The true row-i model has four terms:

    +1 * x_{i+1} x_{i-1}    -1 * x_{i-2} x_{i-1}    -1 * x_i    +F * 1

Among these, -x_i is the *weak* one: its column energy is far below the
quadratic-coupling terms, so it sits near the spurious-feature floor. The coarse
pass keeps a feature when its normalized weight exceeds a threshold. With a short
trajectory the chaotic attractor is under-explored, the spurious features carry
finite-sample weight, and the floor sits ABOVE the weak -x_i weight -> no
threshold separates them and -x_i is dropped. A longer trajectory is more
ergodic: spurious weights decay toward zero, a clean gap opens, and the coarse
pass keeps all four true terms.

The exact coarse support has 2^4 = 16 surviving candidates per equation (the
four stencil oscillators {i-2, i-1, i, i+1} carry {1, x}; the other five carry
{1} only), i.e. 16 * D = 144 total for D = 9. Full recovery means recall = 4*D
true terms AND support size = 144.

Conditioning note
-----------------
The test-function window must be NARROW for the large D=9 library (see
weakvstrong_lorenz96.py): a wide window is poorly conditioned once the candidate
count is large, which by itself inflates the spurious floor. We use r_frac =
1/80. The coarse pass here is the fast single-solve variant (one_pass=True).

Output
------
results/coarse_recovery_vs_timeseries.png  (recall and support size vs t_end)
results/coarse_recovery_vs_timeseries.txt  (raw data)

Run from the experiments/ directory with the scikit_tt env:
    python coarse_recovery_vs_timeseries.py
"""
import os, sys
sys.path.insert(0, '.')
sys.path.insert(0, '../TT-WSINDy')
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.integrate import odeint
from ttwsindy import TT_WSINDy

# ----------------------------- parameters --------------------------------
F = 8.0                                   # chaotic forcing
D = 9                                      # state dimension (2^9 = 512 candidates)
f = [lambda x: 1.0, lambda x: x]          # J=2: {1, x} spans Lorenz-96 exactly
J = len(f)
dt = 0.01                                  # snapshot spacing (M = t_end / dt)
r_frac = 1.0 / 80.0                        # NARROW test-function window
degree = 16
threshold = 1e-10                          # TT-PI SVD truncation
TTlambs = 10.0 ** ((3.7 / 20) * np.arange(0, 21) - 4)   # coarse thresholds, 1e-4..0.5
flatlambs = np.linspace(1e-11, 1e-1, 10)   # fine MSTLS thresholds (unused for coarse metric)

t_ends = [6, 8, 10, 13, 16, 20, 24, 30]    # trajectory lengths to sweep
n_seeds = 5                                # trajectories per length (chaos -> average)
burn = 20.0                                # time integrated and discarded (land on attractor)


def l96_rhs(x, t):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def x_true_dims(d):
    """Dimensions where 'x' is a true feature of equation d."""
    return {d, (d + 1) % D, (d - 1) % D, (d - 2) % D}


def l96_true_terms(d):
    """Set of true feature tuples (powers over {0,1}) for equation d."""
    terms = {tuple([0] * D)}                                  # +F (constant)
    t = [0] * D; t[d] = 1; terms.add(tuple(t))               # -x_d
    t = [0] * D; t[(d + 1) % D] = 1; t[(d - 1) % D] = 1; terms.add(tuple(t))   # x_{d+1}x_{d-1}
    t = [0] * D; t[(d - 2) % D] = 1; t[(d - 1) % D] = 1; terms.add(tuple(t))   # x_{d-2}x_{d-1}
    return terms


def true_coarse_size():
    """Size of the exact coarse (product) support, summed over equations."""
    return int(sum(np.prod([2 if e in x_true_dims(d) else 1 for e in range(D)])
                   for d in range(D)))


def coarse_recall(coarse_supps):
    """(# true terms surviving the product coarse support, total true terms)."""
    kept = total = 0
    for d in range(D):
        surv = [set(int(j) for j in coarse_supps[d][e]) for e in range(D)]
        for tup in l96_true_terms(d):
            kept += all(tup[e] in surv[e] for e in range(D))
            total += 1
    return kept, total


def coarse_size(coarse_supps):
    return int(np.sum([np.prod([max(s.size, 1) for s in coarse_supps[d]])
                       for d in range(D)]))


def trajectory(t_end, M, seed):
    rng = np.random.default_rng(seed)
    x0 = F * np.ones(D) + 0.01 * rng.standard_normal(D)
    x0 = odeint(l96_rhs, x0, np.linspace(0.0, burn, 1000))[-1]   # on attractor
    return odeint(l96_rhs, x0, np.linspace(0.0, t_end, M)).T


# ------------------------------- sweep -----------------------------------
recompute = False       # set False to re-plot from the saved .txt without re-running
DATA = "results/coarse_recovery_vs_timeseries.txt"

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    n_true = 4 * D                              # 36 true terms total
    true_size = true_coarse_size()             # 144 for D=9

    if recompute:
        print(f"Lorenz-96 D={D}, J={J} ({J**D} candidates/eqn), narrow window r=1/{int(1/r_frac)}")
        print(f"true terms = {n_true}, exact coarse support size = {true_size}")
        print(f"{'t_end':>6} {'M':>6} {'recall(mean)':>13} {'size(mean)':>11}  per-seed recall", flush=True)
        rows = []
        for t_end in t_ends:
            M = int(round(t_end / dt))
            for seed in range(n_seeds):
                X = trajectory(t_end, M, seed)
                r = TT_WSINDy(X, 0.0, t_end, f, TTlambs, flatlambs,
                              testfn=('piecewise_polynomial', r_frac, degree, 1),
                              threshold=threshold, verbosity=0, low_rank=True,
                              one_pass=True)
                k, _ = coarse_recall(r[6])
                rows.append((t_end, M, seed, k, coarse_size(r[6])))
            sub = np.array([row for row in rows if row[0] == t_end])
            print(f"{t_end:>6} {int(t_end/dt):>6} {(sub[:,3]/n_true).mean():>13.3f} "
                  f"{sub[:,4].mean():>11.0f}  {[int(x) for x in sub[:,3]]}", flush=True)
            np.savetxt(DATA, np.array(rows),
                       header="t_end M seed recall_count coarse_size", fmt="%.6g")
        rows = np.array(rows)
    else:
        rows = np.loadtxt(DATA)

    # aggregate per trajectory length
    t = np.unique(rows[:, 0])
    recall_mean = np.array([(rows[rows[:, 0] == tv, 3] / n_true).mean() for tv in t])
    recall_lo = np.array([(rows[rows[:, 0] == tv, 3] / n_true).min() for tv in t])
    recall_hi = np.array([(rows[rows[:, 0] == tv, 3] / n_true).max() for tv in t])
    size_mean = np.array([rows[rows[:, 0] == tv, 4].mean() for tv in t])
    size_lo = np.array([rows[rows[:, 0] == tv, 4].min() for tv in t])
    size_hi = np.array([rows[rows[:, 0] == tv, 4].max() for tv in t])

    # onset of exact recovery (recall==1 for every seed at that length)
    exact = np.array([np.all(rows[rows[:, 0] == tv, 3] == n_true) for tv in t])
    t_star = t[exact][0] if exact.any() else None

    # ------------------------------ plot ---------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)
    xlo, xhi = t.min() - 1, t.max() + 1.5

    for ax in (ax1, ax2):
        if t_star is not None:
            ax.axvspan(xlo, t_star, color="red", alpha=0.05)
            ax.axvspan(t_star, xhi, color="green", alpha=0.06)
        #for tv, col in [(10, "red"), (30, "green")]:
            #ax.axvline(tv, ls=":", color=col, lw=1.5)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.set_xlim(xlo, xhi)

    ax1.fill_between(t, recall_lo, recall_hi, alpha=0.2, color="C0")
    ax1.plot(t, recall_mean, "o-", color="C0", label=f"coarse recall (mean of {n_seeds} trajectories)")
    ax1.axhline(1.0, ls="--", color="green", lw=1, label="all true terms kept")
    ax1.set_ylabel("fraction of true terms\nkept by coarse pass")
    ax1.set_ylim(0.6, 1.03)
    ax1.legend(loc="lower right")
    ax1.set_title(f"Lorenz-96 D={D}, J={J} ({{1, x}}): coarse-support recovery vs. time-series length")
    #ax1.text(10, 1.005, " t=10: misses $-x_i$", color="red", fontsize=9, ha="left", va="top")
    #ax1.text(30, 1.005, "t=30: exact ", color="green", fontsize=9, ha="right", va="top")

    ax2.fill_between(t, size_lo, size_hi, alpha=0.2, color="C1")
    ax2.plot(t, size_mean, "s-", color="C1", label="coarse support size (mean)")
    ax2.axhline(true_size, ls="--", color="green", lw=1, label=f"true support size = {true_size}")
    ax2.set_yscale("log")
    ax2.set_ylabel("coarse support size")
    ax2.set_xlabel(r"trajectory length  $t_{end}$   (data on $[0, t_{end}]$,  $dt=%.2g$)" % dt)
    ax2.legend(loc="upper right")
    #if t_star is not None:
    #    ax2.text(t_star + 0.3, size_mean.max(), f"exact recovery\nfor $t_{{end}}\\geq{t_star:g}$",
    #             color="green", fontsize=9, ha="left", va="top")

    plt.tight_layout()
    plt.savefig("results/coarse_recovery_vs_timeseries.png", dpi=150)
    print("saved results/coarse_recovery_vs_timeseries.png")
    plt.show()
