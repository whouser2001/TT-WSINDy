"""
Rank truncation demonstration.

Wall-clock comparison of the two feature-tensor *constructions* for one-pass
TT-WSINDy on Lorenz-96, swept along two axes and shown as side-by-side plots:

  * vs M : number of time points, at fixed dimension D = 8
  * vs D : system dimension,      at fixed M = 2000

Both arms run the same compressed solve path. The original construction
(low_rank=False) materializes the diagonal (M, J, 1, M) cores and is then
compressed by the orthonormalization sweep inside TT.__init__; the low-rank
construction (low_rank=True) reaches the same train directly, via a
left-to-right SVD sweep over a small carry matrix. So the gap plotted here is
the cost of *materializing* the O(D J M^2) intermediate, not the cost of
carrying redundant bonds through TT_PI.

That distinction is why EPS is 1e-300 rather than 0. scikit-tt's rank
reduction is gated on `threshold != 0`, so threshold=0 skips the
orthonormalization sweep in TT.__init__ altogether and leaves the last bond at
its stored size M -- TT_PI then pays an (M x Mp) SVD on the time core instead
of a (J^D x Mp) one, which is a defect of the *gate*, not of the construction.
EPS = 1e-300 opens the gate while keeping every nonzero singular value: the
filter is `s/s[0] > EPS`, so only exact zeros are dropped. Verified lossless
against threshold=0 at D=8 (identical ranks, 256 singular values, cond
4.103e11); at D=9 it keeps all 512 where eps=1e-16 would drop 62.

Note EPS also reaches `ttutils.truncated_svd`, whose randomized branch only
engages for min(shape) > 700, i.e. J^D > 700 -> D >= 10 at J = 2. Above the
D range swept here, a tiny EPS would stop that branch from ever exiting early
and degrade it to a full SVD.

The original construction is skipped wherever its (D-1) J M^2 cores would not
fit in MEM_BUDGET_GB; its curve simply stops at that wall, which is shaded. No
walltime is extrapolated past it -- total walltime carries a large M-independent
floor (the flat MSTLS pass), so a log-log fit to these points describes the
floor rather than the O(M^2) materialization and would understate the gap.
Set recompute_data = True to regenerate the timings, otherwise the cached
results in results/ranktruncation.txt are loaded and plotted.
"""
import os, sys
sys.path.insert(0, '.')
sys.path.insert(0, '../TT-WSINDy')
sys.path.insert(0, '../WSINDy')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
from scipy.integrate import odeint
from ttwsindy import TT_WSINDy

# ------------------------------------------------------------------ model
F = 8

def L96(x, t):
    """Lorenz 96 model with constant forcing F."""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

f    = [lambda x: 1, lambda x: x]
fstr = [lambda n: '', lambda n: f'x_{n}']
if len(fstr) != len(f):
    raise ValueError('fstr does not correspond to f')

TTlambs   = np.linspace(1e-5, 1e-1, 10)
flatlambs = np.linspace(1e-8, 1e-1, 10)

# nonzero (so scikit-tt takes the compressed path) but below any nonzero
# relative singular value, so nothing is ever truncated. Shared by both arms.
EPS = 1e-300

# ------------------------------------------------------------------ sweeps
RESULTS = 'results'
DATA    = f'{RESULTS}/ranktruncation.txt'

MS        = [500, 1000, 2000, 3000, 5000, 10000, 20000]  # vs-M sweep, at D_FIXED
D_FIXED   = 8

DS        = [4, 5, 6, 7, 8, 9]                           # vs-D sweep, at M_FIXED
M_FIXED   = 5000

MEM_BUDGET_GB = 10.5     # cap on the original construction's core storage


def fits_in_memory(D, M):
    """
    Whether the original construction's cores fit the budget.

    It holds one (1, J, 1, M) core and (D-1) dense (M, J, 1, M) cores, plus an
    (M, Mp) time core; the (D-1) M^2 term dominates.
    """
    gb = (D - 1) * len(f) * M * M * 8 / 2**30
    return gb <= MEM_BUDGET_GB


# ------------------------------------------------------------------ helpers
def print_supp(fstr, supps, feature_maps):
    D = len(supps)
    for d1 in range(D):
        supp = supps[d1]
        feature_map = feature_maps[d1]
        line = f"x_{d1 + 1}' : "
        if supp is not None:
            for k in supp:
                substr = ''
                for d2 in range(D):
                    substr += fstr[feature_map[k][d2]](d2 + 1)
                if substr == '':
                    substr += '1'
                line += substr
                if k != supp[-1]:
                    line += '  '
        print(line)


def simulate(D, M, t0=0, tM=30):
    """Integrate Lorenz-96 to a D x M data matrix."""
    t = np.linspace(t0, tM, M)
    x0 = F * np.ones(D)
    x0[0] += 0.01
    return odeint(L96, x0, t).T, t0, tM


def time_ttwsindy(X, t0, tM, low_rank, threshold):
    """Run one-pass TT-WSINDy once; return (walltime, supps, feature_maps)."""
    ret = TT_WSINDy(X, t0, tM, f, TTlambs, flatlambs, verbosity=0,
                    threshold=threshold, low_rank=low_rank, one_pass=True)
    return ret[3], ret[1], ret[2]


def sweep(cases, nAvg, run_dense):
    """
    Time both constructions over a list of (D, M) cases.

    run_dense : callable (D, M) -> bool. When it returns False the original
    construction is skipped and its walltime is left at 0.

    Returns (dense, lowrank), each an array of shape (len(cases), nAvg).
    """
    dense   = np.zeros((len(cases), nAvg))
    lowrank = np.zeros((len(cases), nAvg))
    for k, (D, M) in enumerate(cases):
        X, t0, tM = simulate(D, M)
        print(f'\nD = {D}, M = {M}')
        if run_dense(D, M):
            print('Original construction')
            for i in range(nAvg):
                dense[k, i], supps, fmaps = time_ttwsindy(X, t0, tM, False, EPS)
            print('original discovered support:')
            print_supp(fstr, supps, fmaps)
        else:
            print(f'Original construction skipped (needs '
                  f'{(D - 1) * len(f) * M * M * 8 / 2**30:.1f} GB of cores)')
        print('Low-rank construction')
        for i in range(nAvg):
            lowrank[k, i], supps, fmaps = time_ttwsindy(X, t0, tM, True, EPS)
        print('low-rank discovered support:')
        print_supp(fstr, supps, fmaps)
    return dense, lowrank


def save_data(m_dense, m_low, d_dense, d_low):
    """
    Write both sweeps to a single text file. One row per sweep point:
    the vs-M rows first, then the vs-D rows. The first nAvg columns hold
    the original-construction trials, the last nAvg the low-rank trials.
    """
    combined = np.vstack([
        np.hstack([m_dense, m_low]),
        np.hstack([d_dense, d_low]),
    ])
    header = (
        f'Rank-truncation walltimes (seconds), eps = {EPS:.0e} in both arms.\n'
        f'Rows 0..{len(MS) - 1}: vs-M sweep at D={D_FIXED}, M={MS}\n'
        f'Rows {len(MS)}..{len(MS) + len(DS) - 1}: vs-D sweep at M={M_FIXED}, D={DS}\n'
        'Columns: first half = original construction, second half = low-rank.\n'
        'A zero marks a point where the original construction was skipped.'
    )
    np.savetxt(DATA, combined, header=header)


def load_data():
    """Inverse of save_data; returns averaged (m_dense, m_low, d_dense, d_low)."""
    data = np.loadtxt(DATA).reshape(len(MS) + len(DS), -1)
    nAvg = data.shape[1] // 2
    m, d = data[:len(MS)], data[len(MS):]
    return (m[:, :nAvg].mean(axis=1), m[:, nAvg:].mean(axis=1),
            d[:, :nAvg].mean(axis=1), d[:, nAvg:].mean(axis=1))


def mem_wall_M(D):
    """Largest M whose original-construction cores fit MEM_BUDGET_GB."""
    return np.sqrt(MEM_BUDGET_GB * 2**30 / ((D - 1) * len(f) * 8))


# ------------------------------------------------------------------ main
if __name__ == '__main__':

    recompute_data = False   # cached in results/ranktruncation.txt; True re-times everything
    nAvg = 3

    if recompute_data:
        # vs M, at D = D_FIXED (skip the original construction once it OOMs)
        m_dense, m_low = sweep(
            [(D_FIXED, M) for M in MS], nAvg, run_dense=fits_in_memory,
        )
        # vs D, at M = M_FIXED
        d_dense, d_low = sweep(
            [(D, M_FIXED) for D in DS], nAvg, run_dense=fits_in_memory,
        )
        save_data(m_dense, m_low, d_dense, d_low)

    m_dense, m_low, d_dense, d_low = load_data()

    fig, (axM, axD) = plt.subplots(1, 2, figsize=(12, 5))

    # --- vs M (D fixed): the original construction stops at its memory wall
    measured = np.array([fits_in_memory(D_FIXED, M) for M in MS])
    MS_arr = np.array(MS, dtype=float)
    wall = mem_wall_M(D_FIXED)
    axM.axvspan(wall, MS_arr[-1] * 1.15, color='blue', alpha=0.06)
    axM.axvline(wall, color='blue', ls='dashed', lw=1)
    axM.plot(MS_arr[measured], m_dense[measured], marker='o', color='blue',
             label='Original construction')
    axM.plot(MS_arr, m_low, marker='o', color='orange',
             label='Low-rank construction')
    axM.annotate(f'original construction\n> {MEM_BUDGET_GB:.0f} GB of cores',
                 xy=(wall * 1.15, m_low[0]), color='blue', fontsize=9, va='bottom')
    axM.set_xlim(MS_arr[0] * 0.85, MS_arr[-1] * 1.15)
    axM.set_title(f'Lorenz96, D = {D_FIXED}')
    axM.set_xlabel('M')
    axM.set_ylabel('walltime (s)')
    #axM.set_xscale('log')
    axM.set_yscale('log')
    axM.grid(True, which='both', ls=':', alpha=0.5)
    axM.legend()

    # --- vs D (M fixed)
    axD.plot(DS, d_dense, marker='o', color='blue', label='Original construction')
    axD.plot(DS, d_low, marker='o', color='orange', label='Low-rank construction')
    axD.set_title(f'Lorenz96, M = {M_FIXED}')
    axD.set_xlabel('D')
    axD.set_ylabel('walltime (s)')
    axD.set_yscale('log')
    axD.grid(True, which='both', ls=':', alpha=0.5)
    axD.legend()

    # readable decimal labels on the log axes, rather than a lone 10^0
    for ax, ticks in ((axM, [0.4, 0.6, 1, 2, 4]), (axD, [0.06, 0.1, 0.3, 1, 3])):
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

    fig.suptitle('Feature-tensor construction walltimes, both arms compressed '
                 f'($\\epsilon = 10^{{{int(np.log10(EPS))}}}$, nothing truncated)')
    fig.tight_layout()
    fig.savefig(f'{RESULTS}/ranktruncation.png', dpi=150)
    plt.show()
