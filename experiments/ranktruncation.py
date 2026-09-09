"""
Rank truncation demonstration.

Compares the two feature-tensor constructions -- the original dense build and
the low-rank left-to-right SVD sweep (low_rank=True) -- at matched accuracy:
both arms are compressed with the same eps, set nonzero only so scikit-tt takes
the compressed path but far below any nonzero relative singular value, so
nothing is ever truncated and the two arms return the same tensor.

Sampling follows the walltime_support_vs_D regime rather than a fixed time
window: the step is a fixed DT and the test-function radius is pinned at
TAPS*DT = 0.5 time units (r_frac = TAPS/M), so a larger M buys a LONGER
trajectory -- more of the attractor -- rather than a finer grid or a wider phi.
That is what makes the true support recoverable at every D in the scan, D=9
included; the earlier fixed-window regime (tM=30, r_frac=1/40, no burn-in) gave
phi a 250-tap support at M=10000 and only 30 time units of attractor, and the
coarse pass could not prune to the true product support at high D.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, 'TT-WSINDy'))
sys.path.insert(0, os.path.join(_ROOT, 'WSINDy'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy.integrate import odeint
from ttwsindy import TT_WSINDy

# ------------- model -------------
F = 8.0                 # Lorenz-96 forcing
C = 1.0                 # linear damping coefficient (-C x_i term)

def L96(x, t):
    """Lorenz 96 model with constant forcing F."""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - C*x + F

f    = [lambda x: 1, lambda x: x]
fstr = [lambda n: '', lambda n: f'x_{n}']
if len(fstr) != len(f):
    raise ValueError('fstr does not correspond to f')

# ------------- sampling / test function -------------
T0     = 0.0            # sampling starts here
DT     = 0.05           # sampling step, FIXED across both sweeps
BURN   = 20.0           # integrated away before sampling, to land on the attractor
TAPS   = 10             # test-function radius in samples: r_frac = TAPS/M, so
                        # phi spans TAPS*DT = 0.5 time units at every M
DEGREE = 16             # piecewise-polynomial test-function degree

TTlambs   = np.linspace(1e-5, 5e-1, 10)   # coarse TT-MSTLS thresholds
flatlambs = np.linspace(1e-5, 1e-1, 10)   # fine matrix-MSTLS thresholds

# nonzero (so scikit-tt takes the compressed path) but below any nonzero
# relative singular value, so nothing is ever truncated. Shared by both arms.
EPS = 1e-300

# ------------- sweeps -------------
RESULTS = os.path.join(_HERE, 'results')
DATA    = f'{RESULTS}/ranktruncation.txt'

MS        = [500, 1000, 2000, 3000, 5000, 10000, 20000, 40000]
D_FIXED   = 8

DS        = [4, 5, 6, 7, 8, 9]
# Measured: the exact product support is recovered at every D in DS at this M
# (clean data, 10-tap phi) -- D=9 included, which is what this M was chosen for.
# It also keeps the dense arm inside MEM_BUDGET_GB at D=9
# (8*2*5000^2*8 = 3.0 GB), so both curves span the whole constant-M panel.
M_FIXED   = 5000

MEM_BUDGET_GB = 10.5

def fits_in_memory(D, M):
    gb = (D - 1) * len(f) * M * M * 8 / 2**30
    return gb <= MEM_BUDGET_GB

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

def simulate(D, M, seed=0):
    """Integrate Lorenz-96 to a D x M data matrix, sampled on the attractor.

    The step is DT, so the window tM - t0 = DT*M lengthens with M rather than
    the grid refining. Returns the data together with the sampling window and
    the test-function radius fraction TAPS/M, which pins phi at TAPS*DT time
    units for every M.
    """
    tM = T0 + DT*M
    rng = np.random.default_rng(seed)
    x0 = (F/C)*np.ones(D) + 0.01*rng.standard_normal(D)
    x0 = odeint(L96, x0, np.linspace(0, BURN, 1000))[-1]     # burn in to attractor
    X = odeint(L96, x0, np.linspace(T0, tM, M)).T
    return X, T0, tM, TAPS/M

def time_ttwsindy(X, t0, tM, r_frac, low_rank, threshold):
    """Run one-pass TT-WSINDy once; return (walltime, supps, feature_maps)."""
    ret = TT_WSINDy(X, t0, tM, f, TTlambs, flatlambs,
                    testfn=('piecewise_polynomial', r_frac, DEGREE, 1),
                    verbosity=0, threshold=threshold,
                    low_rank=low_rank, one_pass=True)
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
        X, t0, tM, r_frac = simulate(D, M)
        print(f'\nD = {D}, M = {M}, t in [{t0:g},{tM:g}], dt = {DT}, '
              f'phi radius = {TAPS*DT:g} ({TAPS} taps)')
        if run_dense(D, M):
            print('Original construction')
            for i in range(nAvg):
                dense[k, i], supps, fmaps = time_ttwsindy(
                    X, t0, tM, r_frac, False, EPS)
            print('original discovered support:')
            print_supp(fstr, supps, fmaps)
        else:
            print(f'Original construction skipped (needs '
                  f'{(D - 1) * len(f) * M * M * 8 / 2**30:.1f} GB of cores)')
        print('Low-rank construction')
        for i in range(nAvg):
            lowrank[k, i], supps, fmaps = time_ttwsindy(
                X, t0, tM, r_frac, True, EPS)
        print('low-rank discovered support:')
        print_supp(fstr, supps, fmaps)
    return dense, lowrank

def save_data(m_dense, m_low, d_dense, d_low):
    combined = np.vstack([
        np.hstack([m_dense, m_low]),
        np.hstack([d_dense, d_low]),
    ])
    header = (
        f'Rank-truncation walltimes (seconds), eps = {EPS:.0e} in both arms.\n'
        f'Lorenz-96, dt = {DT}, phi radius = {TAPS} samples = {TAPS*DT:g} time '
        f'units, degree {DEGREE}.\n'
        f'Rows 0..{len(MS) - 1}: vs-M sweep at D={D_FIXED}, M={MS}\n'
        f'Rows {len(MS)}..{len(MS) + len(DS) - 1}: vs-D sweep at M={M_FIXED}, D={DS}\n'
        'Columns: first half = original construction, second half = low-rank.\n'
        'A zero marks a point where the original construction was skipped.'
    )
    np.savetxt(DATA, combined, header=header)

def load_data():
    data = np.loadtxt(DATA).reshape(len(MS) + len(DS), -1)
    nAvg = data.shape[1] // 2
    m, d = data[:len(MS)], data[len(MS):]
    return (m[:, :nAvg].mean(axis=1), m[:, nAvg:].mean(axis=1),
            d[:, :nAvg].mean(axis=1), d[:, nAvg:].mean(axis=1))

def extrapolate_dense(M_meas, t_meas, M_pred):
    """Extrapolate the original construction's walltime past its memory wall.

    Fits t(M) = a + b*M + c*M^2 by least squares to the measured points: a
    fixed per-D overhead, a term linear in M (the correlations and the target),
    and the quadratic term from the dense M x J x M cores, which is what
    eventually dominates. Over the measured range 500 <= M <= 10000 at D=8 the
    fit holds every point to within 5.4%; the pure power law and the pure
    a + c*M^2 model are both markedly worse.
    """
    A = np.vstack([np.ones_like(M_meas), M_meas, M_meas**2]).T
    coef, *_ = np.linalg.lstsq(A, t_meas, rcond=None)
    return np.vstack([np.ones_like(M_pred), M_pred, M_pred**2]).T @ coef

# ------------- main -------------
if __name__ == '__main__':

    recompute_data = False   # cached in results/ranktruncation.txt
    nAvg = 10

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

    # --- vs M (D fixed): the original construction stops at its memory wall,
    # and is extrapolated (dotted) over the M it cannot be run at
    measured = np.array([fits_in_memory(D_FIXED, M) for M in MS])
    MS_arr = np.array(MS, dtype=float)
    t_pred = extrapolate_dense(MS_arr[measured], m_dense[measured],
                               MS_arr[~measured])
    axM.plot(MS_arr[measured], m_dense[measured], marker='o', color='blue',
             label='Original construction')
    # prepend the last measured point so the dotted segment continues the solid one
    axM.plot(np.r_[MS_arr[measured][-1], MS_arr[~measured]],
             np.r_[m_dense[measured][-1], t_pred],
             marker='o', mfc='none', color='blue', ls='dotted',
             label='Original construction (extrapolated)')
    axM.plot(MS_arr, m_low, marker='o', color='orange',
             label='Low-rank construction')
    axM.set_xlim(MS_arr[0] * 0.85, MS_arr[-1] * 1.15)
    axM.set_title(f'Lorenz 96, D = {D_FIXED}')
    axM.set_xlabel('M')
    axM.set_ylabel('walltime (s)')
    #axM.set_xscale('log')
    axM.set_yscale('log')
    axM.grid(True, which='both', ls=':', alpha=0.5)
    axM.legend()

    # --- vs D (M fixed); the dense arm is drawn only where it was measured
    dense_run = np.array([fits_in_memory(D, M_FIXED) for D in DS])
    DS_arr = np.array(DS, dtype=float)
    axD.plot(DS_arr[dense_run], d_dense[dense_run], marker='o', color='blue',
             label='Original construction')
    axD.plot(DS_arr, d_low, marker='o', color='orange',
             label='Low-rank construction')
    axD.set_title(f'Lorenz 96, M = {M_FIXED}')
    axD.set_xlabel('D')
    axD.set_ylabel('walltime (s)')
    axD.set_yscale('log')
    axD.grid(True, which='both', ls=':', alpha=0.5)
    axD.legend()

    # readable decimal labels on the log axes, rather than a lone 10^0.
    # ScalarFormatter picks its precision from the tick set and renders 0.5 as
    # "0" on the narrower left panel, so format each tick with %g instead.
    for ax in (axM, axD):
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())

    #fig.suptitle('Feature-tensor construction walltimes, both arms compressed '
    #             f'($\\epsilon = 10^{{{int(np.log10(EPS))}}}$, nothing truncated)')
    fig.tight_layout()
    fig.savefig(f'{RESULTS}/ranktruncation.png', dpi=150)
    plt.show()
