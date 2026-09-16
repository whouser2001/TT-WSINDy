"""
Walltime and support-recovery scan vs system dimension D for the standard
Lorenz-96 identification problem.
"""
import os, sys, itertools
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, 'TT-WSINDy'))
sys.path.insert(0, os.path.join(_ROOT, 'WSINDy'))
RESULTS = os.path.join(_HERE, 'results')
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.signal import correlate
from scipy.integrate import odeint
import test_function
import exputils as xu
from ttwsindy import TT_WSINDy
from wsindy import wsindy


F = 8.0                     # Lorenz-96 forcing
C = 1.0                     # linear damping coefficient (-C x_i term)
t0 = 0.0                    
DT = 0.1                    # sampling step
M = 20000                   # number of time snapshots
tM = t0 + DT*M              
D_MIN, D_MAX = 5, 9         # dimension scan (inclusive)
NUM_TRIALS = 3              # trials averaged per datapoint

f = [lambda x: 1.0, lambda x: x]
J = len(f)

NOISE_LEVEL = 1e-3
FILE_SUFFIX = '' if NOISE_LEVEL == 0 else f'_noise{NOISE_LEVEL*100:g}pct'
NOISE_LABEL = ('clean data' if NOISE_LEVEL == 0
               else f'{NOISE_LEVEL*100:g}% noise')

TAPS = 10                   # time points spanned by the test-function radius
R_FRAC, DEGREE = TAPS/M, 16 

EPS16 = 1e-16               # TT-PI SVD truncation tolerance

TTlambs   = np.linspace(1e-5, 5e-1, 10)   # coarse TT-MSTLS thresholds
flatlambs = np.linspace(1e-5, 1e-1, 10)   # fine matrix-MSTLS thresholds (TT side)
FLAT_LDS  = np.logspace(-3, 3, 12)        # STLS thresholds swept for flat WSINDy.

def l96_data(D, seed=0, burn=20.0):
    """Simulate Lorenz-96, landing on the attractor before sampling.
    """
    rhs = lambda x, t: (np.roll(x, -1) - np.roll(x, 2))*np.roll(x, 1) - C*x + F
    rng = np.random.default_rng(seed)
    x0 = (F/C)*np.ones(D) + 0.01*rng.standard_normal(D)
    x0 = odeint(rhs, x0, np.linspace(0, burn, 1000))[-1]     # burn-in to attractor
    X = odeint(rhs, x0, np.linspace(t0, tM, M)).T             # (D, M)
    if NOISE_LEVEL:
        X = X + NOISE_LEVEL*(np.linalg.norm(X, 'fro')/np.sqrt(X.size)) \
            * rng.standard_normal(X.shape)
    return X

def l96_true(D):
    """Per-equation true support as {tuple-of-powers over {1,x}: coefficient}."""
    true = []
    for d in range(D):
        terms = {tuple([0]*D): F}                              # constant  F
        t = [0]*D; t[d] = 1;                           terms[tuple(t)] = -C
        t = [0]*D; t[(d+1) % D] = 1; t[(d-1) % D] = 1; terms[tuple(t)] = 1.0
        t = [0]*D; t[(d-2) % D] = 1; t[(d-1) % D] = 1; terms[tuple(t)] = -1.0
        true.append(terms)
    return true

def support_error(supps, fmaps, D, true):
    """
    Aggregate support recovery over the D equations.
    """
    found = total = spur = 0
    for d in range(D):
        td = set(true[d].keys())
        total += len(td)
        if supps[d] is None:
            continue
        rec = {tuple(int(p) for p in fmaps[d][k]) for k in supps[d]}
        found += len(rec & td)
        spur += len(rec - td)
    return 100.0*found/total, spur/D

def coeff_error(W, supps, fmaps, D, true):
    """
    Mean over equations of the relative l2 coefficient error to the truth,
    taken over the union of the true and recovered terms. Missing true terms
    and spurious terms both contribute. Returned as a percentage.
    """
    errs = []
    for d in range(D):
        rec = {}
        if supps[d] is not None:
            for i, k in enumerate(supps[d]):
                rec[tuple(int(p) for p in fmaps[d][k])] = float(W[d][i])
        keys = set(true[d]) | set(rec)
        wt = np.array([true[d].get(k, 0.0) for k in keys])
        wh = np.array([rec.get(k, 0.0) for k in keys])
        denom = np.linalg.norm(wt) or 1.0
        errs.append(np.linalg.norm(wh - wt)/denom)
    return 100.0*np.mean(errs)

def run_flat(X, D):
    """
    Flat WSINDy on the full J^D tensor-product library.
    """
    st = time()
    phi, dphi = test_function.piecewise_polynomial((tM-t0)*R_FRAC, DEGREE, t0, tM, M, order=1)
    fmap = list(itertools.product(*[range(J)]*D))
    basis = np.stack([np.vectorize(fj)(X).astype(float) for fj in f])   # (J, D, M)
    G = np.ones((J**D, M))
    for k, tup in enumerate(fmap):
        for d in range(D):
            G[k] *= basis[tup[d], d]
    Gc = correlate(G, np.expand_dims(phi, 0), mode='valid').T           # (Mp, J^D)
    del G                          
    Y = -1*correlate(X, np.expand_dims(dphi, 0), mode='valid').T        # (Mp, D)

    M_diag = np.linalg.norm(Gc, 2, 0); M_diag[M_diag == 0] = 1.0
    Gc /= M_diag; Gn = Gc
    Gn_pinv = np.linalg.pinv(Gn)                                        # cache once, reuse

    supps, Ws = [], []
    for d in range(D):
        yb = Y[:, d]
        Gw0 = Gn @ (Gn_pinv @ yb)                                      
        Gw0_norm = np.linalg.norm(Gw0);  Gw0_norm = Gw0_norm or 1.0
        best_loss, best_s, best_w = np.inf, None, None
        for ld in FLAT_LDS:
            model = wsindy(ld=ld, gamma=0.0, scaled_theta=2)
            Xi = np.ndarray.flatten(model.sparsifyDynamics(Gn, yb, 1, pinv=Gn_pinv))
            s = np.where(Xi != 0)[0]
            if s.size == 0:
                continue
            diffnorm = np.linalg.norm(Gn[:, s] @ Xi[s] - Gw0)/Gw0_norm
            loss = diffnorm + s.size/(J**D)                           
            if loss <= best_loss:
                best_loss, best_s, best_w = loss, s, Xi[s]/M_diag[s]
        supps.append(best_s)
        Ws.append(best_w)
    return time()-st, supps, [fmap]*D, Ws

def run_tt(X, D, eps):
    """One-pass low-rank TT-WSINDy at TT-PI SVD tolerance `eps`."""
    r = TT_WSINDy(X, t0, tM, f, TTlambs, flatlambs,
                  testfn=('piecewise_polynomial', R_FRAC, DEGREE, 1),
                  threshold=eps, verbosity=0, low_rank=True, one_pass=True)
    # walltime, supp, feature_maps, coefficients, coarse_supps
    return r[3], r[1], r[2], r[0], r[6]

def tt_support_sizes(supp, coarse_supps, D):
    """Support sizes through the TT-WSINDy pipeline for one run.
    """
    initial = D * (J**D)
    coarse = sum(int(np.prod([max(len(s), 1) for s in coarse_supps[d]]))
                 for d in range(D))
    fine = sum(0 if supp[d] is None else len(supp[d]) for d in range(D))
    return initial, coarse, fine

def make_figures(rows, sizes_rows):
    """Draw the figures from already-computed rows.

    Kept separate from the sweep so the figures can be redrawn from the saved
    data files without recomputing anything.

    Parameters
    ----------
    rows : ndarray (n_D, 18)
        Main metrics, one row per D: [D, 8 means, 8 sds, M].
    sizes_rows : array-like (n_D, 7)
        Support-size funnel rows: [D, initial, coarse, fine, coarse_sd,
        fine_sd, M].
    """
    prows = rows
    Ds  = prows[:, 0]
    t16, tf = prows[:, 1], prows[:, 2]
    r16, rf = prows[:, 3], prows[:, 4]
    sp16, spf = prows[:, 5], prows[:, 6]
    ce16, cef = prows[:, 7], prows[:, 8]

    Dticks = np.unique(Ds.astype(int))      # D is an integer count

    def int_D_axis(*axes):
        """Label a D axis with only the integer D actually plotted."""
        for a in axes:
            a.set_xticks(Dticks)
            a.set_xticklabels([str(d) for d in Dticks])

    # --------------------------- Figure 1: walltime --------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    #ax.set_yscale('log')
    ax.plot(Ds, t16, 'o-', color='C0', label='TT-WSINDy')
    ax.plot(Ds, tf, '^-', color='C3', label='flat WSINDy')
    ax.set_xlabel('number of dimensions $D$')
    ax.set_ylabel('walltime (s)')
    int_D_axis(ax)
    ax.set_title(rf'Walltime vs $D$  (Lorenz 96, $\Delta t={DT}$, $M={M}$)')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{RESULTS}/walltime_vs_D{FILE_SUFFIX}.png', dpi=150)
    print(f"\nsaved {RESULTS}/walltime_vs_D{FILE_SUFFIX}.png")

    # ------- Figure 2: support recovery & coefficient error -------
    n_true = len(l96_true(int(Ds[0]))[0])              # true terms per equation
    jac_tt = (r16/100.0*n_true) / (n_true + sp16)      # TT   support Jaccard
    jac_fl = (rf/100.0*n_true) / (n_true + spf)        # flat support Jaccard

    fig, (axP, axC) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # (a) support recovery: Jaccard index / true positivity ratio
    axP.axhline(1.0, color='0.5', ls='--', lw=1.2, zorder=1)
    axP.plot(Ds, jac_fl, '^-', color='C3', lw=2, ms=8, label='flat WSINDy')
    axP.plot(Ds, jac_tt, 'o-', color='C0', lw=2, ms=8, label='TT-WSINDy')
    axP.set_ylim(-0.03, 1.07)
    #axP.set_yscale('log')
    axP.set_xlabel('number of dimensions $D$')
    axP.set_ylabel(r'support Jaccard  $|S\cap S^*|/|S\cup S^*|$  (TPR)')
    axP.set_title('(a) Support recovery (Jaccard / TPR)')
    axP.grid(True, ls=':', alpha=0.5)
    axP.legend(loc='upper right')

    # (b) coefficient error vs truth
    axC.plot(Ds, cef, '^-', color='C3', lw=2, ms=8, label='flat WSINDy')
    axC.plot(Ds, ce16, 'o-', color='C0', lw=2, ms=8, label='TT-WSINDy')
    #axC.set_yscale('log')
    axC.set_xlabel('number of dimensions $D$')
    axC.set_ylabel(r'mean relative $\ell_2$ coefficient error (%)')
    axC.set_title('(b) Coefficient error vs truth')
    axC.grid(True, which='both', ls=':', alpha=0.5)
    axC.legend(loc='center left')
    int_D_axis(axP, axC)

    fig.suptitle('TT-WSINDy vs flat WSINDy: support recovery and coefficient '
                 f'error  (Lorenz-96, {NUM_TRIALS} trials)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{RESULTS}/jaccard_coeff_vs_D{FILE_SUFFIX}.png', dpi=150)
    print(f"saved {RESULTS}/jaccard_coeff_vs_D{FILE_SUFFIX}.png")

    # ------- Figure 3: TT-WSINDy support reduction (initial -> coarse -> fine) -------
    srows = np.array(sizes_rows, float)
    s_init, s_coarse, s_fine = srows[:, 1], srows[:, 2], srows[:, 3]
    s_coarse_sd, s_fine_sd = srows[:, 4], srows[:, 5]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xpos = [0, 1, 2]
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(srows)))
    for i, c in enumerate(colors):
        ax.plot(xpos, [s_init[i], s_coarse[i], s_fine[i]],
                'o-', color=c, lw=2, label=f'$D={int(srows[i, 0])}$')
    ax.set_xticks(xpos)
    ax.set_xticklabels(['initial\n$D\\cdot 2^D$', 'after coarse\npass', 'after fine\npass'])
    ax.set_xlim(-0.15, 2.15)
    ax.set_ylabel('total support size')
    ax.set_title('TT-WSINDy support reduction (initial -> coarse -> fine)')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(title='dimension', ncol=2)
    plt.tight_layout()
    plt.savefig(f'{RESULTS}/support_reduction_vs_D{FILE_SUFFIX}.png', dpi=150)
    print(f"saved {RESULTS}/support_reduction_vs_D{FILE_SUFFIX}.png")

DATA = f'{RESULTS}/walltime_support_vs_D{FILE_SUFFIX}.txt'
SIZES = f'{RESULTS}/support_sizes_vs_D{FILE_SUFFIX}.txt'
METRICS = ["t_TT16", "t_flat",
           "recall_TT16", "recall_flat",
           "spur_TT16", "spur_flat",
           "cerr_TT16", "cerr_flat"]
HEADER = (f"mean(cols 1..8) then sd(cols 9..16) over {NUM_TRIALS} trials; "
          f"dt={DT}, {NOISE_LABEL}, M={M}\n"
          "D  " + " ".join(METRICS) + "  "
          + " ".join(m + "_sd" for m in METRICS) + "  M")
SIZES_HEADER = ("mean over %d trials (TT-WSINDy, %s); M=%d\n"
                "D initial coarse fine coarse_sd fine_sd M"
                % (NUM_TRIALS, NOISE_LABEL, M))

def run_scan():
    """Run the whole D_MIN..D_MAX sweep and write DATA and SIZES.
    """
    os.makedirs(RESULTS, exist_ok=True)

    Ds = list(range(D_MIN, D_MAX + 1))

    data_rows = []            # one row per D, in scan order
    sizes_rows = []

    for D in Ds:
        true = l96_true(D)

        print(f"\n=== D={D}  (J^D={J**D} candidates/eqn, M={M}, "
              f"t in [{t0:g},{tM:g}], dt={DT}, {NUM_TRIALS} trials) ===",
              flush=True)

        trials = []            # per-trial [12 metrics]
        size_trials = []       # per-trial [initial, coarse, fine]
        for trial in range(NUM_TRIALS):
            X = l96_data(D, seed=trial)   # distinct trajectory per trial

            t16, s16, fm16, W16, cs16 = run_tt(X, D, EPS16)
            r16, sp16 = support_error(s16, fm16, D, true)
            ce16 = coeff_error(W16, s16, fm16, D, true)

            tf, sf, fmf, Wf = run_flat(X, D)
            rf, spf = support_error(sf, fmf, D, true)
            cef = coeff_error(Wf, sf, fmf, D, true)

            trials.append([t16, tf, r16, rf, sp16, spf, ce16, cef])
            size_trials.append(tt_support_sizes(s16, cs16, D))
            print(f"  trial {trial+1}/{NUM_TRIALS}:  "
                  f"TT {t16:7.1f}s r={r16:5.1f}% sp={sp16:5.2f} ce={ce16:6.2f}% | "
                  f"flat {tf:7.1f}s r={rf:5.1f}% sp={spf:8.2f} ce={cef:6.2f}%",
                  flush=True)

        trials = np.array(trials, float)
        mean, sd = trials.mean(0), trials.std(0)
        print(f"  --> mean over {NUM_TRIALS}: "
              f"TT {mean[0]:7.1f}s r={mean[2]:5.1f}% sp={mean[4]:5.2f} ce={mean[6]:6.2f}% | "
              f"flat {mean[1]:7.1f}s r={mean[3]:5.1f}% sp={mean[5]:8.2f} ce={mean[7]:6.2f}%",
              flush=True)

        size_trials = np.array(size_trials, float)
        smean, ssd = size_trials.mean(0), size_trials.std(0)

        # record, then rewrite this run's rows so far, so a long scan is not
        # lost if it is interrupted
        data_rows.append(np.array([D, *mean, *sd, M], float))
        sizes_rows.append(np.array([D, smean[0], smean[1], smean[2],
                                    ssd[1], ssd[2], M], float))
        np.savetxt(DATA, np.array(data_rows, float),
                   header=HEADER, fmt="%.6g")
        np.savetxt(SIZES, np.array(sizes_rows, float),
                   header=SIZES_HEADER, fmt="%.6g")

def load_data(): return xu.load_txt(DATA), xu.load_txt(SIZES)

if __name__ == '__main__':

    recompute_data = True

    if recompute_data: run_scan()
    rows, sizes_rows = load_data()

    print(f'read {DATA}')
    have = [int(d) for d in rows[:, 0]]
    print(f'main data D = {have}')

    make_figures(rows, sizes_rows)
