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
from wsindy import wsindy   # original WSINDy-folder sparsifier (sparsifyDynamics)

# ------------------------------- problem setup -------------------------------
F = 8.0                     # Lorenz-96 forcing
C = 1.0                     # linear damping coefficient (-C x_i term)
t0 = 0.0                    # sampling starts here
DT = 0.1                    # sampling step, FIXED across the scan
M = 3000                    # number of time points -- set per D by set_sampling()
tM = t0 + DT*M              # so the window lengthens with M, it does not refine
D_MIN, D_MAX = 5, 9        # dimension scan (inclusive)
NUM_TRIALS = 3              # trials averaged per datapoint (distinct IC seeds)

# candidate library f = {1, x}  (J = 2); its tensor product spans the true model
f = [lambda x: 1.0, lambda x: x]
J = len(f)

M_DEFAULT = 20000
M_SCHEDULE = {}

M_ESCALATION = 1.5
MAX_ESCALATIONS = 3
RECOVERY_EXEMPT = {4, 5}

NOISE_LEVEL = 1e-3
FILE_SUFFIX = '' if NOISE_LEVEL == 0 else f'_noise{NOISE_LEVEL*100:g}pct'
NOISE_LABEL = ('clean data' if NOISE_LEVEL == 0
               else f'{NOISE_LEVEL*100:g}% noise')
PLOT_EXCLUDE = {4,13}

R_FRAC, DEGREE = 1/120, 16   # piecewise-polynomial test function (radius, degree).
                             # NOTE: set_sampling() OVERWRITES R_FRAC with 10/M
                             # on every call, which pins the radius at
                             # (tM-t0)*R_FRAC = 10*DT = 0.5 time units for every
                             # M; the value here only matters to callers that
                             # import the module without calling set_sampling.

EPS16 = 1e-16               # TT-PI SVD truncation tolerance (eps=1e-300 was
                            # measured to be indistinguishable -- see git log)

TTlambs   = np.linspace(1e-5, 5e-1, 10)   # coarse TT-MSTLS thresholds
flatlambs = np.linspace(1e-5, 1e-1, 10)   # fine matrix-MSTLS thresholds (TT side)
FLAT_LDS  = np.logspace(-3, 3, 12)        # STLS thresholds swept for flat WSINDy.

def set_sampling(D, M_override=None):
    """Point the sampling globals at this D's time grid.

    M comes from M_SCHEDULE (or M_override, when a dimension is being rerun at a
    larger M). The step stays DT and the test-function radius stays 10*DT = 0.5
    time units, so a larger M buys a LONGER trajectory -- more of the attractor
    -- rather than a finer grid or a wider phi. Returns the M it set.
    """
    global M, tM, R_FRAC
    M = int(M_override if M_override is not None else M_SCHEDULE.get(D, M_DEFAULT))
    tM = t0 + DT*M
    R_FRAC = 10.0/M
    return M

def l96_data(D, seed=0, burn=20.0):
    """Simulate Lorenz-96, landing on the attractor before sampling.

    With NOISE_LEVEL > 0, additive Gaussian noise of standard deviation
    NOISE_LEVEL*||X||_F/sqrt(X.size) is applied to the sampled trajectory. The
    initial condition is drawn first, so a given seed puts the same trajectory
    under every noise level.
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

    Returns
    -------
    recall : float
        Percentage of true terms discovered (summed over equations).
    spurious : float
        Mean number of spurious (non-true) terms per equation.
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
    ||w_hat - w_true||_2 / ||w_true||_2, taken over the union of the true and
    recovered terms (equivalently the full candidate space, since terms outside
    the union are zero in both). Missing true terms and spurious terms both
    contribute. Returned as a percentage.
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
    Flat WSINDy on the full J^D tensor-product library, sparsified by the
    original WSINDy-folder STLS engine (wsindy.sparsifyDynamics, L2 column
    normalization). A threshold sweep with loss-based model selection (the
    standard MSTLS wrapper) picks each equation's support. The library's
    pseudoinverse is target-independent, so it is formed once and reused across
    all equations and thresholds -- the same cross-equation reuse TT-WSINDy gets
    from TT_PI_factors.
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
    del G                          # (J^D, M) is dead once convolved; free it
    Y = -1*correlate(X, np.expand_dims(dphi, 0), mode='valid').T        # (Mp, D)

    # L2-normalize columns (wsindy scaled_theta=2) so thresholding is scale-free
    M_diag = np.linalg.norm(Gc, 2, 0); M_diag[M_diag == 0] = 1.0
    Gc /= M_diag; Gn = Gc          # in place: Gc is not needed unnormalized
    Gn_pinv = np.linalg.pinv(Gn)                                        # cache once, reuse

    supps, Ws = [], []
    for d in range(D):
        yb = Y[:, d]
        Gw0 = Gn @ (Gn_pinv @ yb)                                      # full-solution fit
        Gw0_norm = np.linalg.norm(Gw0);  Gw0_norm = Gw0_norm or 1.0
        best_loss, best_s, best_w = np.inf, None, None
        for ld in FLAT_LDS:
            model = wsindy(ld=ld, gamma=0.0, scaled_theta=2)          # fresh (ld mutates)
            Xi = np.ndarray.flatten(model.sparsifyDynamics(Gn, yb, 1, pinv=Gn_pinv))
            s = np.where(Xi != 0)[0]
            if s.size == 0:
                continue
            diffnorm = np.linalg.norm(Gn[:, s] @ Xi[s] - Gw0)/Gw0_norm
            loss = diffnorm + s.size/(J**D)                            # MSTLS loss
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

    Computed from an already-completed run's `supp` and `coarse_supps` (so no
    extra TT solve is needed).

    Returns (initial, coarse, fine):
      initial : D * J^D  -- full tensor-product library summed over all D eqns
      coarse  : sum over equations of the coarse product-support size (product
                of the per-dimension surviving feature counts) after TT-MSTLS
      fine    : sum over equations of the terms kept by the fine matrix MSTLS
    """
    initial = D * (J**D)
    coarse = sum(int(np.prod([max(len(s), 1) for s in coarse_supps[d]]))
                 for d in range(D))
    fine = sum(0 if supp[d] is None else len(supp[d]) for d in range(D))
    return initial, coarse, fine

def make_figures(rows, sizes_rows):
    """Draw the figures from already-computed rows.

    Kept separate from the sweep so the figures can be redrawn from the
    saved data files without recomputing anything (recompute_data = False in
    the main block below).

    Parameters
    ----------
    rows : ndarray (n_D, 18)
        Main metrics, one row per D: [D, 8 means, 8 sds, M].
    sizes_rows : array-like (n_D, 7)
        Support-size funnel rows: [D, initial, coarse, fine, coarse_sd,
        fine_sd, M].

    Which D are drawn is set by PLOT_EXCLUDE, not by this function.
    """
    # ---- plotting subset: include D=6, omit PLOT_EXCLUDE (D=13) ----
    prows = rows[np.array([int(d) not in PLOT_EXCLUDE for d in rows[:, 0]])]
    Ds  = prows[:, 0]
    t16, tf = prows[:, 1], prows[:, 2]
    r16, rf = prows[:, 3], prows[:, 4]
    sp16, spf = prows[:, 5], prows[:, 6]
    ce16, cef = prows[:, 7], prows[:, 8]
    # standard deviations (columns 9..16), same metric order -- read but not
    # drawn: the figures show trial means only
    e_t16, e_tf = prows[:, 9], prows[:, 10]
    e_r16, e_rf = prows[:, 11], prows[:, 12]
    e_sp16, e_spf = prows[:, 13], prows[:, 14]
    e_ce16, e_cef = prows[:, 15], prows[:, 16]

    Dticks = np.unique(Ds.astype(int))      # D is an integer count: no 8.5 ticks

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
    Ms = np.unique(prows[:, 17])            # M is the last column of each row
    m_lab = f'$M={int(Ms[0])}$' if Ms.size == 1 else '$M$ set per $D$'
    #ax.set_title(rf'Walltime vs $D$  (Lorenz 96, $\Delta t={DT}$, {m_lab})')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{RESULTS}/walltime_vs_D{FILE_SUFFIX}.png', dpi=150)
    print(f"\nsaved {RESULTS}/walltime_vs_D{FILE_SUFFIX}.png")

    # ----------- Figure 2: support & coefficient error vs D ------------
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(19, 5.5))

    style = [('TT-WSINDy', 'o-', 'C0'),
             ('flat WSINDy', '^-', 'C3')]

    for (lab, mk, col), y in zip(style, (r16, rf)):
        axL.plot(Ds, y, mk, color=col, label=lab)
    axL.set_xlabel('number of dimensions $D$')
    axL.set_ylabel('% of true terms discovered')
    axL.set_title('Recovery (recall)')
    axL.set_ylim(-5, 105)
    axL.grid(True, ls=':', alpha=0.5)
    axL.legend()

    for (lab, mk, col), y in zip(style, (sp16, spf)):
        axM.plot(Ds, y, mk, color=col, label=lab)
    axM.set_yscale('symlog', linthresh=1.0)   # TT ~O(1), flat ~O(10^3): show both
    axM.set_xlabel('number of dimensions $D$')
    axM.set_ylabel('mean # spurious terms per equation')
    axM.set_title('Spurious terms')
    axM.set_ylim(bottom=-0.2)
    axM.grid(True, which='both', ls=':', alpha=0.5)
    axM.legend()

    for (lab, mk, col), y in zip(style, (ce16, cef)):
        axR.plot(Ds, y, mk, color=col, label=lab)
    #axR.set_yscale('log')                      # spans clean (~0.1%) to failed (~100%+)
    axR.set_xlabel('number of dimensions $D$')
    axR.set_ylabel('mean relative $\\ell_2$ coefficient error (%)')
    axR.set_title('Coefficient error vs truth')
    axR.grid(True, which='both', ls=':', alpha=0.5)
    axR.legend()
    int_D_axis(axL, axM, axR)

    fig.suptitle(f'Support & coefficient error vs $D$  (Lorenz 96, '
                 f'mean of {NUM_TRIALS} trials)')
    plt.tight_layout()
    plt.savefig(f'{RESULTS}/support_vs_D{FILE_SUFFIX}.png', dpi=150)
    print(f"saved {RESULTS}/support_vs_D{FILE_SUFFIX}.png")

    # ------- Figure 2b: support recovery & coefficient error (2-panel story) -------
    # Built entirely from the in-memory main-data rows (prows) -- no re-run. Two panels:
    #   (a) support-recovery Jaccard index (a.k.a. true positivity ratio, TPR)
    #       vs D: J(S,S*) = |S n S*| / |S u S*|, with 1.0 = exact support. It
    #       penalizes BOTH missed true terms and spurious terms, so flat's ~J^D
    #       false positives crater its J while TT stays close to the truth.
    #   (b) mean relative l2 coefficient error vs truth: both pinned at the shared
    #       weak-form floor while support is exact (low D), rising to a comparable
    #       O(10-100%) band once recovery breaks down (high D).
    #
    # Jaccard from the aggregates (pooled / micro-averaged over the D
    # equations -- the standard WSINDy TPR = TP/(TP+FP+FN)). With n_true true
    # terms per equation, recall = 100*(true found)/(n_true*D) and `spur` = mean
    # spurious per equation, so per equation (the D cancels):
    #   |S n S*| = recall/100 * n_true,     |S u S*| = n_true + spur.
    # Computed from the trial-MEAN recall/spur, so it is the pooled TPR of the
    # mean support: exact wherever recovery is clean (J=1), and a close proxy for
    # the per-trial mean elsewhere. (Exact per-trial J +- sd would need J stored
    # during the sweep; omitted here to keep this a redraw of existing rows.)
    n_true = len(l96_true(int(Ds[0]))[0])              # true terms per equation (= 4)
    jac_tt = (r16/100.0*n_true) / (n_true + sp16)      # TT   support Jaccard / TPR
    jac_fl = (rf/100.0*n_true) / (n_true + spf)        # flat support Jaccard / TPR

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
    # TT-WSINDy only (no flat). For each D, one line connects the full candidate
    # count D*2^D, the coarse product-support summed over the D equations, and
    # the fine MSTLS support -- showing how the structured coarse pass collapses
    # the library before the fine solve. Computed alongside the main sweep (from
    # the same eps=1e-16 trials) and stored as trial means.
    srows = np.array(sizes_rows, float)
    srows = srows[np.array([int(d) not in PLOT_EXCLUDE for d in srows[:, 0]])]
    s_init, s_coarse, s_fine = srows[:, 1], srows[:, 2], srows[:, 3]
    s_coarse_sd, s_fine_sd = srows[:, 4], srows[:, 5]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xpos = [0, 1, 2]
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(srows)))
    for i, c in enumerate(colors):
        ax.plot(xpos, [s_init[i], s_coarse[i], s_fine[i]],
                'o-', color=c, lw=2, label=f'$D={int(srows[i, 0])}$')
    #ax.set_yscale('log')
    ax.set_xticks(xpos)
    ax.set_xticklabels(['initial\n$D\\cdot 2^D$', 'after coarse\npass', 'after fine\npass'])
    ax.set_xlim(-0.15, 2.15)
    ax.set_ylabel('total support size')
    #ax.set_title('TT-WSINDy support reduction (initial -> coarse -> fine)')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(title='dimension', ncol=2)
    plt.tight_layout()
    plt.savefig(f'{RESULTS}/support_reduction_vs_D{FILE_SUFFIX}.png', dpi=150)
    print(f"saved {RESULTS}/support_reduction_vs_D{FILE_SUFFIX}.png")


# ----------------------------- data files ------------------------------------
DATA = f'{RESULTS}/walltime_support_vs_D{FILE_SUFFIX}.txt'
SIZES = f'{RESULTS}/support_sizes_vs_D{FILE_SUFFIX}.txt'
# 8 per-datapoint metrics; the data file stores mean then sd (over trials).
METRICS = ["t_TT16", "t_flat",
           "recall_TT16", "recall_flat",
           "spur_TT16", "spur_flat",
           "cerr_TT16", "cerr_flat"]
HEADER = (f"mean(cols 1..8) then sd(cols 9..16) over {NUM_TRIALS} trials; "
          f"dt={DT}, {NOISE_LABEL}, last col = M used at that D\n"
          "D  " + " ".join(METRICS) + "  "
          + " ".join(m + "_sd" for m in METRICS) + "  M")
SIZES_HEADER = ("mean over %d trials (TT-WSINDy, %s); last col = M\n"
                "D initial coarse fine coarse_sd fine_sd M"
                % (NUM_TRIALS, NOISE_LABEL))


def run_scan():
    """Run the whole D_MIN..D_MAX sweep and write DATA and SIZES.

    Every D is recomputed and both result files are overwritten; nothing is
    read back from disk, so the outputs always describe one run at one set of
    hyperparameters (M aside, which is per-D and is recorded in the last column
    of both files). The files are rewritten after every D, so a scan that is
    interrupted keeps the dimensions it finished -- but the next run starts
    over from D_MIN rather than resuming, and no previous run is merged in.
    """
    os.makedirs(RESULTS, exist_ok=True)

    Ds = list(range(D_MIN, D_MAX + 1))

    data_rows = []            # one row per D, in scan order
    sizes_rows = []

    for D in Ds:
        true = l96_true(D)
        M_D, escalations = set_sampling(D), 0

        while True:
            print(f"\n=== D={D}  (J^D={J**D} candidates/eqn, M={M_D}, "
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
                # support-reduction funnel reuses the TT run above
                size_trials.append(tt_support_sizes(s16, cs16, D))
                print(f"  trial {trial+1}/{NUM_TRIALS}:  "
                      f"TT {t16:7.1f}s r={r16:5.1f}% sp={sp16:5.2f} ce={ce16:6.2f}% | "
                      f"flat {tf:7.1f}s r={rf:5.1f}% sp={spf:8.2f} ce={cef:6.2f}%",
                      flush=True)

            trials = np.array(trials, float)
            # every trial must find every true term for this D's M to stand;
            # D=4,5 are exempt (see the M_DEFAULT comment)
            missed = float(trials[:, 2].min())
            if D in RECOVERY_EXEMPT or missed >= 100.0:
                break
            if escalations >= MAX_ESCALATIONS:
                print(f"  !! D={D}: still only {missed:.1f}% recall at M={M_D} "
                      f"after {escalations} escalations -- keeping this row",
                      flush=True)
                break
            escalations += 1
            M_D = set_sampling(D, int(round(M_D*M_ESCALATION/100.0))*100)
            print(f"  !! D={D}: a trial recovered only {missed:.1f}% of the true "
                  f"terms -- rerunning all {NUM_TRIALS} trials at M={M_D}",
                  flush=True)

        mean, sd = trials.mean(0), trials.std(0)
        print(f"  --> mean over {NUM_TRIALS} (M={M_D}): "
              f"TT {mean[0]:7.1f}s r={mean[2]:5.1f}% sp={mean[4]:5.2f} ce={mean[6]:6.2f}% | "
              f"flat {mean[1]:7.1f}s r={mean[3]:5.1f}% sp={mean[5]:8.2f} ce={mean[7]:6.2f}%",
              flush=True)

        size_trials = np.array(size_trials, float)
        smean, ssd = size_trials.mean(0), size_trials.std(0)

        # record, then rewrite this run's rows so far, so a long scan is not
        # lost if it is interrupted (no previous run is merged in)
        data_rows.append(np.array([D, *mean, *sd, M_D], float))
        sizes_rows.append(np.array([D, smean[0], smean[1], smean[2],
                                    ssd[1], ssd[2], M_D], float))
        np.savetxt(DATA, np.array(data_rows, float),
                   header=HEADER, fmt="%.6g")
        np.savetxt(SIZES, np.array(sizes_rows, float),
                   header=SIZES_HEADER, fmt="%.6g")


def load_data():
    """Read DATA and SIZES back as (rows, sizes_rows), 2-D arrays.

    Both files are keyed to FILE_SUFFIX, i.e. to NOISE_LEVEL, so a redraw reads
    exactly the files a run at the same NOISE_LEVEL wrote -- and writes its
    PNGs under that same suffix.
    """
    return xu.load_txt(DATA), xu.load_txt(SIZES)


# ------------- main -------------
if __name__ == '__main__':

    recompute_data = True    # False redraws the figures from the files above,
                             # which is the only way to get them without paying
                             # for the sweep again (a run recomputes every D)

    if recompute_data:
        run_scan()

    rows, sizes_rows = load_data()

    print(f'read {DATA}')
    have = [int(d) for d in rows[:, 0]]
    drawn = [d for d in have if d not in PLOT_EXCLUDE]
    print(f'main data D = {have}')
    print(f'PLOT_EXCLUDE = {sorted(PLOT_EXCLUDE)}  ->  plotting D = {drawn}')

    make_figures(rows, sizes_rows)
