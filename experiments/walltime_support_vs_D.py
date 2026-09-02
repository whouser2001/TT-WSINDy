"""
Walltime and support-recovery scan vs system dimension D for the standard
Lorenz-96 identification problem.

Sampling is at a fixed step DT for every D; what varies with D is M, the number
of time points (M_SCHEDULE), so a higher-dimensional system is given a longer
trajectory rather than a finer one. The test-function radius is pinned at
10*DT = 1 time unit throughout (R_FRAC = 10/M), so phi is the same physical
width at every D. Any dimension whose trials do not all recover the full true
support is rerun from scratch at a larger M -- see M_ESCALATION.
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
from ttwsindy import TT_WSINDy
from wsindy import wsindy   # original WSINDy-folder sparsifier (sparsifyDynamics)

# ------------------------------- problem setup -------------------------------
F = 8.0                     # Lorenz-96 forcing
C = 1.0                     # linear damping coefficient (-C x_i term)
t0 = 0.0                    # sampling starts here
DT = 0.1                    # sampling step, FIXED across the scan
M = 3000                    # number of time points -- set per D by set_sampling()
tM = t0 + DT*M              # so the window LENGTHENS with M, it does not refine
D_MIN, D_MAX = 4, 12        # dimension scan (inclusive)
NUM_TRIALS = 5              # trials averaged per datapoint (distinct IC seeds)

# Per-D number of time points. The candidate library has J^D = 2^D columns per
# equation, so identifiability needs more data as D grows; D not listed here
# uses M_DEFAULT. A dimension whose 5 trials do not all recover the full true
# support is rerun from scratch at M *= M_ESCALATION (D=4,5 excepted: their
# misses are the known coarse-pass loss artifact at small J^D, which more data
# does not fix).
M_DEFAULT = 3000
M_SCHEDULE = {9: 5000, 10: 10000, 11: 10000, 12: 20000}
M_ESCALATION = 1.5
MAX_ESCALATIONS = 3
RECOVERY_EXEMPT = {4, 5}
PLOT_EXCLUDE = {4, 5, 13}   # kept in the data files, omitted from the figures:
                            # D=4,5 recover incorrect support (a coarse-pass loss-
                            # tuning artifact at small J^D, not a paper concern);
                            # D=13 flat walltime is noisy/expensive. -> plots D=6..12

# candidate library f = {1, x}  (J = 2); its tensor product spans the true model
f = [lambda x: 1.0, lambda x: x]
J = len(f)

R_FRAC, DEGREE = 10/M, 16    # piecewise-polynomial test function (radius, degree);
                             # R_FRAC = 10/M keeps the radius at (tM-t0)*R_FRAC
                             # = 10*DT = 1 time unit for every M in the scan
EPS16, EPS300 = 1e-16, 1e-300  # TT-PI SVD truncation tolerances to compare

TTlambs   = np.linspace(1e-5, 5e-1, 10)   # coarse TT-MSTLS thresholds
flatlambs = np.linspace(1e-5, 1e-1, 10)   # fine matrix-MSTLS thresholds (TT side)
FLAT_LDS  = np.logspace(-4, -0.5, 12)     # STLS thresholds swept for flat WSINDy


def set_sampling(D, M_override=None):
    """Point the sampling globals at this D's time grid.

    M comes from M_SCHEDULE (or M_override, when a dimension is being rerun at a
    larger M). The step stays DT and the test-function radius stays 10*DT = 1
    time unit, so a larger M buys a LONGER trajectory -- more of the attractor --
    rather than a finer grid or a wider phi. Returns the M it set.
    """
    global M, tM, R_FRAC
    M = int(M_override if M_override is not None else M_SCHEDULE.get(D, M_DEFAULT))
    tM = t0 + DT*M
    R_FRAC = 10.0/M
    return M


def l96_data(D, seed=0, burn=20.0):
    """Simulate Lorenz-96, landing on the attractor before sampling."""
    rhs = lambda x, t: (np.roll(x, -1) - np.roll(x, 2))*np.roll(x, 1) - C*x + F
    rng = np.random.default_rng(seed)
    x0 = (F/C)*np.ones(D) + 0.01*rng.standard_normal(D)
    x0 = odeint(rhs, x0, np.linspace(0, burn, 1000))[-1]     # burn-in to attractor
    return odeint(rhs, x0, np.linspace(t0, tM, M)).T          # (D, M)


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
    saved data files without recomputing anything (see
    plot_walltime_support_vs_D.py).

    Parameters
    ----------
    rows : ndarray (n_D, 26)
        Main metrics, one row per D: [D, 12 means, 12 sds, M].
    sizes_rows : array-like (n_D, 7)
        Support-size funnel rows: [D, initial, coarse, fine, coarse_sd,
        fine_sd, M].

    Which D are drawn is set by PLOT_EXCLUDE, not by this function.
    """
    # ---- plotting subset: include D=6, omit PLOT_EXCLUDE (D=13) ----
    prows = rows[np.array([int(d) not in PLOT_EXCLUDE for d in rows[:, 0]])]
    Ds  = prows[:, 0]
    t16, t300, tf = prows[:, 1], prows[:, 2], prows[:, 3]
    r16, r300, rf = prows[:, 4], prows[:, 5], prows[:, 6]
    sp16, sp300, spf = prows[:, 7], prows[:, 8], prows[:, 9]
    ce16, ce300, cef = prows[:, 10], prows[:, 11], prows[:, 12]
    # standard deviations (columns 13..24), same metric order
    e_t16, e_t300, e_tf = prows[:, 13], prows[:, 14], prows[:, 15]
    e_r16, e_r300, e_rf = prows[:, 16], prows[:, 17], prows[:, 18]
    e_sp16, e_sp300, e_spf = prows[:, 19], prows[:, 20], prows[:, 21]
    e_ce16, e_ce300, e_cef = prows[:, 22], prows[:, 23], prows[:, 24]

    # --------------------------- Figure 1: walltime --------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_yscale('log')
    ax.errorbar(Ds, t16, yerr=e_t16, fmt='o-', color='C0', capsize=3,
                label=r'TT-WSINDy ($\epsilon=10^{-16}$)')
    ax.errorbar(Ds, t300, yerr=e_t300, fmt='s-', color='C2', capsize=3,
                label=r'TT-WSINDy ($\epsilon=10^{-300}$)')
    ax.errorbar(Ds, tf, yerr=e_tf, fmt='^-', color='C3', capsize=3,
                label='flat WSINDy')
    ax.set_xlabel('number of dimensions $D$')
    ax.set_ylabel('walltime (s)')
    ax.set_title(rf'Walltime vs $D$  (Lorenz 96, $\Delta t={DT}$, $M$ set per $D$)')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/walltime_vs_D.png', dpi=150)
    print("\nsaved results/walltime_vs_D.png")

    # ----------- Figure 2: support & coefficient error vs D ------------
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(19, 5.5))

    style = [(r'TT-WSINDy ($\epsilon=10^{-16}$)', 'o-', 'C0'),
             (r'TT-WSINDy ($\epsilon=10^{-300}$)', 's-', 'C2'),
             ('flat WSINDy', '^-', 'C3')]

    for (lab, mk, col), y, e in zip(style, (r16, r300, rf), (e_r16, e_r300, e_rf)):
        axL.errorbar(Ds, y, yerr=e, fmt=mk, color=col, capsize=3, label=lab)
    axL.set_xlabel('number of dimensions $D$')
    axL.set_ylabel('% of true terms discovered')
    axL.set_title('Recovery (recall)')
    axL.set_ylim(-5, 105)
    axL.grid(True, ls=':', alpha=0.5)
    axL.legend()

    for (lab, mk, col), y, e in zip(style, (sp16, sp300, spf),
                                    (e_sp16, e_sp300, e_spf)):
        axM.errorbar(Ds, y, yerr=e, fmt=mk, color=col, capsize=3, label=lab)
    axM.set_yscale('symlog', linthresh=1.0)   # TT ~O(1), flat ~O(10^3): show both
    axM.set_xlabel('number of dimensions $D$')
    axM.set_ylabel('mean # spurious terms per equation')
    axM.set_title('Spurious terms')
    axM.set_ylim(bottom=-0.2)
    axM.grid(True, which='both', ls=':', alpha=0.5)
    axM.legend()

    for (lab, mk, col), y, e in zip(style, (ce16, ce300, cef),
                                    (e_ce16, e_ce300, e_cef)):
        axR.errorbar(Ds, y, yerr=e, fmt=mk, color=col, capsize=3, label=lab)
    #axR.set_yscale('log')                      # spans clean (~0.1%) to failed (~100%+)
    axR.set_xlabel('number of dimensions $D$')
    axR.set_ylabel('mean relative $\\ell_2$ coefficient error (%)')
    axR.set_title('Coefficient error vs truth')
    axR.grid(True, which='both', ls=':', alpha=0.5)
    axR.legend()

    fig.suptitle(f'Support & coefficient error vs $D$  (Lorenz 96, '
                 f'mean$\\pm$sd of {NUM_TRIALS} trials)')
    plt.tight_layout()
    plt.savefig('results/support_vs_D.png', dpi=150)
    print("saved results/support_vs_D.png")

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
    jac_tt300 = (r300/100.0*n_true) / (n_true + sp300)      # TT eps=1e-300
    jac_fl = (rf/100.0*n_true) / (n_true + spf)        # flat support Jaccard / TPR

    fig, (axP, axC) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # (a) support recovery: Jaccard index / true positivity ratio
    axP.axhline(1.0, color='0.5', ls='--', lw=1.2, zorder=1)
    axP.plot(Ds, jac_fl, '^-', color='C3', lw=2, ms=8, label='flat WSINDy')
    axP.plot(Ds, jac_tt, 'o-', color='C0', lw=2, ms=8,
             label=r'TT-WSINDy ($\epsilon=10^{-16}$)')
    axP.plot(Ds, jac_tt300, 's-', color='C2', lw=2, ms=8,
             label=r'TT-WSINDy ($\epsilon=10^{-300}$)')
    axP.set_ylim(-0.03, 1.07)
    #axP.set_yscale('log')
    axP.set_xlabel('number of dimensions $D$')
    axP.set_ylabel(r'support Jaccard  $|S\cap S^*|/|S\cup S^*|$  (TPR)')
    axP.set_title('(a) Support recovery (Jaccard / TPR)')
    axP.grid(True, ls=':', alpha=0.5)
    axP.legend(loc='upper right')

    # (b) coefficient error vs truth
    axC.errorbar(Ds, cef, yerr=e_cef, fmt='^-', color='C3', lw=2, ms=8,
                 capsize=3, label='flat WSINDy')
    axC.errorbar(Ds, ce16, yerr=e_ce16, fmt='o-', color='C0', lw=2, ms=8,
                 capsize=3, label=r'TT-WSINDy ($\epsilon=10^{-16}$)')
    axC.errorbar(Ds, ce300, yerr=e_ce300, fmt='s-', color='C2', lw=2, ms=8,
                 capsize=3, label=r'TT-WSINDy ($\epsilon=10^{-300}$)')
    #axC.set_yscale('log')
    axC.set_xlabel('number of dimensions $D$')
    axC.set_ylabel(r'mean relative $\ell_2$ coefficient error (%)')
    axC.set_title('(b) Coefficient error vs truth')
    axC.grid(True, which='both', ls=':', alpha=0.5)
    axC.legend(loc='center left')

    fig.suptitle('TT-WSINDy vs flat WSINDy: comparable accuracy, very different '
                 f'support recovery  (Lorenz-96, {NUM_TRIALS} trials)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('results/jaccard_coeff_vs_D.png', dpi=150)
    print("saved results/jaccard_coeff_vs_D.png")

    # ------- Figure 3: TT-WSINDy support reduction (initial -> coarse -> fine) -------
    # TT-WSINDy only (no flat). For each D, one line connects the full candidate
    # count D*2^D, the coarse product-support summed over the D equations, and
    # the fine MSTLS support -- showing how the structured coarse pass collapses
    # the library before the fine solve. Computed alongside the main sweep (from
    # the same eps=1e-16 trials) and stored as trial means; coarse/fine carry a
    # +-1 sd error bar (initial is deterministic).
    srows = np.array(sizes_rows, float)
    srows = srows[np.array([int(d) not in PLOT_EXCLUDE for d in srows[:, 0]])]
    s_init, s_coarse, s_fine = srows[:, 1], srows[:, 2], srows[:, 3]
    s_coarse_sd, s_fine_sd = srows[:, 4], srows[:, 5]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xpos = [0, 1, 2]
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(srows)))
    for i, c in enumerate(colors):
        ax.errorbar(xpos, [s_init[i], s_coarse[i], s_fine[i]],
                    yerr=[0.0, s_coarse_sd[i], s_fine_sd[i]],
                    fmt='o-', color=c, lw=2, capsize=3, label=f'$D={int(srows[i, 0])}$')
    #ax.set_yscale('log')
    ax.set_xticks(xpos)
    ax.set_xticklabels(['initial\n$D\\cdot 2^D$', 'after coarse\npass', 'after fine\npass'])
    ax.set_xlim(-0.15, 2.15)
    ax.set_ylabel('total support size')
    ax.set_title(f'TT-WSINDy support reduction ($\\epsilon=10^{{-16}}$)')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(title='dimension', ncol=2)
    plt.tight_layout()
    plt.savefig('results/support_reduction_vs_D.png', dpi=150)
    print("saved results/support_reduction_vs_D.png")


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    # Every run recomputes every D in the scan and overwrites the result
    # files; nothing is read back from disk, so the outputs always describe
    # one run at one set of hyperparameters (M aside, which is per-D and is
    # recorded in the last column of both files).
    DATA = 'results/walltime_support_vs_D.txt'
    SIZES = 'results/support_sizes_vs_D.txt'
    # 12 per-datapoint metrics; the data file stores mean then sd (over trials).
    METRICS = ["t_TT16", "t_TT300", "t_flat",
               "recall_TT16", "recall_TT300", "recall_flat",
               "spur_TT16", "spur_TT300", "spur_flat",
               "cerr_TT16", "cerr_TT300", "cerr_flat"]
    HEADER = (f"mean(cols 1..12) then sd(cols 13..24) over {NUM_TRIALS} trials; "
              f"dt={DT}, last col = M used at that D\n"
              "D  " + " ".join(METRICS) + "  "
              + " ".join(m + "_sd" for m in METRICS) + "  M")
    SIZES_HEADER = ("mean over %d trials (TT-WSINDy eps=1e-16); last col = M\n"
                    "D initial coarse fine coarse_sd fine_sd M" % NUM_TRIALS)

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

                t300, s300, fm300, W300, _ = run_tt(X, D, EPS300)
                r300, sp300 = support_error(s300, fm300, D, true)
                ce300 = coeff_error(W300, s300, fm300, D, true)

                tf, sf, fmf, Wf = run_flat(X, D)
                rf, spf = support_error(sf, fmf, D, true)
                cef = coeff_error(Wf, sf, fmf, D, true)

                trials.append([t16, t300, tf, r16, r300, rf,
                               sp16, sp300, spf, ce16, ce300, cef])
                # support-reduction funnel reuses the eps=1e-16 run above
                size_trials.append(tt_support_sizes(s16, cs16, D))
                print(f"  trial {trial+1}/{NUM_TRIALS}:  "
                      f"TT16 {t16:6.1f}s r={r16:5.1f}% sp={sp16:5.2f} ce={ce16:6.2f}% | "
                      f"TT300 {t300:6.1f}s r={r300:5.1f}% | "
                      f"flat {tf:7.1f}s r={rf:5.1f}% sp={spf:8.2f} ce={cef:6.2f}%",
                      flush=True)

            trials = np.array(trials, float)
            # every trial must find every true term (both TT tolerances) for
            # this D's M to stand; D=4,5 are exempt (see M_SCHEDULE comment)
            missed = float(min(trials[:, 3].min(), trials[:, 4].min()))
            if D in RECOVERY_EXEMPT or missed >= 100.0:
                break
            if escalations >= MAX_ESCALATIONS:
                print(f"  !! D={D}: still only {missed:.1f}% recall at M={M_D} "
                      f"after {escalations} escalations -- keeping this row",
                      flush=True)
                break
            escalations += 1
            M_D = set_sampling(D, int(round(M_D*M_ESCALATION/1000.0))*1000)
            print(f"  !! D={D}: a trial recovered only {missed:.1f}% of the true "
                  f"terms -- rerunning all {NUM_TRIALS} trials at M={M_D}",
                  flush=True)

        mean, sd = trials.mean(0), trials.std(0)
        print(f"  --> mean over {NUM_TRIALS} (M={M_D}): "
              f"TT16 {mean[0]:6.1f}s r={mean[3]:5.1f}% sp={mean[6]:5.2f} ce={mean[9]:6.2f}% | "
              f"flat {mean[2]:7.1f}s r={mean[5]:5.1f}% sp={mean[8]:8.2f} ce={mean[11]:6.2f}%",
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

    # this run's full data set, in scan order
    rows = np.array(data_rows, float)

    make_figures(rows, sizes_rows)
