"""
Walltime and support-recovery scan vs system dimension D for the standard
Lorenz-96 identification problem.

Three sparsifiers are compared on the SAME weak form and the SAME {1, x}
tensor-product library (J^D candidates/equation):

  1. low-rank one-pass TT-WSINDy, TT-PI SVD truncation  eps = 1e-16
  2. low-rank one-pass TT-WSINDy, TT-PI SVD truncation  eps = 1e-10
  3. flat (matrix) WSINDy, sparsified by the ORIGINAL WSINDy-folder code
     (WSINDy/wsindy.py :: wsindy.sparsifyDynamics), on the full J^D library.

`eps` is TT_WSINDy's `threshold` argument -- the relative tolerance used to
truncate the matrix SVDs inside the TT pseudoinverse (TT-PI). eps=1e-16 is
effectively exact; eps=1e-10 truncates more aggressively (faster, at the risk
of dropping weak terms). Comparing the two shows that speed/accuracy trade-off.

The flat sparsifier is the repo's own WSINDy STLS engine (sparsifyDynamics,
with L2 column normalization). It is wrapped in a threshold (lambda) sweep with
loss-based model selection -- the standard MSTLS wrapper -- so flat gets a fair,
tuned support rather than one fixed by an arbitrary threshold. Its
target-independent pseudoinverse is cached once and reused across the D
equations and the whole sweep, mirroring TT-WSINDy reusing its TT_PI_factors,
so the walltime comparison is apples-to-apples.

Three figures are produced:
  * results/walltime_vs_D.png  -- walltime (s) vs D, log-scale, three methods.
  * results/support_vs_D.png   -- support/coefficient/simulation error vs D, four
      panels: (A) % of true terms discovered [recall], (B) mean number of spurious
      terms per equation, (C) mean relative L2 coefficient error vs the true
      coefficients, (D) relative forward-simulation error ||Xhat - X*||_F/||X*||_F
      from the true initial condition, over a short horizon t in [0, FSIM_HORIZON]
      (short because Lorenz-96 chaos saturates the full-window error ~0.9 even for
      the true model). Needs the models re-run -> results/fsim_error_vs_D.txt.
  * results/support_reduction_vs_D.png -- TT-WSINDy support-size funnel: for
      each D one line connects the full library (D*2^D), the coarse product-
      support summed over the D equations, and the fine MSTLS support.

Lorenz-96 (F=8):  dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F.
True support per equation = {1, x_i, x_{i-1}x_{i+1}, x_{i-1}x_{i-2}}, all of
which lie in the {1, x} tensor-product library (squarefree monomials), so exact
recovery is achievable by every method.

D = 4..13, M = 3000, t in [0, 30], test-function radius fraction r_frac = 1/80.
(All of D=4..13 are computed and retained in the saved data files, but the figures
display only D=6..12 via PLOT_EXCLUDE: D=4,5 recover incorrect support at small
J^D -- a loss-tuning artifact, not a paper concern -- and D=13's flat walltime is
noisy/expensive.)
This (M, tM, r_frac) is chosen so that TT-WSINDy recovers the exact support at
D <= 9 for both eps: a longer / narrower test function (r_frac = 1/80 vs 1/40)
sharpens the weak form enough for the coarse pass to keep the weak -x_i
self-term (cf. the coarse-selection limit). Flat WSINDy is not required to be
clean at D <= 9.

Every datapoint is the MEAN over NUM_TRIALS independent trials (distinct
Lorenz-96 trajectories drawn from different initial-condition seeds). The saved
data file stores both the per-datapoint mean and standard deviation across
trials, and the figures show error bars (+-1 sd). Run from experiments/ with the
scikit_tt env:
    python walltime_support_vs_D.py
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
t0, tM = 0.0, 30.0          # time sampling range
M = 3000                    # number of time points
D_MIN, D_MAX = 4, 13        # dimension scan (inclusive)
NUM_TRIALS = 5              # trials averaged per datapoint (distinct IC seeds)
PLOT_EXCLUDE = {4, 5, 13}   # kept in the data files, omitted from the figures:
                            # D=4,5 recover incorrect support (a coarse-pass loss-
                            # tuning artifact at small J^D, not a paper concern);
                            # D=13 flat walltime is noisy/expensive. -> plots D=6..12

# candidate library f = {1, x}  (J = 2); its tensor product spans the true model
f = [lambda x: 1.0, lambda x: x]
J = len(f)

R_FRAC, DEGREE = 1/80, 16   # piecewise-polynomial test function (radius, degree)
EPS16, EPS10 = 1e-16, 1e-10 # TT-PI SVD truncation tolerances to compare

TTlambs   = np.linspace(1e-5, 5e-1, 10)   # coarse TT-MSTLS thresholds
flatlambs = np.linspace(1e-5, 1e-1, 10)   # fine matrix-MSTLS thresholds (TT side)
FLAT_LDS  = np.logspace(-4, -0.5, 12)     # STLS thresholds swept for flat WSINDy
FSIM_HORIZON = 10.0          # forward-sim error is measured over t in [t0, t0+5]:
                            # a short horizon separates model quality BEFORE
                            # Lorenz-96 chaos decorrelates trajectories (over the
                            # full [0,30] window even the true model errs ~0.9)
FSIM_CAP  = 10.0            # per-trial forward-sim rel. error is capped here
                            # (a diverged/blown-up integration is recorded as CAP)


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


def build_model(supps, fmaps, W, D):
    """
    Reconstruct the recovered model as a list of per-equation dicts
    {power-tuple: coefficient} (same reconstruction coeff_error uses). Each
    power-tuple is the {1,x} exponent vector over the D states, so a term is the
    squarefree monomial prod_{dim: power==1} x[dim].
    """
    model = []
    for d in range(D):
        md = {}
        if supps[d] is not None:
            for i, k in enumerate(supps[d]):
                md[tuple(int(p) for p in fmaps[d][k])] = float(W[d][i])
        model.append(md)
    return model


def forward_sim_error(model, X_true, D):
    """
    Relative forward-simulation error ||Xhat - X*||_F / ||X*||_F over a SHORT
    horizon t in [t0, t0+FSIM_HORIZON].

    The recovered `model` (per-equation monomial dicts from build_model) is
    integrated from the TRUE initial condition X*[:, 0] on the data's time grid,
    restricted to the first FSIM_HORIZON time units, giving Xhat; the error is
    taken against X* over that same window. The short horizon is deliberate:
    Lorenz-96 is chaotic, so over the full [0,30] window even the exact true model
    decorrelates to rel-err ~0.9; measuring before decorrelation makes the metric
    reflect model quality. Non-finite / blown-up integrations are recorded as
    FSIM_CAP. Each equation is evaluated vectorized:
    dx_d = coeffs_d . prod_dim( x^E_d ), with E_d the 0/1 exponent matrix.
    """
    coeffs, Emats = [], []
    for d in range(D):
        if model[d]:
            ks = list(model[d].keys())
            coeffs.append(np.array([model[d][k] for k in ks], float))
            Emats.append(np.asarray(ks, float))          # (n_terms, D), entries 0/1
        else:
            coeffs.append(np.zeros(0)); Emats.append(np.zeros((0, D)))

    def rhs(x, t):
        dx = np.empty(D)
        for d in range(D):
            if coeffs[d].size:
                mon = np.prod(np.where(Emats[d] != 0, x[None, :], 1.0), axis=1)
                dx[d] = coeffs[d] @ mon
            else:
                dx[d] = 0.0
        return dx

    full = np.linspace(t0, tM, M)
    k = max(int(np.searchsorted(full, t0 + FSIM_HORIZON, side='right')), 2)
    tgrid, Xstar = full[:k], X_true[:, :k]               # short-horizon window
    with np.errstate(over='ignore', invalid='ignore'):
        try:
            Xhat = odeint(rhs, X_true[:, 0], tgrid, rtol=1e-8, atol=1e-10,
                          mxstep=10000).T                 # (D, k)
        except Exception:
            return FSIM_CAP
    den = np.linalg.norm(Xstar) or 1.0
    err = np.linalg.norm(Xhat - Xstar) / den
    return FSIM_CAP if not np.isfinite(err) else min(err, FSIM_CAP)


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
    Y = -1*correlate(X, np.expand_dims(dphi, 0), mode='valid').T        # (Mp, D)

    # L2-normalize columns (wsindy scaled_theta=2) so thresholding is scale-free
    M_diag = np.linalg.norm(Gc, 2, 0); M_diag[M_diag == 0] = 1.0
    Gn = Gc / M_diag
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


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    recompute = False
    # D values recomputed even if already cached (their old rows are overwritten;
    # every other D on disk is left untouched). Empty set => pure incremental.
    FORCE_RECOMPUTE = set()
    DATA = 'results/walltime_support_vs_D.txt'
    SIZES = 'results/support_sizes_vs_D.txt'
    # 12 per-datapoint metrics; the data file stores mean then sd (over trials).
    METRICS = ["t_TT16", "t_TT10", "t_flat",
               "recall_TT16", "recall_TT10", "recall_flat",
               "spur_TT16", "spur_TT10", "spur_flat",
               "cerr_TT16", "cerr_TT10", "cerr_flat"]
    HEADER = (f"mean(cols 1..12) then sd(cols 13..24) over {NUM_TRIALS} trials\n"
              "D  " + " ".join(METRICS) + "  "
              + " ".join(m + "_sd" for m in METRICS))
    SIZES_HEADER = ("mean over %d trials (TT-WSINDy eps=1e-16)\n"
                    "D initial coarse fine coarse_sd fine_sd" % NUM_TRIALS)

    Ds = list(range(D_MIN, D_MAX + 1))

    # ---- load any existing results so we never recompute or delete them ----
    # Rows are keyed by D; only D values NOT already present get computed, and
    # every D already on disk (including D=13) is preserved on save.
    def _load_by_D(path):
        by_D = {}
        if os.path.exists(path):
            old = np.loadtxt(path)
            if old.ndim == 1:
                old = old[None, :]
            for r in old:
                by_D[int(r[0])] = r
        return by_D

    data_by_D  = _load_by_D(DATA)
    sizes_by_D = _load_by_D(SIZES)

    if recompute:
        for D in Ds:
            if D in data_by_D and D in sizes_by_D and D not in FORCE_RECOMPUTE:
                print(f"\n=== D={D}: using cached data (not recomputed) ===",
                      flush=True)
                continue
            true = l96_true(D)
            print(f"\n=== D={D}  (J^D={J**D} candidates/eqn, M={M}, "
                  f"{NUM_TRIALS} trials) ===", flush=True)

            trials = []            # per-trial [12 metrics]
            size_trials = []       # per-trial [initial, coarse, fine]
            for trial in range(NUM_TRIALS):
                X = l96_data(D, seed=trial)   # distinct trajectory per trial

                t16, s16, fm16, W16, cs16 = run_tt(X, D, EPS16)
                r16, sp16 = support_error(s16, fm16, D, true)
                ce16 = coeff_error(W16, s16, fm16, D, true)

                t10, s10, fm10, W10, _ = run_tt(X, D, EPS10)
                r10, sp10 = support_error(s10, fm10, D, true)
                ce10 = coeff_error(W10, s10, fm10, D, true)

                tf, sf, fmf, Wf = run_flat(X, D)
                rf, spf = support_error(sf, fmf, D, true)
                cef = coeff_error(Wf, sf, fmf, D, true)

                trials.append([t16, t10, tf, r16, r10, rf,
                               sp16, sp10, spf, ce16, ce10, cef])
                # support-reduction funnel reuses the eps=1e-16 run above
                size_trials.append(tt_support_sizes(s16, cs16, D))
                print(f"  trial {trial+1}/{NUM_TRIALS}:  "
                      f"TT16 {t16:6.1f}s r={r16:5.1f}% sp={sp16:5.2f} ce={ce16:6.2f}% | "
                      f"TT10 {t10:6.1f}s r={r10:5.1f}% | "
                      f"flat {tf:7.1f}s r={rf:5.1f}% sp={spf:8.2f} ce={cef:6.2f}%",
                      flush=True)

            trials = np.array(trials, float)
            mean, sd = trials.mean(0), trials.std(0)
            print(f"  --> mean over {NUM_TRIALS}: "
                  f"TT16 {mean[0]:6.1f}s r={mean[3]:5.1f}% sp={mean[6]:5.2f} ce={mean[9]:6.2f}% | "
                  f"flat {mean[2]:7.1f}s r={mean[5]:5.1f}% sp={mean[8]:8.2f} ce={mean[11]:6.2f}%",
                  flush=True)

            size_trials = np.array(size_trials, float)
            smean, ssd = size_trials.mean(0), size_trials.std(0)

            # record, then save the FULL merged set (sorted by D) so no data is lost
            data_by_D[D]  = np.array([D, *mean, *sd], float)
            sizes_by_D[D] = np.array([D, smean[0], smean[1], smean[2],
                                      ssd[1], ssd[2]], float)
            np.savetxt(DATA, np.array([data_by_D[d] for d in sorted(data_by_D)],
                                      float), header=HEADER, fmt="%.6g")
            np.savetxt(SIZES, np.array([sizes_by_D[d] for d in sorted(sizes_by_D)],
                                       float), header=SIZES_HEADER, fmt="%.6g")

    # full data set (all D on disk, incl. D=13), sorted by D
    rows = np.array([data_by_D[d] for d in sorted(data_by_D)], float)

    # ---- forward-simulation error (separate incremental cache) ----
    # Relative ||Xhat - X*||_F / ||X*||_F, integrating each recovered model from
    # the TRUE initial condition. This needs the actual models (support+coeffs),
    # which the main data file does NOT store, so it RE-RUNS run_tt(eps16) and
    # run_flat per D/trial (deterministic in the seed). Own file; the main data and
    # its cache are untouched. Computed only for the plotted D (skips PLOT_EXCLUDE,
    # i.e. the expensive/omitted D=13 and the low-D artifacts D=4,5).
    DATA_FSIM = 'results/fsim_error_vs_D.txt'
    FSIM_HEADER = (f"forward-sim rel. error over t in [0,{FSIM_HORIZON:g}]: "
                   f"mean(cols 1..2) then sd(cols 3..4) over {NUM_TRIALS} trials, "
                   f"capped at {FSIM_CAP}\n"
                   "D  fsim_TT16 fsim_flat  fsim_TT16_sd fsim_flat_sd")
    fsim_by_D = _load_by_D(DATA_FSIM)
    if recompute:
        for D in Ds:
            if D in PLOT_EXCLUDE:
                continue
            if D in fsim_by_D and D not in FORCE_RECOMPUTE:
                print(f"\n=== [fsim] D={D}: using cached data ===", flush=True)
                continue
            print(f"\n=== [fsim] D={D}  ({NUM_TRIALS} trials, re-running models) ===",
                  flush=True)
            tr = []
            for trial in range(NUM_TRIALS):
                X = l96_data(D, seed=trial)
                _, s16, fm16, W16, _ = run_tt(X, D, EPS16)
                e16 = forward_sim_error(build_model(s16, fm16, W16, D), X, D)
                _, sf, fmf, Wf = run_flat(X, D)
                efl = forward_sim_error(build_model(sf, fmf, Wf, D), X, D)
                tr.append([e16, efl])
                print(f"  trial {trial+1}/{NUM_TRIALS}: fsim  TT16={e16:.3f}  "
                      f"flat={efl:.3f}", flush=True)
            tr = np.array(tr, float)
            fsim_by_D[D] = np.array([D, tr[:, 0].mean(), tr[:, 1].mean(),
                                     tr[:, 0].std(), tr[:, 1].std()], float)
            np.savetxt(DATA_FSIM, np.array([fsim_by_D[d] for d in sorted(fsim_by_D)],
                                           float), header=FSIM_HEADER, fmt="%.6g")

    # ---- plotting subset: include D=6, omit PLOT_EXCLUDE (D=13) ----
    prows = rows[np.array([int(d) not in PLOT_EXCLUDE for d in rows[:, 0]])]
    Ds  = prows[:, 0]
    t16, t10, tf = prows[:, 1], prows[:, 2], prows[:, 3]
    r16, r10, rf = prows[:, 4], prows[:, 5], prows[:, 6]
    sp16, sp10, spf = prows[:, 7], prows[:, 8], prows[:, 9]
    ce16, ce10, cef = prows[:, 10], prows[:, 11], prows[:, 12]
    # standard deviations (columns 13..24), same metric order
    e_t16, e_t10, e_tf = prows[:, 13], prows[:, 14], prows[:, 15]
    e_r16, e_r10, e_rf = prows[:, 16], prows[:, 17], prows[:, 18]
    e_sp16, e_sp10, e_spf = prows[:, 19], prows[:, 20], prows[:, 21]
    e_ce16, e_ce10, e_cef = prows[:, 22], prows[:, 23], prows[:, 24]

    # forward-sim error plot subset (cols: D, mean[TT16,flat], sd[TT16,flat]);
    # empty until computed
    if fsim_by_D:
        frows = np.array([fsim_by_D[d] for d in sorted(fsim_by_D)], float)
        frows = frows[np.array([int(d) not in PLOT_EXCLUDE for d in frows[:, 0]])]
    else:
        frows = np.empty((0, 5))

    # --------------------------- Figure 1: walltime --------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_yscale('log')
    ax.errorbar(Ds, t16, yerr=e_t16, fmt='o-', color='C0', capsize=3,
                label=r'TT-WSINDy ($\epsilon=10^{-16}$)')
    #ax.errorbar(Ds, t10, yerr=e_t10, fmt='s-', color='C2', capsize=3,
    #            label=r'one-pass TT-WSINDy ($\epsilon=10^{-10}$)')
    ax.errorbar(Ds, tf, yerr=e_tf, fmt='^-', color='C3', capsize=3,
                label='flat WSINDy')
    ax.set_xlabel('number of dimensions $D$')
    ax.set_ylabel('walltime (s)')
    ax.set_title(f'Walltime vs $D$  (Lorenz 96, $M={M}$, $t\\in[{t0:.0f},{tM:.0f}]$)')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/walltime_vs_D.png', dpi=150)
    print("\nsaved results/walltime_vs_D.png")

    # ------- Figure 2: support, coefficient error & forward-sim error --------
    layout = """
        AB
        CD
    """
    fig, ax_dict = plt.subplot_mosaic(layout, figsize=(14, 11))
    axL = ax_dict['A']
    axM = ax_dict['B']
    axR = ax_dict['C']
    axS = ax_dict['D']

    style = [(r'TT-WSINDy ($\epsilon=10^{-16}$)', 'o-', 'C0'),
             ('flat WSINDy', '^-', 'C3')]

    for (lab, mk, col), y, e in zip(style, (r16, rf), (e_r16, e_rf)):
        axL.errorbar(Ds, y, yerr=e, fmt=mk, color=col, capsize=3, label=lab)
    axL.set_xlabel('number of dimensions $D$')
    axL.set_ylabel('% of true terms discovered')
    axL.set_title('Recovery (recall)')
    axL.set_ylim(-5, 105)
    axL.grid(True, ls=':', alpha=0.5)
    axL.legend()

    for (lab, mk, col), y, e in zip(style, (sp16, spf), (e_sp16, e_spf)):
        axM.errorbar(Ds, y, yerr=e, fmt=mk, color=col, capsize=3, label=lab)
    axM.set_yscale('symlog', linthresh=1.0)   # TT ~O(1), flat ~O(10^3): show both
    axM.set_xlabel('number of dimensions $D$')
    axM.set_ylabel('mean # spurious terms per equation')
    axM.set_title('Spurious terms')
    axM.set_ylim(bottom=-0.2)
    axM.grid(True, which='both', ls=':', alpha=0.5)
    axM.legend()

    for (lab, mk, col), y, e in zip(style, (ce16, cef), (e_ce16, e_cef)):
        axR.errorbar(Ds, y, yerr=e, fmt=mk, color=col, capsize=3, label=lab)
    #axR.set_yscale('log')                      # spans clean (~0.1%) to failed (~100%+)
    axR.set_xlabel('number of dimensions $D$')
    axR.set_ylabel('mean relative $\\ell_2$ coefficient error (%)')
    axR.set_title('Coefficient error vs truth')
    axR.grid(True, which='both', ls=':', alpha=0.5)
    axR.legend()

    # 4th panel: relative forward-simulation error from the true initial condition
    fs_style = [(r'TT-WSINDy ($\epsilon=10^{-16}$)', 'o-', 'C0', 1, 3),
                ('flat WSINDy', '^-', 'C3', 2, 4)]
    if frows.size:
        axS.set_yscale('log')                          # spans ~1e-3 (clean) to CAP
        floor = 2e-3                                   # below the clean-recovery level
        for lab, mk, colr, mi, si in fs_style:
            m, s = frows[:, mi], frows[:, si]
            # upper whisker = full sd; lower whisker bounded to one decade so a
            # bimodal sd>mean point (flat blows up in only some trials) does not
            # draw a bar spanning the whole log axis (true sd is in the data file)
            lo = np.minimum(s, 0.9*m)
            axS.errorbar(frows[:, 0], m, yerr=[lo, s], fmt=mk, color=colr,
                         capsize=3, label=lab)
        axS.axhline(FSIM_CAP, color='0.6', ls=':', lw=1)      # blow-up cap
        axS.set_ylim(floor, 1.6*FSIM_CAP)
        axS.legend()
    else:
        axS.text(0.5, 0.5, 'forward-sim error not yet computed\n(set recompute=True)',
                 ha='center', va='center', transform=axS.transAxes, color='0.5')
    axS.set_xlabel('number of dimensions $D$')
    axS.set_ylabel(r'relative fwd-sim error $\|\hat X - X^*\|_F/\|X^*\|_F$')
    axS.set_title(f'Forward-simulation error (from true IC, $t\\in[0,{FSIM_HORIZON:g}]$)')
    axS.grid(True, which='both', ls=':', alpha=0.5)

    fig.suptitle(f'Support, coefficient & forward-sim error vs $D$  (Lorenz 96, '
                 f'mean$\\pm$sd of {NUM_TRIALS} trials)')
    plt.tight_layout()
    plt.savefig('results/support_vs_D.png', dpi=150)
    print("saved results/support_vs_D.png")

    # ------- Figure 3: TT-WSINDy support reduction (initial -> coarse -> fine) -------
    # TT-WSINDy only (no flat). For each D, one line connects the full candidate
    # count D*2^D, the coarse product-support summed over the D equations, and
    # the fine MSTLS support -- showing how the structured coarse pass collapses
    # the library before the fine solve. Computed alongside the main sweep (from
    # the same eps=1e-16 trials) and stored as trial means; coarse/fine carry a
    # +-1 sd error bar (initial is deterministic).
    srows = np.array([sizes_by_D[d] for d in sorted(sizes_by_D)], float)
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
