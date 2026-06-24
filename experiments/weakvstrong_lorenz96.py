"""
Higher-dimensional weak (TT-WSINDy) vs. strong (MANDy) accuracy on Lorenz-96.

This is the higher-dimensional companion to `weakvstrongform.py` (FPUT). FPUT is
quasi-periodic, so a single trajectory leaves most candidate monomials
un-excited and the J^D library is badly rank deficient -- pure TT-PI (no
sequential thresholding) cannot recover the coefficients beyond a small system
(it fails already at D=4 under noise, and the rank collapses entirely by D=6).

Lorenz-96,

    dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F ,    indices mod D,

fixes both problems:
  * It is CHAOTIC at F=8, so one trajectory is ergodic and excites the library
    -> the candidate monomials are linearly independent and the un-thresholded
    TT-PI recovers the true (sparse) coefficients.
  * Its nonlinearity is a product of DISTINCT variables, so the separable
    library f = {1, x} (J=2) already spans it exactly. The true row-i model is

        +1 * x_{i+1} x_{i-1}   -1 * x_{i-2} x_{i-1}   -1 * x_i   +F * 1

    Every other monomial of the 2^D candidates (other pair products, all
    higher-order products) is a spurious distractor. Using J=2 keeps the
    candidate count at 2^D instead of FPUT's 4^D, so the tensor train reaches
    higher state dimension before the problem size explodes.

It is a first-order system, so:
  * Strong form (MANDy): the LHS x_dot is a central finite difference of the
    trajectory; library sampled pointwise.   Theta(x) W = x_dot.
  * Weak form (TT-WSINDy): the LHS is the weak projection <x_dot, phi> =
    -<x, phi'> against an order-1 test function; library convolved with phi.
    No derivative of the data is taken.

This script runs two experiments:
  Part 1 -- noise sweep at a moderate dimension (D=6): coefficient error vs.
    additive noise. The weak form beats finite-difference MANDy across noise
    levels and degrades gracefully (the robustness story FPUT could not deliver:
    there both forms blew up under any noise).
  Part 2 -- dimension scan (D = 5..8) on clean data: the weak form recovers the
    coefficients to ~5e-4 essentially independently of D, up to 256 candidate
    functions, ~10-50x sharper than the strong form.

Identifiability / scaling note
------------------------------
Full library rank (the precondition for un-thresholded TT-PI) holds robustly
through D=8 (2^8 = 256 candidates) at M~2000. Beyond that the *high-order*
monomials (products of many oscillators) stay un-excited even on the chaotic
attractor, so the library goes rank deficient (e.g. D=10 gives rank ~640/1024),
and restoring it would need M far larger than the O(M^2) feature cores allow in
memory. The scripts print the realized rank; trust results only where it is
full.

The test-function width is the key knob and it pulls two ways: a WIDE window
smooths measurement noise but is poorly conditioned once the candidate library
is large, while a NARROW window conditions the large high-D library but its
peaked phi' amplifies noise. Hence Part 1 (noise) uses a wide window at moderate
D and Part 2 (high-D clean) a narrow one. At the top of the range (D=8) the
narrow window required for clean recovery does sacrifice noise robustness.

Run from the experiments/ directory:  python weakvstrong_lorenz96.py
"""
import os, sys, itertools
sys.path.insert(0, '.')
sys.path.insert(0, '../TT-WSINDy')
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.signal import correlate
from scipy.integrate import odeint
from feature_tensor import feature_tensor
from test_function import piecewise_polynomial


# ---------------------------------------------------------------------------
# Lorenz-96 trajectory
# ---------------------------------------------------------------------------
def l96_rhs(x, t, F):
    """Lorenz-96 vector field, periodic boundaries."""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def lorenz96_trajectory(D, M, dt=0.01, F=8.0, seed=0, burn=10.0):
    """One chaotic Lorenz-96 trajectory, started on the attractor.

    Parameters
    ----------
    D : int
        Number of oscillators (state dimension).
    M : int
        Number of recorded snapshots (spaced by dt).
    dt : float
        Snapshot spacing.
    F : float
        Constant forcing (F=8 is the standard chaotic regime).
    seed : int
        RNG seed for the initial perturbation.
    burn : float
        Time integrated and discarded so the sample starts on the attractor.

    Returns
    -------
    t : ndarray (M,)
        Sample times.
    X : ndarray (D, M)
        Trajectory x(t).
    """
    rng = np.random.default_rng(seed)
    x0 = F * np.ones(D) + 0.01 * rng.standard_normal(D)
    x0 = odeint(l96_rhs, x0, np.linspace(0.0, burn, 1000), args=(F,))[-1]
    t = np.arange(M) * dt
    X = odeint(l96_rhs, x0, t, args=(F,)).T
    return t, X


def l96_true_coeffs(D, J, F):
    """Exact coefficient tensor of the Lorenz-96 RHS, per output oscillator.

    Returns Ws with Ws[i][j_0, ..., j_{D-1}] the coefficient of
    prod_d f[j_d](x_d) in dx_i/dt. Assumes D >= 5 so the four neighbour
    positions are distinct (J=2, so f index = power, which never exceeds 1 here).
    """
    Ws = []
    for i in range(D):
        W = np.zeros((J,) * D)
        W[(0,) * D] += F                                   # + F (constant)
        idx = [0] * D; idx[i] = 1; W[tuple(idx)] += -1.0   # - x_i
        idx = [0] * D; idx[(i + 1) % D] += 1; idx[(i - 1) % D] += 1
        W[tuple(idx)] += 1.0                               # + x_{i+1} x_{i-1}
        idx = [0] * D; idx[(i - 2) % D] += 1; idx[(i - 1) % D] += 1
        W[tuple(idx)] += -1.0                              # - x_{i-2} x_{i-1}
        Ws.append(W)
    return Ws


def _mono_label(idx):
    """Readable label for a monomial multi-index, e.g. (1,0,1,0) -> 'x0 x2'."""
    parts = [f"x{d}" if p == 1 else f"x{d}^{p}" for d, p in enumerate(idx) if p]
    return "1" if not parts else " ".join(parts)


# ---------------------------------------------------------------------------
# TT-PI regression -> dense, true-scale coefficient tensor
# ---------------------------------------------------------------------------
def _norm_product(Theta, D, J):
    """Per-monomial product of feature_tensor's per-feature norms (see below)."""
    norms = Theta.feature_norms                  # (J, D)
    s = np.ones((J,) * D)
    for idx in np.ndindex(*([J] * D)):
        p = 1.0
        for d in range(D):
            p *= norms[idx[d], d]
        s[idx] = p
    return s


def tt_pi_coeffs(Theta, y, threshold, D, J):
    """One TT-PI solve as a dense (J,)*D tensor in original units.

    feature_tensor scales each feature by its column norm, so the raw TT-PI
    estimate is in the normalized space; dividing by the per-monomial norm
    product undoes that scaling. (feature_tensor.unscale multiplies by the
    norms, which is the wrong direction, so we divide here.) TT_PI uses
    overwrite=False, so Theta is not mutated and no copy is needed.
    """
    W = Theta.TT_PI(y, threshold=threshold)
    raw = np.asarray(W.full()).reshape((J,) * D)
    return raw / _norm_product(Theta, D, J)


def weak_coefficients(X, f, t0, tM, M, D, J,
                      r_frac=1.0 / 50.0, degree=16, threshold=1e-10):
    """TT-WSINDy (weak form) coefficient tensors, one row per output dim.

    First-order weak form: LHS = <x_dot, phi> = -<x, phi'> (order-1 test
    function); library convolved with phi. No data derivative is computed.
    """
    phi, dphi = piecewise_polynomial((tM - t0) * r_frac, degree, t0, tM, M,
                                     order=1)
    Theta = feature_tensor(X, f, phi=phi)
    # order-1 dphi equals phi', so -correlate(X, dphi) = -<x, phi'> = <x_dot, phi>
    Y = -1 * correlate(X, np.expand_dims(dphi, axis=0), mode='valid').transpose()
    return np.stack([tt_pi_coeffs(Theta, Y[:, d], threshold, D, J)
                     for d in range(D)])


def strong_coefficients(X, f, dt, D, J, threshold=1e-10):
    """MANDy (strong form) coefficient tensors, one row per output dim.

    LHS x_dot is a 3-point central finite difference; library sampled pointwise
    at the interior snapshots where x_dot is defined.
    """
    Xdot = (X[:, 2:] - X[:, :-2]) / (2.0 * dt)     # (D, M-2)
    Theta = feature_tensor(X[:, 1:-1], f, phi=None)
    return np.stack([tt_pi_coeffs(Theta, Xdot[d], threshold, D, J)
                     for d in range(D)])


def rel_err(W, Wtrue):
    """Relative 2-norm error over the full stacked coefficient tensor."""
    return np.linalg.norm(W - Wtrue) / np.linalg.norm(Wtrue)


def library_rank(X, f, J, tol=1e-10):
    """Numerical rank of the strong (pointwise) monomial library, and J^D."""
    D, M = X.shape
    fmap = list(itertools.product(*[range(J)] * D))
    bdata = np.stack([np.vectorize(f[j])(X).astype(float) for j in range(J)])
    G = np.ones((len(fmap), M))
    for k, tup in enumerate(fmap):
        for d in range(D):
            G[k] *= bdata[tup[d], d]
    s = np.linalg.svd(G, compute_uv=False)
    return int(np.sum(s / s[0] > tol)), len(fmap)


# ---------------------------------------------------------------------------
# Experiment pieces
# ---------------------------------------------------------------------------
def noise_sweep(D, M, dt, F, f, J, r_frac, threshold, noise_levels, n_trials):
    """Coefficient error vs. additive noise at fixed D. Returns means/stds."""
    t, X = lorenz96_trajectory(D, M, dt=dt, F=F, seed=0)
    t0, tM = t[0], t[-1]
    rank, jD = library_rank(X, f, J)
    Wtrue = np.stack(l96_true_coeffs(D, J, F))

    # nonlinear (quadratic) fraction of the RHS, as an excitation diagnostic
    rhs = np.stack([l96_rhs(X[:, k], 0.0, F) for k in range(M)])
    lin = np.stack([-X[:, k] + F for k in range(M)])
    nl_frac = np.linalg.norm(rhs - lin) / np.linalg.norm(rhs)

    print("=" * 70)
    print(f"NOISE SWEEP  (Lorenz-96, D={D}, M={M}, dt={dt}, F={F})")
    print(f"  displacement range      : [{X.min():.2f}, {X.max():.2f}]")
    print(f"  nonlinear RHS fraction  : {nl_frac:.3f}")
    print(f"  library rank            : {rank}/{jD}"
          f"  {'(full rank)' if rank == jD else '(RANK DEFICIENT -- lower D or raise M)'}")
    print("=" * 70)

    Ww0 = weak_coefficients(X, f, t0, tM, M, D, J, r_frac=r_frac, threshold=threshold)
    Ws0 = strong_coefficients(X, f, dt, D, J, threshold=threshold)
    print("clean-data relative coefficient error:")
    print(f"  TT-WSINDy (weak)  : {rel_err(Ww0, Wtrue):.3e}")
    print(f"  MANDy     (strong): {rel_err(Ws0, Wtrue):.3e}")
    print("-" * 70)
    print("oscillator 0 -- true vs. recovered (nonzero true terms):")
    print(f"  {'monomial':>12}  {'true':>7}  {'weak':>11}  {'strong':>11}")
    for idx in map(tuple, np.argwhere(np.abs(Wtrue[0]) > 1e-12)):
        print(f"  {_mono_label(idx):>12}  {Wtrue[0][idx]:>7.3f}"
              f"  {Ww0[0][idx]:>11.4f}  {Ws0[0][idx]:>11.4f}")
    print("=" * 70)

    weak_err = np.zeros((len(noise_levels), n_trials))
    strong_err = np.zeros((len(noise_levels), n_trials))
    xstd = X.std()
    print(f"noise sweep (mean over {n_trials} trials):")
    print(f"  {'sigma':>10}  {'TT-WSINDy':>12}  {'MANDy':>12}  {'ratio S/W':>10}")
    st = time()
    for i, sigma in enumerate(noise_levels):
        for tr in range(n_trials):
            rng = np.random.default_rng(1000 * i + tr)
            Xn = X + sigma * xstd * rng.standard_normal(X.shape)
            Ww = weak_coefficients(Xn, f, t0, tM, M, D, J,
                                   r_frac=r_frac, threshold=threshold)
            Ws = strong_coefficients(Xn, f, dt, D, J, threshold=threshold)
            weak_err[i, tr] = rel_err(Ww, Wtrue)
            strong_err[i, tr] = rel_err(Ws, Wtrue)
        wm, sm = weak_err[i].mean(), strong_err[i].mean()
        print(f"  {sigma:>10.0e}  {wm:>12.3e}  {sm:>12.3e}  {sm / wm:>10.1f}")
    print(f"(noise sweep walltime: {time() - st:.1f}s)")
    print("=" * 70)
    return weak_err, strong_err


def dimension_scan(D_list, M, dt, F, f, J, r_frac, threshold, sigma):
    """Clean and noisy coefficient error vs. state dimension D."""
    print("=" * 70)
    print(f"DIMENSION SCAN  (Lorenz-96, M={M}, dt={dt}, F={F}, sigma={sigma:.0e})")
    print("=" * 70)
    print(f"  {'D':>3}  {'J^D':>6}  {'rank':>9}  {'weak(clean)':>12}"
          f"  {'strong(clean)':>13}  {'weak(noisy)':>12}  {'strong(noisy)':>13}")
    res = []
    for D in D_list:
        t, X = lorenz96_trajectory(D, M, dt=dt, F=F, seed=0)
        t0, tM = t[0], t[-1]
        rank, jD = library_rank(X, f, J)
        Wtrue = np.stack(l96_true_coeffs(D, J, F))

        Wwc = weak_coefficients(X, f, t0, tM, M, D, J, r_frac=r_frac, threshold=threshold)
        Wsc = strong_coefficients(X, f, dt, D, J, threshold=threshold)

        rng = np.random.default_rng(0)
        Xn = X + sigma * X.std() * rng.standard_normal(X.shape)
        Wwn = weak_coefficients(Xn, f, t0, tM, M, D, J, r_frac=r_frac, threshold=threshold)
        Wsn = strong_coefficients(Xn, f, dt, D, J, threshold=threshold)

        row = (D, jD, rank, rel_err(Wwc, Wtrue), rel_err(Wsc, Wtrue),
               rel_err(Wwn, Wtrue), rel_err(Wsn, Wtrue))
        res.append(row)
        tag = '' if rank == jD else '  <-- rank deficient'
        print(f"  {D:>3}  {jD:>6}  {rank:>9}  {row[3]:>12.3e}"
              f"  {row[4]:>13.3e}  {row[5]:>12.3e}  {row[6]:>13.3e}{tag}")
    print("=" * 70)
    return np.array(res)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ----- shared parameters -----
    F = 8.0                       # chaotic forcing
    dt = 0.01                     # snapshot spacing
    threshold = 1e-10             # TT-PI singular-value truncation
    f = [lambda x: 1, lambda x: x]   # J=2 spans Lorenz-96 exactly
    J = len(f)
    # The test-function width trades off two things. A WIDE window smooths noise
    # (good robustness) but is poorly conditioned once the candidate library is
    # large; a NARROW window conditions the large high-D library (good clean
    # recovery) but its peaked phi' amplifies noise. So the noise-robustness
    # demo uses a moderate D with a wide window, and the high-D scaling demo a
    # narrow window on clean data.
    r_frac_noise = 1.0 / 25.0     # wide  window: noise sweep (moderate D)
    r_frac_scan = 1.0 / 50.0      # narrow window: dimension scan (clean, high D)

    os.makedirs("results", exist_ok=True)

    # ===== Part 1: noise robustness at moderate dimension =====
    # The weak form's advantage over finite-difference MANDy is a noise story,
    # so we show it where the wide window keeps the weak form well behaved.
    D_noise = 6                   # 2^6 = 64 candidate functions
    M_noise = 2000
    noise_levels = np.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1])
    n_trials = 3

    weak_err, strong_err = noise_sweep(
        D_noise, M_noise, dt, F, f, J, r_frac_noise, threshold, noise_levels, n_trials)

    np.savetxt("results/weakvstrong_l96_noise.txt",
               np.column_stack([noise_levels, weak_err.mean(1), weak_err.std(1),
                                strong_err.mean(1), strong_err.std(1)]),
               header="noise weak_mean weak_std strong_mean strong_std")

    wm, ws = weak_err.mean(1), weak_err.std(1)
    sm, ss = strong_err.mean(1), strong_err.std(1)
    floor = max(noise_levels[1] / 10, 1e-5)
    x_axis = np.where(noise_levels > 0, noise_levels, floor)
    plt.figure(figsize=(7, 5))
    plt.errorbar(x_axis, wm, yerr=ws, marker='o', capsize=3,
                 label='TT-WSINDy (weak form)')
    plt.errorbar(x_axis, sm, yerr=ss, marker='s', capsize=3,
                 label='MANDy (strong form, finite diff.)')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel(r'relative noise level $\sigma$  (leftmost point = clean data)')
    plt.ylabel('relative coefficient error')
    plt.title(f'Weak vs. strong TT-PI on Lorenz-96 (D={D_noise}, {J**D_noise} candidates)')
    plt.legend(); plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/weakvstrong_l96_noise.png", dpi=150)

    # ===== Part 2: scaling with state dimension (clean recovery) =====
    D_list = [5, 6, 7, 8]
    M_scan = 2000
    scan_sigma = 1e-2
    res = dimension_scan(D_list, M_scan, dt, F, f, J, r_frac_scan, threshold, scan_sigma)
    np.savetxt("results/weakvstrong_l96_dimscan.txt", res,
               header="D JpowD rank weak_clean strong_clean weak_noisy strong_noisy")

    Ds = res[:, 0]
    plt.figure(figsize=(7, 5))
    plt.plot(Ds, res[:, 3], marker='o', label='weak, clean')
    plt.plot(Ds, res[:, 4], marker='s', label='strong, clean')
    plt.plot(Ds, res[:, 5], marker='o', ls='--', label=f'weak, $\\sigma$={scan_sigma:g}')
    plt.plot(Ds, res[:, 6], marker='s', ls='--', label=f'strong, $\\sigma$={scan_sigma:g}')
    plt.yscale('log')
    plt.xlabel('state dimension $D$  (candidate library size $2^D$)')
    plt.ylabel('relative coefficient error')
    plt.title('Weak vs. strong TT-PI on Lorenz-96: scaling with dimension')
    plt.legend(); plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.xticks(Ds.astype(int))
    plt.tight_layout()
    plt.savefig("results/weakvstrong_l96_dimscan.png", dpi=150)

    plt.show()
