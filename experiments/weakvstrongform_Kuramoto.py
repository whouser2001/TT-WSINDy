"""
Weak form (TT-WSINDy) vs. strong form (MANDy) coefficient accuracy on Kuramoto.

Mirrors weakvstrongform_FPUT.py: both regressions pool n_traj trajectories,
held as a (P, D, M) array and flattened to (D, P*M) for the library, with the
test-function convolution (weak form) and the finite difference (strong form)
taken one trajectory at a time so no row of the regression mixes trajectories.
Kuramoto is FIRST order, so the test function carries one derivative (order 1)
and the strong form uses a central first difference, as in the L96 script.
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

# Tight tolerances so the integrator is never what limits MANDy: the point of
# the comparison is the O(dt^2) error of the finite-difference stencil, and an
# integrator whose own local rule resembled that stencil would hide it.
RTOL = ATOL = 1e-12


# ---------------------------------------------------------------------------
# Kuramoto model with distributed frequencies and a pinning term
# ---------------------------------------------------------------------------
def kuramoto(x, t, omega, K, h):
    """Kuramoto right-hand side for arbitrary d.

        x_i' = omega_i + (K/d) sum_{j=1}^{d} sin(x_j - x_i) + h sin(x_i)

    Parameters
    ----------
    x : ndarray (d,)
        Phases. Unwrapped (not reduced mod 2 pi); see kuramoto_trajectory.
    t : float
        Time, unused (autonomous), present for odeint's signature.
    omega : ndarray (d,)
        Natural frequencies.
    K : float
        Coupling strength.
    h : float
        Pinning / forcing amplitude.

    Returns
    -------
    dx : ndarray (d,)
        Phase velocities.
    """
    S, C = np.sin(x), np.cos(x)
    # Mean-field form of the coupling: expanding sin(x_j - x_i) gives
    #     sum_j sin(x_j - x_i) = cos(x_i) sum_j sin(x_j) - sin(x_i) sum_j cos(x_j),
    # which is O(d) rather than O(d^2) and exact for any d. The j = i term
    # contributes sin(0) = 0 to both forms, so including it changes nothing.
    return omega + (K / x.size) * (C * S.sum() - S * C.sum()) + h * S


def gen_kuramoto(x0, t, omega, K, h):
    """Integrate the Kuramoto model to a d x M data matrix."""
    return odeint(kuramoto, x0, t, args=(omega, K, h),
                  rtol=RTOL, atol=ATOL).T


def kuramoto_trajectory(d, M, dt=0.01, K=0.5, h=0.5, omega_min=-5.0,
                        omega_max=5.0, seed=0, omega=None):
    """Generate one Kuramoto trajectory of d oscillators, uniformly sampled.

    Frequencies are EQUIDISTANT on [omega_min, omega_max] --
    omega = linspace(omega_min, omega_max, d) -- unless an omega is supplied,
    and the initial phases are drawn uniformly from (-pi, pi]. The frequencies
    are therefore deterministic, and `seed` only sets the initial phases.

    Spreading the frequencies this wide keeps the phases from LOCKING, which is
    what would destroy the {1, sin, cos} candidate library: once the phases lock
    into one rigidly rotating cluster every sin(x_j - x_i) goes constant, the
    trajectory collapses onto a near-1-D curve on the d-torus and the library
    goes rank deficient -- the same trap as starting Lorenz-96 at its
    equilibrium. For a uniform frequency spread on [-g, g] the asymptotic
    locking threshold is K_c = 4g/pi (~6.4 at g=5), and finite size plus the
    pinning term (h sin(x_i) pulls every phase toward pi) both push locking
    below that, so keep K well under it.

    The orbit also needs enough samples and a long enough span for the
    {1, sin, cos} library to reach full rank. Equidistant frequencies are
    commensurate -- every omega_i is an integer multiple of the spacing
    2g/(d-1) -- so the orbit closes rather than filling the torus densely, and
    it can in principle explore less than a random draw would. Measured at
    d = 4, K = 2, h = 0.2, 4 x 10201 snapshots (T = 102): order parameter
    r = 0.45 and the 81-candidate library is FULL rank, the same as the random
    draw it replaced. Still worth checking the rank the experiment prints when
    changing d, rather than assuming it.

    With an odd d the grid contains omega = 0 exactly; keep h below the smallest
    |omega_i| so that oscillator is not pinned outright (at d = 4 the smallest
    is 5/3, at d = 5 it is 0).

    Parameters
    ----------
    d : int
        Number of oscillators (arbitrary; the right-hand side is O(d)).
    M : int
        Number of recorded snapshots, spaced by dt.
    dt : float
        Spacing between recorded snapshots. Not an integrator step -- odeint
        adapts its own, which is what leaves the finite-difference LHS with a
        genuine O(dt^2) truncation error.
    K : float
        Coupling strength.
    h : float
        Pinning / forcing amplitude.
    omega_min, omega_max : float
        Endpoints of the equidistant frequency grid.
    seed : int
        RNG seed for the initial phases (the frequencies are deterministic).
    omega : ndarray (d,), optional
        Frequencies to use instead of drawing them. Pooled trajectories must
        share one omega, or they are not samples of the same system.

    Returns
    -------
    t : ndarray (M,)
        Sample times.
    X : ndarray (d, M)
        Phases along the trajectory, UNWRAPPED: they grow without bound at
        roughly the mean frequency per unit time. Do not reduce them mod 2 pi -- the
        library {1, sin, cos} cannot tell the difference, but a 2 pi jump would
        wreck both the finite difference and the weak-form convolution.
        Because |X| grows with the time span, scaling a noise level by rms(X)
        (as the FPUT and L96 scripts do) is meaningless here; the noise sweep
        below perturbs the phases by an absolute number of radians.
    omega : ndarray (d,)
        The natural frequencies used, needed to state the true coefficients.
    """
    rng = np.random.default_rng(seed)
    if omega is None:
        omega = np.linspace(omega_min, omega_max, d)
    x0 = rng.uniform(-np.pi, np.pi, d)
    t = np.arange(M) * dt
    return t, gen_kuramoto(x0, t, omega, K, h), omega


def kuramoto_trajectories(d, M, n_traj, dt=0.01, K=0.5, h=0.5,
                          omega_min=-5.0, omega_max=5.0, seed=0):
    """Generate n_traj Kuramoto trajectories of one system, uniformly sampled.

    The equidistant frequencies are shared by every trajectory; the
    trajectories differ only in their random initial phases (drawn with
    seed + p). Pooling trajectories with different omega would pool different
    systems, so no single coefficient tensor could fit them.

    Returns
    -------
    t : ndarray (M,)
        Sample times, common to all trajectories.
    positions : ndarray (n_traj, d, M)
    omega : ndarray (d,)
    """
    omega = np.linspace(omega_min, omega_max, d)
    trajs = [kuramoto_trajectory(d, M, dt=dt, K=K, h=h, seed=seed + p,
                                 omega=omega) for p in range(n_traj)]
    return trajs[0][0], np.stack([X for _, X, _ in trajs]), omega


def flatten_trajectories(Xs):
    """Concatenate trajectories along time: (P, D, M) -> (D, P*M).

    The blocking is trajectory-major, matching what feature_tensor's n_traj
    and the per-trajectory left-hand sides below assume.
    """
    return np.concatenate(list(Xs), axis=1)


# ---------------------------------------------------------------------------
# True coefficient tensor (exact expansion of the Kuramoto right-hand side)
# ---------------------------------------------------------------------------
# With f = {1, sin, cos} the candidate index in a dimension picks which of the
# three functions acts on that oscillator, so every term of the right-hand side
# maps onto one entry of the (J,)*D coefficient tensor:
#     x_i' = omega_i * 1
#          + (K/d) sum_{j != i} [ sin(x_j) cos(x_i) - cos(x_j) sin(x_i) ]
#          + h sin(x_i).
# The j = i term is what makes this exact: sin(x_i) cos(x_i) is NOT in the span
# of {1, sin, cos} (it is sin(2 x_i)/2), but it appears twice with opposite
# signs and cancels analytically, so it never has to be represented. Hence the
# sum below skips j = i rather than storing anything for it.
def true_coeffs(D, J, omega, K, h):
    """Exact coefficient tensor of the Kuramoto right-hand side, per output dim.

    Parameters
    ----------
    D : int
        Number of oscillators.
    J : int
        Number of candidate functions per dimension, assumed to be
        f = {1, sin, cos}; J >= 3 is required.
    omega : ndarray (D,)
        Natural frequencies.
    K, h : float
        Coupling strength and pinning amplitude.

    Returns
    -------
    Ws : list of ndarray
        Ws[i] is the (J,)*D coefficient tensor of oscillator i's equation, i.e.
        Ws[i][j_0, ..., j_{D-1}] is the coefficient of prod_d f[j_d](x_d) in
        x_i'. Each equation has 2D nonzeros: omega_i, h, and +-K/D for each of
        the D-1 other oscillators.

    Raises
    ------
    ValueError
        If J < 3, since the model needs all of {1, sin, cos}.
    """
    if J < 3:
        raise ValueError(
            f"Kuramoto needs f = {{1, sin, cos}}, so J >= 3 (got J={J})"
        )

    Ws = []
    for i in range(D):
        W = np.zeros((J,) * D)

        idx = [0] * D
        W[tuple(idx)] += omega[i]                   # omega_i * 1

        idx = [0] * D; idx[i] = 1
        W[tuple(idx)] += h                          # h sin(x_i)

        for j in range(D):
            if j == i:
                continue                            # cancels analytically
            idx = [0] * D; idx[j] = 1; idx[i] = 2
            W[tuple(idx)] += K / D                  # +(K/d) sin(x_j) cos(x_i)
            idx = [0] * D; idx[j] = 2; idx[i] = 1
            W[tuple(idx)] -= K / D                  # -(K/d) cos(x_j) sin(x_i)

        Ws.append(W)
    return Ws


_FLABEL = ['1', 'sin(x{})', 'cos(x{})']

def _mono_label(idx):
    """Readable label for a candidate multi-index, e.g. 'sin(x0) cos(x1)'."""
    parts = [_FLABEL[j].format(d) for d, j in enumerate(idx) if j > 0]
    return "1" if not parts else " ".join(parts)


# ---------------------------------------------------------------------------
# TT-PI regression -> dense, true-scale coefficient tensor
# ---------------------------------------------------------------------------
def tt_pi_coeffs(Theta, y, threshold, D, J):
    """One TT-PI solve, returned as a dense (J,)*D tensor in original units.

    TT_PI uses overwrite=False, so Theta is not mutated and no copy is needed.
    """
    W = Theta.TT_PI(y)
    return np.asarray(W.full()).reshape((J,) * D)


def weak_coefficients(Xs, f, t0, tM, D, J,
                      r_frac=1.0 / 60.0, degree=16, threshold=1e-10):
    """TT-WSINDy (weak form) coefficient tensors, one row per output dim.

    Kuramoto is first order, so one integration by parts moves the single
    derivative onto the test function: the LHS is -<x, phi'> = <x', phi>
    (test-function order 1, so dphi is phi'), and the library is convolved with
    phi. No derivative of the data is computed.

    Xs is (P, D, M): P trajectories, each of M snapshots spanning [t0, tM]. The
    test-function radius is a fraction r_frac of that per-trajectory span, and
    both the library and the LHS are convolved one trajectory at a time, so no
    weak-form row straddles a trajectory boundary.
    """
    M = Xs.shape[2]
    phi, dphi = piecewise_polynomial((tM - t0) * r_frac, degree, t0, tM, M,
                                     order=1)
    Theta = feature_tensor(flatten_trajectories(Xs), f, phi=phi, low_rank=True,
                           n_traj=Xs.shape[0])
    # the order-1 dphi equals phi', so -correlate(x, dphi) = -<x, phi'>
    Y = -1 * np.concatenate(
        [correlate(X, np.expand_dims(dphi, axis=0), mode='valid') for X in Xs],
        axis=1).transpose()                                     # (P*Mp, D)
    return np.stack([tt_pi_coeffs(Theta, Y[:, d], threshold, D, J)
                     for d in range(D)])


def strong_coefficients(Xs, f, dt, D, J, threshold=1e-10):
    """MANDy (strong form) coefficient tensors, one row per output dim.

    The LHS x' is a 3-point central finite difference of each trajectory; the
    library is sampled pointwise at the interior snapshots where x' is defined.
    Xs is (P, D, M), as in weak_coefficients.
    """
    Xdot = (Xs[:, :, 2:] - Xs[:, :, :-2]) / (2 * dt)            # (P, D, M-2)
    Theta = feature_tensor(flatten_trajectories(Xs[:, :, 1:-1]), f, phi=None,
                           low_rank=True, n_traj=Xs.shape[0])
    return np.stack([tt_pi_coeffs(Theta, Xdot[:, d, :].ravel(), threshold, D, J)
                     for d in range(D)])


def rel_err(W, Wtrue):
    """Relative 2-norm error over the full stacked coefficient tensor."""
    return np.linalg.norm(W - Wtrue) / np.linalg.norm(Wtrue)

def rel_err_dims(W, Wtrue):
    """Relative 2-norm error of EACH output dimension's coefficient tensor.

    Returns an array of length D, entry d being
    ||W[d] - Wtrue[d]|| / ||Wtrue[d]||, i.e. the error of the equation for
    x_d alone. The stacked rel_err above is a norm-weighted blend of these, so
    an equation with small true coefficients can be recovered badly without the
    stacked number showing it -- which is what the per-dimension curves expose.
    """
    return np.array([np.linalg.norm(W[d] - Wtrue[d]) / np.linalg.norm(Wtrue[d])
                     for d in range(W.shape[0])])


def library_rank(X, f, J, tol=1e-10):
    """Numerical rank of the strong (pointwise) candidate library, and J^D.

    A rank below J^D means the candidates are linearly dependent along the data,
    so the un-thresholded TT-PI cannot recover the sparse truth. Pass the
    flattened (D, P*M) data to see the rank pooled over all trajectories, which
    is what the regression actually sees.
    """
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
# Experiment
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ----- parameters -----
    D = 4               # oscillators
    n_traj = 5          # independent trajectories (distinct initial phases)
    M = 10000            # snapshots per trajectory
    #M = 1000
    dt = 0.01           # snapshot spacing
    #K = 0.5             # coupling strength
    K = 2
    #h = 0.5             # pinning amplitude
    h = 0.2
    omega_min = -5.0    # frequencies equidistant on [omega_min, omega_max],
    omega_max = 5.0     # spread wide to keep the phases unlocked
    r_frac = 1.0 / 120.0 # test-fn radius as a fraction of the per-trajectory span
    degree = 16         # test-function polynomial degree
    threshold = 1e-16   # TT-PI singular-value truncation (regularized pinv)

    noise_levels = np.array([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 4e-1])
    n_trials = 1        # noise realizations averaged per level

    f = [lambda x: 1, lambda x: np.sin(x), lambda x: np.cos(x)]
    J = len(f)

    # ----- generate the trajectories -----
    t, Xs, omega = kuramoto_trajectories(D, M, n_traj, dt=dt, K=K, h=h,
                                         omega_min=omega_min,
                                         omega_max=omega_max, seed=3)
    t0, tM = t[0], t[-1]
    Xflat = flatten_trajectories(Xs)

    order_param = np.mean([np.abs(np.exp(1j * Xs[p]).mean(axis=0)).mean()
                           for p in range(n_traj)])
    Ffull = np.stack([kuramoto(Xflat[:, k], 0.0, omega, K, h)
                      for k in range(Xflat.shape[1])])
    coupling_frac = np.linalg.norm(Ffull - omega) / np.linalg.norm(Ffull)
    rank, jD = library_rank(Xflat, f, J)

    print("=" * 66)
    print("Kuramoto trajectories")
    print(f"  D={D} oscillators, {n_traj} trajectories x M={M} snapshots"
          f" = {n_traj * M} total, dt={dt}, T={tM:.1f}")
    print(f"  K={K}, h={h}, omega equidistant on "
          f"[{omega_min:g}, {omega_max:g}]: "
          + ", ".join(f"{w:g}" for w in omega))
    print(f"  order parameter r       : {order_param:.3f}"
          f"  {'(unlocked)' if order_param < 0.9 else '(LOCKED -- lower K)'}")
    print(f"  phase range             : [{Xflat.min():.3f}, {Xflat.max():.3f}]")
    print(f"  coupling force fraction : {coupling_frac:.3f}")
    print(f"  library rank            : {rank}/{jD}"
          f"  {'(full rank)' if rank == jD else '(RANK DEFICIENT -- raise M/T)'}")
    print("=" * 66)

    Wtrue = np.stack(true_coeffs(D, J, omega, K, h))

    # ----- clean-data sanity report -----
    Ww0 = weak_coefficients(Xs, f, t0, tM, D, J, r_frac=r_frac, degree=degree,
                            threshold=threshold)
    Ws0 = strong_coefficients(Xs, f, dt, D, J, threshold=threshold)
    print("clean-data relative coefficient error (no noise):")
    print(f"  TT-WSINDy (weak)  : {rel_err(Ww0, Wtrue):.3e}")
    print(f"  MANDy     (strong): {rel_err(Ws0, Wtrue):.3e}")
    print("-" * 66)
    print("oscillator 0 -- true vs. recovered coefficients (nonzero true terms):")
    print(f"  {'candidate':>18}  {'true':>9}  {'weak':>11}  {'strong':>11}")
    for idx in map(tuple, np.argwhere(np.abs(Wtrue[0]) > 1e-12)):
        print(f"  {_mono_label(idx):>18}  {Wtrue[0][idx]:>9.3f}"
              f"  {Ww0[0][idx]:>11.4f}  {Ws0[0][idx]:>11.4f}")
    print("=" * 66)

    # ----- noise sweep -----
    # errors are kept per output dimension: (n_sigma, n_trials, D)
    weak_err = np.zeros((len(noise_levels), n_trials, D))
    strong_err = np.zeros((len(noise_levels), n_trials, D))
    # sigma is an ABSOLUTE phase perturbation in radians: the phases are
    # unwrapped and grow with the time span, so scaling sigma by rms(X) (as the
    # FPUT and L96 scripts do) would tie the noise level to T.

    print(f"noise sweep (relative coefficient error per output dim, "
          f"mean over {n_trials} trials):")
    print("  " + f"{'noise sigma':>12}  {'form':>6}"
          + "".join(f"{f'x{d}':>12}" for d in range(D)))
    sweep_st = time()
    for i, sigma in enumerate(noise_levels):
        for tr in range(n_trials):
            rng = np.random.default_rng(1000 * i + tr)
            Xn = Xs + sigma * rng.standard_normal(Xs.shape)
            Ww = weak_coefficients(Xn, f, t0, tM, D, J, r_frac=r_frac,
                                   degree=degree, threshold=threshold)
            Ws = strong_coefficients(Xn, f, dt, D, J, threshold=threshold)
            weak_err[i, tr] = rel_err_dims(Ww, Wtrue)
            strong_err[i, tr] = rel_err_dims(Ws, Wtrue)
        wm, sm = weak_err[i].mean(0), strong_err[i].mean(0)
        print("  " + f"{sigma:>12.0e}  {'weak':>6}"
              + "".join(f"{v:>12.3e}" for v in wm))
        print("  " + f"{'':>12}  {'strong':>6}"
              + "".join(f"{v:>12.3e}" for v in sm))
    print(f"(noise sweep walltime: {time() - sweep_st:.1f}s)")
    print("=" * 66)

    # ----- save results -----
    # column layout: noise, then four D-wide blocks (weak_mean, weak_std,
    # strong_mean, strong_std), one column per output dimension in each block.
    # plot_weakvstrongform.py recovers D as (ncols - 1) // 4.
    os.makedirs("results", exist_ok=True)
    out = np.column_stack([noise_levels,
                           weak_err.mean(1), weak_err.std(1),
                           strong_err.mean(1), strong_err.std(1)])
    cols = " ".join(["noise"]
                    + [f"{stat}_d{d}"
                       for stat in ("weak_mean", "weak_std",
                                    "strong_mean", "strong_std")
                       for d in range(D)])
    np.savetxt(f"results/weakvstrongformKuramoto.txt", out,
               header=(f"D={D} n_traj={n_traj} M={M} dt={dt} K={K} h={h} "
                       f"omega_min={omega_min} omega_max={omega_max} "
                       f"r_frac={r_frac:.4f} n_trials={n_trials}")
                      + "\n" + cols)

    # ----- plot -----
    wm, ws = weak_err.mean(1), weak_err.std(1)        # (n_sigma, D)
    sm, ss = strong_err.mean(1), strong_err.std(1)
    x_axis = noise_levels

    # one shade per output dimension, blues for the weak form and reds for the
    # strong form, so the two families stay separable with D lines each
    wcol = plt.cm.Blues(np.linspace(0.45, 0.95, D))
    scol = plt.cm.Reds(np.linspace(0.45, 0.95, D))

    plt.figure(figsize=(7.5, 5))
    # all weak lines first, then all strong, so the two-column legend fills
    # column-major with one form per column
    for d in range(D):
        plt.errorbar(x_axis, wm[:, d], yerr=ws[:, d], marker='o', capsize=3,
                     color=wcol[d], label=rf'weak $x_{{{d}}}$')
    for d in range(D):
        plt.errorbar(x_axis, sm[:, d], yerr=ss[:, d], marker='s', capsize=3,
                     ls='--', color=scol[d], label=rf'strong $x_{{{d}}}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-6, 1e0)
    plt.xlabel(r'phase noise $\sigma$ (radians)')
    plt.ylabel('relative coefficient error')
    plt.title(('Weak vs. strong form TT regression on Kuramoto '
               f'(D={D}, {n_traj} traj. x M={M})'))
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"results/weakvstrongformKuramoto.png", dpi=150)
    plt.show()
