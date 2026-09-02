"""
Weak form (TT-WSINDy) vs. strong form (MANDy) coefficient accuracy on FPUT.
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
# FPUT trajectory generator (adaptive high-order odeint, fixed walls)
# ---------------------------------------------------------------------------
# The integrator is deliberately NOT velocity-Verlet at step dt. Verlet's own
# update rule is algebraically identical to the 3-point central second
# difference,
#     x_{k+1} - 2 x_k + x_{k-1} = dt^2 F(x_k)     (exactly, no O(dt^2) term),
# so MANDy's finite-difference LHS would just be the generator's update rule
# rearranged, and its clean-data error would collapse to roundoff (~1e-14) for
# reasons that have nothing to do with finite differencing. An adaptive
# high-order integrator shares no such identity with the stencil, so MANDy pays
# its genuine O(dt^2) truncation error and the clean-limit comparison against
# the weak form's quadrature floor is meaningful.
RTOL = ATOL = 1e-12     # far below the O(dt^2) stencil error (~1e-4), so the
                        # finite difference, not the integrator, sets MANDy's floor
def _fpu_force(x, beta):
    """Acceleration field F(x) with fixed walls x_{-1} = x_n = 0."""
    xp = np.concatenate(([0.0], x, [0.0]))
    dr = xp[2:] - xp[1:-1]          # x_{i+1} - x_i
    dl = xp[1:-1] - xp[:-2]         # x_i - x_{i-1}
    return (dr - dl) + beta * (dr ** 3 - dl ** 3)


def _fpu_rhs(z, t, n, beta):
    """First-order form of x'' = F(x): state z = [x, v], so z' = [v, F(x)]."""
    return np.concatenate((z[n:], _fpu_force(z[:n], beta)))


def fpu_energy(x, v, beta):
    """Total Hamiltonian H = 1/2 |v|^2 + U(x) of one configuration."""
    xp = np.concatenate(([0.0], x, [0.0]))
    d = xp[1:] - xp[:-1]
    U = np.sum(0.5 * d ** 2 + (beta / 4.0) * d ** 4)
    return 0.5 * np.sum(v ** 2) + U


def fput_trajectory(n, m, dt=0.015, beta=0.7, amplitude=3.0, seed=0):
    """Generate one uniformly-sampled FPUT trajectory.

    Parameters
    ----------
    n : int
        Number of oscillators.
    m : int
        Number of recorded snapshots (spaced by dt).
    dt : float
        Spacing between recorded snapshots. Not an integrator step: odeint
        chooses its own adaptive steps, which is what leaves MANDy's central
        second difference with a real O(dt^2) truncation error.
    beta : float
        Cubic coupling strength.
    amplitude : float
        Half-width of the uniform random initial displacement box. Larger
        amplitude raises the energy, exciting more monomials so the candidate
        library becomes full rank (see the identifiability note in the module
        docstring). Released from rest (v0 = 0).
    seed : int
        RNG seed for the random initial condition.

    Returns
    -------
    t : ndarray (m,)
        Sample times.
    positions : ndarray (n, m)
        x(t) along the trajectory.
    velocities : ndarray (n, m)
        v(t) along the trajectory.
    """
    rng = np.random.default_rng(seed)
    x0 = 2.0 * amplitude * rng.random(n) - amplitude
    z0 = np.concatenate((x0, np.zeros(n)))       # released from rest

    t = np.arange(m) * dt
    z = odeint(_fpu_rhs, z0, t, args=(n, beta), rtol=RTOL, atol=ATOL)
    return t, z[:, :n].T.copy(), z[:, n:].T.copy()


def fput_trajectories(n, m, n_traj, dt=0.015, beta=0.7, amplitude=3.0, seed=0):
    """Generate n_traj FPUT trajectories from independent initial conditions.

    Every trajectory shares the sampling (m snapshots spaced by dt) and differs
    only in its random initial displacement, drawn with seed + p.

    Returns
    -------
    t : ndarray (m,)
        Sample times, common to all trajectories.
    positions : ndarray (n_traj, n, m)
    velocities : ndarray (n_traj, n, m)
    """
    trajs = [fput_trajectory(n, m, dt=dt, beta=beta, amplitude=amplitude,
                             seed=seed + p) for p in range(n_traj)]
    t = trajs[0][0]
    return t, np.stack([x for _, x, _ in trajs]), np.stack([v for _, _, v in trajs])


def flatten_trajectories(Xs):
    """Concatenate trajectories along time: (P, D, M) -> (D, P*M).

    The blocking is trajectory-major, matching what feature_tensor's n_traj
    and the per-trajectory left-hand sides below assume.
    """
    return np.concatenate(list(Xs), axis=1)


# ---------------------------------------------------------------------------
# True coefficient tensor (exact expansion of the FPUT force)
# ---------------------------------------------------------------------------
# A polynomial is a dict {monomial: coeff}, where a monomial is a sorted tuple
# of (oscillator_index, power) pairs. With f = {1, x, x^2, x^3} the candidate
# index in a dimension equals the power of that oscillator, so a monomial maps
# directly onto an entry of the (J,)*D coefficient tensor.
def _poly_add(p, q):
    r = dict(p)
    for k, v in q.items():
        r[k] = r.get(k, 0.0) + v
    return r

def _poly_scale(p, c):
    return {k: c * v for k, v in p.items()}

def _poly_mul(p, q):
    r = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            powers = {}
            for var, pw in m1 + m2:
                powers[var] = powers.get(var, 0) + pw
            key = tuple(sorted(powers.items()))
            r[key] = r.get(key, 0.0) + c1 * c2
    return r

def true_coeffs(D, J, beta):
    """Exact coefficient tensor of the FPUT force, per output oscillator.

    Returns
    -------
    Ws : list of ndarray
        Ws[i] is the (J,)*D coefficient tensor of oscillator i's equation,
        i.e. Ws[i][j_0, ..., j_{D-1}] is the coefficient of
        prod_d f[j_d](x_d) in x_i''.
    """
    Ws = []
    for i in range(D):
        L, R = i - 1, i + 1
        # linear forms dr = x_R - x_i and dl = x_i - x_L (walls drop a term)
        dr = {}
        if R <= D - 1:
            dr[((R, 1),)] = dr.get(((R, 1),), 0.0) + 1.0
        dr[((i, 1),)] = dr.get(((i, 1),), 0.0) - 1.0
        dl = {}
        dl[((i, 1),)] = dl.get(((i, 1),), 0.0) + 1.0
        if L >= 0:
            dl[((L, 1),)] = dl.get(((L, 1),), 0.0) - 1.0

        lin = _poly_add(dr, _poly_scale(dl, -1.0))            # dr - dl
        dr3 = _poly_mul(_poly_mul(dr, dr), dr)
        dl3 = _poly_mul(_poly_mul(dl, dl), dl)
        cub = _poly_scale(_poly_add(dr3, _poly_scale(dl3, -1.0)), beta)
        force = _poly_add(lin, cub)                           # (dr-dl)+beta(...)

        W = np.zeros((J,) * D)
        for mono, coeff in force.items():
            idx = [0] * D
            for var, pw in mono:
                idx[var] = pw
            W[tuple(idx)] += coeff
        Ws.append(W)
    return Ws

def _mono_label(idx):
    """Readable label for a monomial multi-index, e.g. (1,2,0,0) -> 'x0 x1^2'."""
    parts = []
    for d, p in enumerate(idx):
        if p == 1:
            parts.append(f"x{d}")
        elif p > 1:
            parts.append(f"x{d}^{p}")
    return "1" if not parts else " ".join(parts)

# ---------------------------------------------------------------------------
# TT-PI regression -> dense, true-scale coefficient tensor
# ---------------------------------------------------------------------------
def tt_pi_coeffs(Theta, y, threshold, D, J):
    """One TT-PI solve, returned as a dense (J,)*D tensor in original units.

    TT_PI uses overwrite=False, so Theta is not mutated and no copy is needed.
    The raw estimate lives in the normalized feature space; dividing by the
    per-monomial norm product undoes feature_tensor's column scaling.
    """
    W = Theta.TT_PI(y)
    return np.asarray(W.full()).reshape((J,) * D)


def weak_coefficients(Xs, f, t0, tM, D, J,
                      r_frac=1.0 / 60.0, degree=16, threshold=1e-10):
    """TT-WSINDy (weak form) coefficient tensors, one row per output dim.

    The LHS is the weak projection <x, phi''> (test-function order 2, so phi''
    carries both derivatives); the library is convolved with phi. No derivative
    of the data is computed.

    Xs is (P, D, M): P trajectories, each of M snapshots spanning [t0, tM]. The
    test-function radius is a fraction r_frac of that per-trajectory span, and
    both the library and the LHS are convolved one trajectory at a time, so no
    weak-form row straddles a trajectory boundary.
    """
    M = Xs.shape[2]
    phi, dphi = piecewise_polynomial((tM - t0) * r_frac, degree, t0, tM, M,
                                     order=2)
    Theta = feature_tensor(flatten_trajectories(Xs), f, phi=phi, low_rank=True,
                           n_traj=Xs.shape[0])
    # the order-2 dphi equals -phi'', so -correlate(x, dphi) = <x, phi''>
    Y = -1 * np.concatenate(
        [correlate(X, np.expand_dims(dphi, axis=0), mode='valid') for X in Xs],
        axis=1).transpose()                                     # (P*Mp, D)
    return np.stack([tt_pi_coeffs(Theta, Y[:, d], threshold, D, J)
                     for d in range(D)])


def strong_coefficients(Xs, f, dt, D, J, threshold=1e-10):
    """MANDy (strong form) coefficient tensors, one row per output dim.

    The LHS x'' is a 3-point central finite difference of each trajectory; the
    library is sampled pointwise at the interior snapshots where x'' is defined.
    Xs is (P, D, M), as in weak_coefficients.
    """
    Xddot = (Xs[:, :, 2:] - 2 * Xs[:, :, 1:-1]
             + Xs[:, :, :-2]) / dt ** 2                         # (P, D, M-2)
    Theta = feature_tensor(flatten_trajectories(Xs[:, :, 1:-1]), f, phi=None,
                           low_rank=True, n_traj=Xs.shape[0])
    return np.stack([tt_pi_coeffs(Theta, Xddot[:, d, :].ravel(), threshold, D, J)
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
    """Numerical rank of the strong (pointwise) monomial library, and J^D.

    A rank below J^D means the candidate monomials are linearly dependent along
    the data, so the un-thresholded TT-PI cannot recover the sparse truth. Pass
    the flattened (D, P*M) data to see the rank pooled over all trajectories,
    which is what the regression actually sees.
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

    # ----- parameters (D=4 "balanced" regime: full rank + resolved weak form) -----
    D = 4               # oscillators
    n_traj = 6          # independent trajectories (distinct initial conditions)
    M = 10000           # snapshots per trajectory
    dt = 0.015          # Snapshot spacing
    beta = 0.7          # cubic coupling
    amplitude = 3.0     # initial-displacement half-width (drives to full rank)
    r_frac = 1.0 / 120.0 # test-fn radius as a fraction of the per-trajectory span
    degree = 16         # test-function polynomial degree
    threshold = 1e-16   # TT-PI singular-value truncation (regularized pinv)

    #noise_levels = np.array([1e-1])
    noise_levels = np.array([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 4e-1])
    n_trials = 1        # noise realizations averaged per level

    f = [lambda x: 1, lambda x: x, lambda x: x ** 2, lambda x: x ** 3]
    J = len(f)

    # ----- generate the trajectories (positions only are used in regression) -----
    t, Xs, Vs = fput_trajectories(D, M, n_traj, dt=dt, beta=beta,
                                  amplitude=amplitude, seed=3)
    t0, tM = t[0], t[-1]
    Xflat = flatten_trajectories(Xs)

    e0 = np.array([fpu_energy(Xs[p, :, 0], Vs[p, :, 0], beta)
                   for p in range(n_traj)])
    eN = np.array([fpu_energy(Xs[p, :, -1], Vs[p, :, -1], beta)
                   for p in range(n_traj)])
    Flin = np.stack([_fpu_force(Xflat[:, k], 0.0) for k in range(Xflat.shape[1])])
    Ffull = np.stack([_fpu_force(Xflat[:, k], beta) for k in range(Xflat.shape[1])])
    nl_frac = np.linalg.norm(Ffull - Flin) / np.linalg.norm(Ffull)
    rank, jD = library_rank(Xflat, f, J)

    print("=" * 66)
    print("FPUT trajectories")
    print(f"  D={D} oscillators, {n_traj} trajectories x M={M} snapshots"
          f" = {n_traj * M} total, dt={dt}, T={tM:.1f}")
    print(f"  beta={beta}, amplitude={amplitude}")
    print(f"  energy drift |dH|/H0    : {np.max(np.abs(eN - e0) / np.abs(e0)):.3e}"
          f"  (max over trajectories)")
    print(f"  displacement range      : [{Xflat.min():.3f}, {Xflat.max():.3f}]")
    print(f"  nonlinear force fraction: {nl_frac:.3f}")
    print(f"  library rank            : {rank}/{jD}"
          f"  {'(full rank)' if rank == jD else '(RANK DEFICIENT -- raise amplitude/M)'}")
    print("=" * 66)

    Wtrue = np.stack(true_coeffs(D, J, beta))

    # ----- clean-data sanity report -----
    Ww0 = weak_coefficients(Xs, f, t0, tM, D, J, r_frac=r_frac, degree=degree,
                            threshold=threshold)
    Ws0 = strong_coefficients(Xs, f, dt, D, J, threshold=threshold)
    print("clean-data relative coefficient error (no noise):")
    print(f"  TT-WSINDy (weak)  : {rel_err(Ww0, Wtrue):.3e}")
    print(f"  MANDy     (strong): {rel_err(Ws0, Wtrue):.3e}")
    print("-" * 66)
    print("oscillator 0 -- true vs. recovered coefficients (nonzero true terms):")
    print(f"  {'monomial':>12}  {'true':>9}  {'weak':>11}  {'strong':>11}")
    for idx in map(tuple, np.argwhere(np.abs(Wtrue[0]) > 1e-12)):
        print(f"  {_mono_label(idx):>12}  {Wtrue[0][idx]:>9.3f}"
              f"  {Ww0[0][idx]:>11.4f}  {Ws0[0][idx]:>11.4f}")
    print("=" * 66)

    # ----- noise sweep -----
    # errors are kept per output dimension: (n_sigma, n_trials, D)
    weak_err = np.zeros((len(noise_levels), n_trials, D))
    strong_err = np.zeros((len(noise_levels), n_trials, D))
    xfrob_normalized = np.linalg.norm(Xflat, ord='fro') / np.sqrt(Xs.size)

    print(f"noise sweep (relative coefficient error per output dim, "
          f"mean over {n_trials} trials):")
    print("  " + f"{'noise sigma':>12}  {'form':>6}"
          + "".join(f"{f'x{d}':>12}" for d in range(D)))
    sweep_st = time()
    for i, sigma in enumerate(noise_levels):
        for tr in range(n_trials):
            rng = np.random.default_rng(1000 * i + tr)
            Xn = Xs + sigma * xfrob_normalized * rng.standard_normal(Xs.shape)
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
    np.savetxt(f"results/weakvstrongformFPUT.txt", out,
               header=(f"D={D} n_traj={n_traj} M={M} dt={dt} beta={beta} "
                       f"amplitude={amplitude} r_frac={r_frac:.4f} "
                       f"n_trials={n_trials}")
                      + "\n" + cols)

    # ----- plot -----
    wm, ws = weak_err.mean(1), weak_err.std(1)        # (n_sigma, D)
    sm, ss = strong_err.mean(1), strong_err.std(1)
    floor = max(noise_levels[1] / 10, 1e-6)   # x-position for the sigma=0 point
    x_axis = np.where(noise_levels > 0, noise_levels, floor)

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
    plt.xlabel(r'relative noise level $\sigma$')
    plt.ylabel('relative coefficient error')
    plt.title(('Weak vs. strong form TT regression on FPUT '
               f'(D={D}, {n_traj} traj. x M={M})'))
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"results/weakvstrongformFPUT.png", dpi=150)
    plt.show()
