"""
Weak form (TT-WSINDy) vs. strong form (MANDy) coefficient accuracy on FPUT.

Both methods recover the coefficient tensor of the second-order
Fermi-Pasta-Ulam-Tsingou system

    x_i'' = (x_{i+1} - 2 x_i + x_{i-1})
            + beta [ (x_{i+1} - x_i)^3 - (x_i - x_{i-1})^3 ],   x_{-1} = x_n = 0

over the separable candidate library f = {1, x, x^2, x^3} on every oscillator,
using a SINGLE tensor-train pseudoinverse (TT-PI) regression per output
dimension. There is NO sequential thresholding (no MSTLS): this isolates the
regression step so the two forms are compared on equal footing. The only
regularization is the SVD truncation built into TT-PI.

  * Strong form (MANDy): the left-hand side x'' is estimated directly from the
    trajectory with a 3-point central finite difference, and the library is
    sampled pointwise. Solve  Theta(x) W = x''.
  * Weak form (TT-WSINDy): the left-hand side is the weak projection
    <x, phi''> against a compactly supported test function phi (two
    integrations by parts move both derivatives onto phi), and the library is
    convolved with phi. No derivative of the data is ever taken.
    Solve  <Theta(x), phi> W = <x, phi''>.

Only a single trajectory X is generated, by a symplectic (velocity-Verlet)
forward solve of FPUT. The strong form's derivative data is obtained purely by
finite differencing that trajectory -- which is exactly what makes it sensitive
to measurement noise. Sweeping additive noise on X shows the weak form holding
its accuracy while the finite-difference strong form degrades.

Identifiability note
--------------------
Pure TT-PI (no sparsity) returns the minimum-norm least-squares solution, which
equals the sparse physical truth only when the J^D candidate monomials are
linearly independent along the trajectory. FPUT is famously quasi-periodic, so a
low-energy chain leaves most monomials un-excited and the library is rank
deficient. The default below drives the chain to high energy (large amplitude),
which makes the library full rank so a single trajectory suffices. The script
prints the realized rank; if it is below J^D, raise `amplitude` or `M`.

Run from the experiments/ directory:  python weakvstrongform.py
"""
import os, sys, itertools
sys.path.insert(0, '.')
sys.path.insert(0, '../TT-WSINDy')
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.signal import correlate
from feature_tensor import feature_tensor
from test_function import piecewise_polynomial


# ---------------------------------------------------------------------------
# FPUT trajectory generator (symplectic velocity-Verlet, fixed walls)
# ---------------------------------------------------------------------------
def _fpu_force(x, beta):
    """Acceleration field F(x) with fixed walls x_{-1} = x_n = 0."""
    xp = np.concatenate(([0.0], x, [0.0]))
    dr = xp[2:] - xp[1:-1]          # x_{i+1} - x_i
    dl = xp[1:-1] - xp[:-2]         # x_i - x_{i-1}
    return (dr - dl) + beta * (dr ** 3 - dl ** 3)


def fpu_energy(x, v, beta):
    """Total Hamiltonian H = 1/2 |v|^2 + U(x) of one configuration."""
    xp = np.concatenate(([0.0], x, [0.0]))
    d = xp[1:] - xp[:-1]
    U = np.sum(0.5 * d ** 2 + (beta / 4.0) * d ** 4)
    return 0.5 * np.sum(v ** 2) + U


def fput_trajectory(n, m, dt=0.015, beta=0.7, amplitude=3.0,
                    substeps=60, seed=0):
    """Generate one uniformly-sampled FPUT trajectory.

    Parameters
    ----------
    n : int
        Number of oscillators.
    m : int
        Number of recorded snapshots (spaced by dt).
    dt : float
        Spacing between recorded snapshots.
    beta : float
        Cubic coupling strength.
    amplitude : float
        Half-width of the uniform random initial displacement box. Larger
        amplitude raises the energy, exciting more monomials so the candidate
        library becomes full rank (see the identifiability note in the module
        docstring). Released from rest (v0 = 0).
    substeps : int
        Verlet substeps per recorded snapshot (internal step = dt / substeps),
        increased to keep the symplectic integration accurate while sampling
        coarsely.
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
    x = 2.0 * amplitude * rng.random(n) - amplitude
    v = np.zeros(n)

    positions = np.zeros((n, m))
    velocities = np.zeros((n, m))
    positions[:, 0] = x
    velocities[:, 0] = v

    h = dt / substeps
    F = _fpu_force(x, beta)
    for k in range(1, m):
        for _ in range(substeps):
            v = v + 0.5 * h * F      # half kick
            x = x + h * v            # drift
            F = _fpu_force(x, beta)  # force at new position
            v = v + 0.5 * h * F      # half kick
        positions[:, k] = x
        velocities[:, k] = v

    t = np.arange(m) * dt
    return t, positions, velocities


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
def _norm_product(Theta, D, J):
    """Per-monomial product of the per-feature normalization factors.

    feature_tensor scales feature f_j in dimension d by its column norm, so a
    monomial column is scaled by the product of those norms. Dividing the raw
    TT-PI estimate by this product returns coefficients in the original units.
    (feature_tensor.unscale multiplies by these norms, which recovers the wrong
    magnitude; we divide here instead.)
    """
    norms = Theta.feature_norms                  # (J, D)
    s = np.ones((J,) * D)
    for idx in np.ndindex(*([J] * D)):
        p = 1.0
        for d in range(D):
            p *= norms[idx[d], d]
        s[idx] = p
    return s


def tt_pi_coeffs(Theta, y, threshold, D, J):
    """One TT-PI solve, returned as a dense (J,)*D tensor in original units.

    TT_PI uses overwrite=False, so Theta is not mutated and no copy is needed.
    The raw estimate lives in the normalized feature space; dividing by the
    per-monomial norm product undoes feature_tensor's column scaling.
    """
    W = Theta.TT_PI(y, threshold=threshold)
    raw = np.asarray(W.full()).reshape((J,) * D)
    return raw / _norm_product(Theta, D, J)


def weak_coefficients(X, f, t0, tM, M, D, J,
                      r_frac=1.0 / 60.0, degree=16, threshold=1e-10):
    """TT-WSINDy (weak form) coefficient tensors, one row per output dim.

    The LHS is the weak projection <x, phi''> (test-function order 2, so phi''
    carries both derivatives); the library is convolved with phi. No derivative
    of the data is computed.
    """
    phi, dphi = piecewise_polynomial((tM - t0) * r_frac, degree, t0, tM, M,
                                     order=2)
    Theta = feature_tensor(X, f, phi=phi)
    # the order-2 dphi equals -phi'', so -correlate(X, dphi) = <x, phi''>
    Y = -1 * correlate(X, np.expand_dims(dphi, axis=0), mode='valid').transpose()
    return np.stack([tt_pi_coeffs(Theta, Y[:, d], threshold, D, J)
                     for d in range(D)])


def strong_coefficients(X, f, dt, D, J, threshold=1e-10):
    """MANDy (strong form) coefficient tensors, one row per output dim.

    The LHS x'' is a 3-point central finite difference of the trajectory; the
    library is sampled pointwise at the interior snapshots where x'' is defined.
    """
    Xddot = (X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]) / dt ** 2   # (D, M-2)
    Theta = feature_tensor(X[:, 1:-1], f, phi=None)             # strong form
    return np.stack([tt_pi_coeffs(Theta, Xddot[d], threshold, D, J)
                     for d in range(D)])


def rel_err(W, Wtrue):
    """Relative 2-norm error over the full stacked coefficient tensor."""
    return np.linalg.norm(W - Wtrue) / np.linalg.norm(Wtrue)


def library_rank(X, f, J, tol=1e-10):
    """Numerical rank of the strong (pointwise) monomial library, and J^D.

    A rank below J^D means the candidate monomials are linearly dependent along
    the trajectory, so the un-thresholded TT-PI cannot recover the sparse truth.
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
    M = 2000            # snapshots (long enough that the library is full rank)
    dt = 0.015          # snapshot spacing
    beta = 0.7          # cubic coupling
    amplitude = 3.0     # initial-displacement half-width (drives to full rank)
    substeps = 60       # Verlet substeps per snapshot
    r_frac = 1.0 / 60.0 # test-function radius as a fraction of the time span
    degree = 16         # test-function polynomial degree
    threshold = 1e-10   # TT-PI singular-value truncation (regularized pinv)

    noise_levels = np.array([0.0, 1e-5, 1e-4, 3e-4, 1e-3, 1e-2])
    n_trials = 3        # noise realizations averaged per level

    f = [lambda x: 1, lambda x: x, lambda x: x ** 2, lambda x: x ** 3]
    J = len(f)

    # ----- generate ONE trajectory (positions only) -----
    t, X, V = fput_trajectory(D, M, dt=dt, beta=beta, amplitude=amplitude,
                              substeps=substeps, seed=2)
    t0, tM = t[0], t[-1]

    e0 = fpu_energy(X[:, 0], V[:, 0], beta)
    eN = fpu_energy(X[:, -1], V[:, -1], beta)
    Flin = np.stack([_fpu_force(X[:, k], 0.0) for k in range(M)])
    Ffull = np.stack([_fpu_force(X[:, k], beta) for k in range(M)])
    nl_frac = np.linalg.norm(Ffull - Flin) / np.linalg.norm(Ffull)
    rank, jD = library_rank(X, f, J)

    print("=" * 66)
    print("FPUT trajectory")
    print(f"  D={D} oscillators, M={M} snapshots, dt={dt}, T={tM:.1f}")
    print(f"  beta={beta}, amplitude={amplitude}, substeps={substeps}")
    print(f"  energy drift |dH|/H0    : {abs(eN - e0) / abs(e0):.3e}")
    print(f"  displacement range      : [{X.min():.3f}, {X.max():.3f}]")
    print(f"  nonlinear force fraction: {nl_frac:.3f}")
    print(f"  library rank            : {rank}/{jD}"
          f"  {'(full rank)' if rank == jD else '(RANK DEFICIENT -- raise amplitude/M)'}")
    print("=" * 66)

    Wtrue = np.stack(true_coeffs(D, J, beta))

    # ----- clean-data sanity report -----
    Ww0 = weak_coefficients(X, f, t0, tM, M, D, J,
                            r_frac=r_frac, degree=degree, threshold=threshold)
    Ws0 = strong_coefficients(X, f, dt, D, J, threshold=threshold)
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
    weak_err = np.zeros((len(noise_levels), n_trials))
    strong_err = np.zeros((len(noise_levels), n_trials))
    xstd = X.std()

    print(f"noise sweep (relative coefficient error, mean over {n_trials} trials):")
    print(f"  {'noise sigma':>12}  {'TT-WSINDy':>12}  {'MANDy':>12}  {'ratio S/W':>10}")
    sweep_st = time()
    for i, sigma in enumerate(noise_levels):
        for tr in range(n_trials):
            rng = np.random.default_rng(1000 * i + tr)
            Xn = X + sigma * xstd * rng.standard_normal(X.shape)
            Ww = weak_coefficients(Xn, f, t0, tM, M, D, J,
                                   r_frac=r_frac, degree=degree, threshold=threshold)
            Ws = strong_coefficients(Xn, f, dt, D, J, threshold=threshold)
            weak_err[i, tr] = rel_err(Ww, Wtrue)
            strong_err[i, tr] = rel_err(Ws, Wtrue)
        wm, sm = weak_err[i].mean(), strong_err[i].mean()
        print(f"  {sigma:>12.0e}  {wm:>12.3e}  {sm:>12.3e}  {sm / wm:>10.1f}")
    print(f"(noise sweep walltime: {time() - sweep_st:.1f}s)")
    print("=" * 66)

    # ----- save results -----
    os.makedirs("results", exist_ok=True)
    out = np.column_stack([noise_levels,
                           weak_err.mean(1), weak_err.std(1),
                           strong_err.mean(1), strong_err.std(1)])
    np.savetxt("results/weakvstrongform.txt", out,
               header="noise weak_mean weak_std strong_mean strong_std")

    # ----- plot -----
    wm, ws = weak_err.mean(1), weak_err.std(1)
    sm, ss = strong_err.mean(1), strong_err.std(1)
    floor = max(noise_levels[1] / 10, 1e-6)   # x-position for the sigma=0 point
    x_axis = np.where(noise_levels > 0, noise_levels, floor)

    plt.figure(figsize=(7, 5))
    plt.errorbar(x_axis, wm, yerr=ws, marker='o', capsize=3,
                 label='TT-WSINDy (weak form)')
    plt.errorbar(x_axis, sm, yerr=ss, marker='s', capsize=3,
                 label='MANDy (strong form, finite diff.)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'relative noise level $\sigma$  (leftmost point = clean data)')
    plt.ylabel('relative coefficient error')
    plt.title(f'Weak vs. strong form TT-PI on FPUT (D={D}, M={M})')
    plt.legend()
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/weakvstrongform.png", dpi=150)
    plt.show()
