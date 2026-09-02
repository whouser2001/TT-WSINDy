"""
Weak form (TT-WSINDy) vs. strong form (MANDy) coefficient accuracy on Chua.

Chua's circuit is first order, so the test function carries one derivative
(order 1) and the strong form uses a central first difference. Its nonlinearity
g(z) = d_1 z + d_2 z|z| puts two candidate functions on the SAME state
variable, which the default dimension-major library cannot express, so the
feature tensor is built with construction='function_major' -- see the note
above true_coeffs.
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

alpha = 10
beta = 14.87
delta = (-8/7, 4/63)

# Tight tolerances so the integrator is never what limits MANDy: the point of
# the comparison is the O(dt^2) error of the finite-difference stencil.
RTOL = ATOL = 1e-12

def chua(x,t):
    """ Chua's ciruit with g(z) = d_1z + d_2z|z|"""
    return [
        alpha*(x[1] - x[0] - delta[0]*x[0] - delta[1]*x[0]*np.abs(x[0])),
        x[0] - x[1] + x[2],
        -beta*x[1]
    ]

def gen_chua(x0, t):
    """Integrate Chua's circuit to a D x M data matrix."""
    return odeint(chua, x0, t, rtol=RTOL, atol=ATOL).T

def _tensor_shape(D, J):
    """Shape of one equation's coefficient tensor, function-major construction."""
    return (D + 1,) * (J - 1)

def true_coeffs(D, J):
    """Exact coefficient matrix of Chua's right-hand side, per output dim.

    Reads the module-level alpha, beta, delta, exactly as chua() does; Chua
    has no forcing parameter.

    Parameters
    ----------
    D : int
        Number of state variables. Must be 3 -- Chua's circuit is 3-D.
    J : int
        Number of candidate functions, assumed to be f = {1, x, |x|}; J == 3.

    Returns
    -------
    Ws : list of ndarray
        Ws[i] is the (D+1) x (D+1) coefficient matrix of x_i's equation, with
        Ws[i][a, b] the coefficient of x_a |x_b|. Index D means that function is
        absent, so Ws[i][a, D] multiplies a bare x_a, Ws[i][D, b] a bare |x_b|,
        and Ws[i][D, D] the constant. Seven entries are nonzero in total.

    Raises
    ------
    ValueError
        If D != 3, or if J != 3 (this is written for f = {1, x, |x|}).
    """
    if D != 3:
        raise ValueError(f"Chua's circuit is 3-dimensional, so D must be 3 (got D={D})")
    if J != 3:
        raise ValueError(
            f"true_coeffs assumes f = {{1, x, |x|}}, so J == 3 (got J={J})"
        )

    d1, d2 = delta
    A = D                      # the "function is absent" slot
    # (coefficient, dimension for x, dimension for |x|) per term, per equation
    equations = [
        [(-alpha * (1.0 + d1), 0, A),   # -alpha (1 + d_1) x_1
         (alpha,               1, A),   # +alpha x_2
         (-alpha * d2,         0, 0)],  # -alpha d_2 x_1|x_1|
        [(1.0, 0, A), (-1.0, 1, A), (1.0, 2, A)],           # x_1 - x_2 + x_3
        [(-beta, 1, A)],                                    # -beta x_2
    ]

    Ws = []
    for terms in equations:
        W = np.zeros(_tensor_shape(D, J))
        for coeff, x_dim, abs_dim in terms:
            W[x_dim, abs_dim] += coeff
        Ws.append(W)
    return Ws

def tt_pi_coeffs(Theta, y, threshold, D, J):
    """One TT-PI solve, returned as a dense coefficient tensor in original units.

    The shape is the function-major one, (D+1)^(J-1) -- a (D+1) x (D+1) matrix
    for Chua -- matching true_coeffs.

    TT_PI uses overwrite=False, so Theta is not mutated and no copy is needed.
    """
    W = Theta.TT_PI(y)
    return np.asarray(W.full()).reshape(_tensor_shape(D, J))

def weak_coefficients(X, f, t0, tM, D, J,
                      r_frac=1.0 / 60.0, degree=16, threshold=1e-10):
    """TT-WSINDy (weak form) coefficient tensors, one row per output dim.

    Chua is first order, so one integration by parts moves the single derivative
    onto the test function: the LHS is -<x, phi'> = <x', phi> (test-function
    order 1, so dphi is phi'), and the library is convolved with phi. No
    derivative of the data is computed.

    X is (D, M): one trajectory of M snapshots spanning [t0, tM], with the
    test-function radius a fraction r_frac of that span.
    """
    M = X.shape[1]
    phi, dphi = piecewise_polynomial((tM - t0) * r_frac, degree, t0, tM, M,
                                     order=1)
    Theta = feature_tensor(X, f, phi=phi, low_rank=True,
                           construction='function_major')
    # the order-1 dphi equals phi', so -correlate(x, dphi) = -<x, phi'>
    Y = -1 * correlate(X, np.expand_dims(dphi, axis=0),
                       mode='valid').transpose()                # (Mp, D)
    return np.stack([tt_pi_coeffs(Theta, Y[:, d], threshold, D, J)
                     for d in range(D)])

def strong_coefficients(X, f, dt, D, J, threshold=1e-10):
    """MANDy (strong form) coefficient tensors, one row per output dim.

    The LHS x' is a 3-point central finite difference of the trajectory; the
    library is sampled pointwise at the interior snapshots where x' is defined.
    X is (D, M), as in weak_coefficients.
    """
    Xdot = (X[:, 2:] - X[:, :-2]) / (2 * dt)                # (D, M-2)
    Theta = feature_tensor(X[:, 1:-1], f, phi=None, low_rank=True,
                           construction='function_major')
    return np.stack([tt_pi_coeffs(Theta, Xdot[d], threshold, D, J)
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

def _mono_label(idx, D=3):
    """Readable label for a function-major index, e.g. (0, 0) -> 'x0 |x0|'.

    idx is (dimension for x, dimension for |x|); D means that factor is absent.
    """
    a, b = idx
    parts = ([f"x{a}"] if a < D else []) + ([f"|x{b}|"] if b < D else [])
    return "1" if not parts else " ".join(parts)

def library_rank(X, f, D, J, tol=1e-10):
    """Numerical rank of the function-major candidate library, and its size.

    Enumerates the (D+1)^(J-1) candidates -- one placement per non-constant
    function, index D meaning absent -- and returns the numerical rank of the
    pointwise library alongside the candidate count. A rank below that count
    means the candidates are linearly dependent along the data, so the
    un-thresholded TT-PI cannot recover the sparse truth.
    """
    M = X.shape[1]
    cols = []
    for picks in itertools.product(range(D + 1), repeat=J - 1):
        c = np.ones(M)
        for j, d in enumerate(picks):
            if d < D:
                c = c * np.vectorize(f[j + 1])(X[d]).astype(float)
        cols.append(c)
    s = np.linalg.svd(np.stack(cols), compute_uv=False)
    return int(np.sum(s / s[0] > tol)), len(cols)

# -----
# main
# -----
if __name__ == '__main__':

    # ----- parameters (the setup of Example 2.1: h = 0.01, t = 0, ..., 20) -----
    D = 3               # state dimension, fixed by the model
    M = 200000            # snapshots
    dt = 0.01           # snapshot spacing
    r_frac = 1.0 / 2000.0 # test-fn radius as a fraction of the time span
    degree = 16         # test-function polynomial degree
    threshold = 1e-16   # TT-PI singular-value truncation (regularized pinv)

    noise_levels = np.array([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 4e-1])
    n_trials = 5       # noise realizations averaged per level

    f = [lambda x: 1, lambda x: x, lambda x: np.abs(x)]
    J = len(f)

    t = np.arange(M) * dt          # exact spacing dt (linspace(0, M*dt, M) is not)
    t0, tM = t[0], t[-1]
    x0 = np.array([-1.13, 0.004, 0.45])
    X = gen_chua(x0, t)

    rank, ncand = library_rank(X, f, D, J)
    sign_changes = int(np.sum(np.diff(np.sign(X[0])) != 0))

    print("=" * 66)
    print("Chua's circuit")
    print(f"  D={D}, M={M} snapshots, dt={dt}, T={tM:.1f}")
    print(f"  alpha={alpha}, beta={beta}, "
          f"delta=({delta[0]:+.4f}, {delta[1]:+.4f})")
    print("  x0=[" + ", ".join(f"{v:g}" for v in x0) + "]")
    print("  state ranges            : " + ",  ".join(
        f"x{i} [{X[i].min():+.2f}, {X[i].max():+.2f}]" for i in range(D)))
    print(f"  x0 sign changes         : {sign_changes}"
          f"   (the |x| kink is only sampled where x0 crosses 0)")
    print(f"  library rank            : {rank}/{ncand}"
          f"  {'(full rank)' if rank == ncand else '(RANK DEFICIENT)'}")
    print("=" * 66)

    # ----- clean-data check -----
    Wtrue = np.stack(true_coeffs(D, J))
    Ww0 = weak_coefficients(X, f, t0, tM, D, J, r_frac=r_frac, degree=degree,
                            threshold=threshold)
    Ws0 = strong_coefficients(X, f, dt, D, J, threshold=threshold)
    print("clean-data relative coefficient error (no noise):")
    print(f"  TT-WSINDy (weak)  : {rel_err(Ww0, Wtrue):.3e}")
    print(f"  MANDy     (strong): {rel_err(Ws0, Wtrue):.3e}")
    print("-" * 66)
    print("true vs. recovered coefficients (nonzero true terms, all equations):")
    print(f"  {'eq':>4}  {'candidate':>10}  {'true':>9}  {'weak':>11}  {'strong':>11}")
    for d in range(D):
        for idx in map(tuple, np.argwhere(np.abs(Wtrue[d]) > 1e-12)):
            print(f"  x{d}'   {_mono_label(idx, D):>10}  {Wtrue[d][idx]:>9.4f}"
                  f"  {Ww0[d][idx]:>11.4f}  {Ws0[d][idx]:>11.4f}")
    print("=" * 66)

    # ----- noise sweep -----
    # errors are kept per output dimension: (n_sigma, n_trials, D)
    weak_err = np.zeros((len(noise_levels), n_trials, D))
    strong_err = np.zeros((len(noise_levels), n_trials, D))
    xfrob_normalized = np.linalg.norm(X, ord='fro') / np.sqrt(X.size)

    print(f"noise sweep (relative coefficient error per output dim, "
          f"mean over {n_trials} trials):")
    print("  " + f"{'noise sigma':>12}  {'form':>6}"
          + "".join(f"{f'x{d}':>12}" for d in range(D)))
    sweep_st = time()
    for i, sigma in enumerate(noise_levels):
        for tr in range(n_trials):
            rng = np.random.default_rng(1000 * i + tr)
            Xn = X + sigma * xfrob_normalized * rng.standard_normal(X.shape)
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
    np.savetxt(f"results/weakvstrongformChua.txt", out,
               header=(f"D={D} M={M} dt={dt} alpha={alpha} beta={beta} "
                       f"delta=({delta[0]:.6f},{delta[1]:.6f}) "
                       f"r_frac={r_frac:.5f} n_trials={n_trials}")
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
    plt.xlabel(r'relative noise level $\sigma$')
    plt.ylabel('relative coefficient error')
    plt.title(("Weak vs. strong form TT regression on Chua's circuit "
               f'(M={M}, T={tM:.0f})'))
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"results/weakvstrongformChua.png", dpi=150)
    plt.show()
