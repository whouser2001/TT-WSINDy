"""
Weak form (TT-WSINDy) vs. strong form (MANDy) coefficient accuracy on Chua's circuit.
"""
import os, sys
sys.path.insert(0, '.')
sys.path.insert(0, '../TT-WSINDy')
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.signal import correlate
from scipy.integrate import odeint
from feature_tensor import feature_tensor
from test_function import piecewise_polynomial
import exputils as xu
from functools import partial

alpha = 10
beta = 14.87
delta = (-8/7, 4/63)

RTOL = ATOL = 1e-12

def chua(x,t):
    """Chua's circuit."""
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

    Reads the module-level alpha, beta and delta, as chua() does.

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
        Ws[i][a, b] the coefficient of x_a |x_b|. Index D means that function
        is absent.

    Raises
    ------
    ValueError
        If D != 3, or if J != 3.
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
        [(-alpha * (1.0 + d1), 0, A),
         (alpha,               1, A),
         (-alpha * d2,         0, 0)],
        [(1.0, 0, A), (-1.0, 1, A), (1.0, 2, A)],
        [(-beta, 1, A)],
    ]

    Ws = []
    for terms in equations:
        W = np.zeros(_tensor_shape(D, J))
        for coeff, x_dim, abs_dim in terms:
            W[x_dim, abs_dim] += coeff
        Ws.append(W)
    return Ws

def weak_coefficients(X, f, t0, tM, D, J,
                      r_frac=1.0 / 60.0, degree=16):
    """TT-WSINDy (weak form) coefficient tensors, one row per output dim.

    X is (D, M): one trajectory of M snapshots spanning [t0, tM], with the
    test-function radius a fraction r_frac of that span. Chua is first order,
    so the test function is taken at order 1.
    """
    M = X.shape[1]
    phi, dphi = piecewise_polynomial((tM - t0) * r_frac, degree, t0, tM, M,
                                     order=1)
    Theta = feature_tensor(X, f, phi=phi, low_rank=True,
                           construction='function_major')
    Y = -1 * correlate(X, np.expand_dims(dphi, axis=0),
                       mode='valid').transpose()                # (Mp, D)
    return np.stack([xu.tt_pi_coeffs(Theta, Y[:, d], _tensor_shape(D, J))
                     for d in range(D)])

def strong_coefficients(X, f, dt, D, J):
    """MANDy (strong form) coefficient tensors, one row per output dim.

    The LHS x' is a 3-point central finite difference; the library is sampled
    pointwise at the interior snapshots. X is (D, M), as in weak_coefficients.
    """
    Xdot = (X[:, 2:] - X[:, :-2]) / (2 * dt)                # (D, M-2)
    Theta = feature_tensor(X[:, 1:-1], f, phi=None, low_rank=True,
                           construction='function_major')
    return np.stack([xu.tt_pi_coeffs(Theta, Xdot[d], _tensor_shape(D, J))
                     for d in range(D)])

# Experiment
if __name__ == '__main__':

    # ----- parameters -----
    D = 3               # state dimension, fixed by the model
    M = 200000            # snapshots
    dt = 0.01           # snapshot spacing
    r_frac = 1.0 / 2000.0 # test-fn radius as a fraction of the time span
    degree = 16         # test-function polynomial degree

    noise_levels = np.array([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 4e-1])
    n_trials = 40      # noise realizations averaged per level

    f = [lambda x: 1, lambda x: x, lambda x: np.abs(x)]
    J = len(f)
    # function-major library: one label per non-constant function; the label
    # function needs D to know which slot means "absent"
    LABELS = ['x{}', '|x{}|']
    xu.check_labels(f, LABELS, function_major=True)
    label_fn = partial(xu.function_major_label, D=D)

    t = np.arange(M) * dt          # exact spacing dt
    t0, tM = t[0], t[-1]           # the plot title needs tM on both paths

    DATA = "results/weakvstrongformChua.txt"
    recompute_data = True   # False skips the sweep and just
                            # replots what DATA already holds

    if recompute_data:

        x0 = np.array([-1.13, 0.004, 0.45])
        X = gen_chua(x0, t)

        rank, ncand = xu.library_rank_function_major(X, f, D, J)
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
        Ww0 = weak_coefficients(X, f, t0, tM, D, J, r_frac=r_frac, degree=degree)
        Ws0 = strong_coefficients(X, f, dt, D, J)
        print("clean-data relative coefficient error (no noise):")
        print(f"  TT-WSINDy (weak)  : {xu.rel_err(Ww0, Wtrue):.3e}")
        print(f"  MANDy     (strong): {xu.rel_err(Ws0, Wtrue):.3e}")
        xu.rule('-')
        xu.print_coeff_table(Wtrue, Ww0, Ws0, LABELS, eqs=range(D),
                             label_fn=label_fn)
        xu.rule()

        # ----- noise sweep -----
        weak_err = np.zeros((noise_levels.size, n_trials))
        strong_err = np.zeros((noise_levels.size, n_trials))
        xfrob_normalized = np.linalg.norm(X, ord='fro') / np.sqrt(X.size)

        print(f"noise sweep (relative coefficient error, mean over {n_trials} trials):")
        print(f"  {'noise sigma':>12}  {'TT-WSINDy':>12}  {'MANDy':>12}  {'ratio S/W':>10}")
        sweep_st = time()
        for i, sigma in enumerate(noise_levels):
            for tr in range(n_trials):
                rng = np.random.default_rng(1000 * i + tr)
                Xn = X + sigma * xfrob_normalized * rng.standard_normal(X.shape)
                Ww = weak_coefficients(Xn, f, t0, tM, D, J, r_frac=r_frac,
                                       degree=degree)
                Ws = strong_coefficients(Xn, f, dt, D, J)
                weak_err[i, tr] = xu.rel_err(Ww, Wtrue)
                strong_err[i, tr] = xu.rel_err(Ws, Wtrue)
            wm, sm = weak_err[i].mean(), strong_err[i].mean()
            print(f"  {sigma:>12.0e}  {wm:>12.3e}  {sm:>12.3e}  {sm / wm:>10.1f}")
        print(f"(noise sweep walltime: {time() - sweep_st:.1f}s)")
        print("=" * 66)

        # ----- save results -----
        os.makedirs("results", exist_ok=True)
        out = np.column_stack([noise_levels,
                               weak_err.mean(1), weak_err.std(1),
                               strong_err.mean(1), strong_err.std(1)])
        np.savetxt(DATA, out,
                   header=f"D={D} M={M} dt={dt} alpha={alpha} beta={beta} "
                          f"delta=({delta[0]:.6f},{delta[1]:.6f}) "
                          f"r_frac={r_frac:.5f} n_trials={n_trials}\n"
                          "noise weak_mean weak_std strong_mean strong_std")

    # the figure is always drawn from DATA, so a rerun and a replot agree
    if not os.path.exists(DATA):
        raise SystemExit(f"{DATA} not found -- set recompute_data = True and rerun")
    noise_levels, wm, ws, sm, ss = np.atleast_2d(np.loadtxt(DATA)).T

    # ----- plot -----

    # mean +- one standard deviation over trials
    ax = plt.figure(figsize=(7, 5)).gca()
    hw = ax.errorbar(noise_levels, wm, yerr=ws, marker='o', capsize=3,
                     color='C0', ls='-', label='TT-WSINDy (weak form)')
    hs = ax.errorbar(noise_levels, sm, yerr=ss, marker='s', capsize=3,
                     color='C1', ls='-',
                     label='MANDy (strong form, finite diff.)')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-6, 1e0)
    plt.xlabel(r'relative noise level $\sigma$')
    plt.ylabel('relative coefficient error')
    plt.title("Weak vs. strong form TT regression on Chua's circuit "
              f'(M={M}, T={tM:.0f})')
    plt.legend(handles=[hw, hs])
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"results/weakvstrongformChua.png", dpi=150)
    plt.show()
