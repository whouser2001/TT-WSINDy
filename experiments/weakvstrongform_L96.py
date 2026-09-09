"""
Weak form (TT-WSINDy) vs. strong form (MANDy) coefficient accuracy on L96.
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
import errorplot

F = 8
def L96(x, t):
    """Lorenz 96 model with constant forcing"""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F 
def gen_L96(x0,t):
    """Generate L96 data"""
    return odeint(L96, x0, t).T

# True coeffs
def true_coeffs(D, J, F):
    """Exact coefficient tensor of the L96 right-hand side, per output dim.
    """
    Ws = []
    for d in range(D):
        # (coefficient, variables multiplied together) for each monomial
        terms = [(1.0, ((d + 1) % D, (d - 1) % D)),     # +x_{d+1} x_{d-1}
                 (-1.0, ((d - 2) % D, (d - 1) % D)),    # -x_{d-2} x_{d-1}
                 (-1.0, (d,)),                          # -x_d
                 (float(F), ())]                        # +F

        W = np.zeros((J,) * D)
        for coeff, variables in terms:
            # a repeated index raises that variable's power (D <= 2 aliasing);
            # two terms landing on the same monomial add, and for D = 3 the two
            # quadratic terms alias onto each other and cancel exactly
            powers = [0] * D
            for v in variables:
                powers[v] += 1
            if max(powers) >= J:
                raise ValueError(
                    f"D={D} aliases a monomial to power {max(powers)}, which "
                    f"needs J >= {max(powers) + 1} candidate functions (got J={J})"
                )
            W[tuple(powers)] += coeff
        Ws.append(W)
    return Ws

def tt_pi_coeffs(Theta, y, threshold, D, J):
    """One TT-PI solve, returned as a dense (J,)*D tensor in original units.

    TT_PI uses overwrite=False, so Theta is not mutated and no copy is needed.
    """
    W = Theta.TT_PI(y)
    return np.asarray(W.full()).reshape((J,) * D)

def weak_coefficients(X, f, t0, tM, D, J,
                      r_frac=1.0 / 60.0, degree=16, threshold=1e-10):
    """TT-WSINDy (weak form) coefficient tensors, one row per output dim.

    L96 is first order, so one integration by parts moves the single derivative
    onto the test function: the LHS is -<x, phi'> = <x', phi> (test-function
    order 1, so dphi is phi'), and the library is convolved with phi. No
    derivative of the data is computed.

    X is (D, M): one trajectory of M snapshots spanning [t0, tM], with the
    test-function radius a fraction r_frac of that span.
    """
    M = X.shape[1]
    phi, dphi = piecewise_polynomial((tM - t0) * r_frac, degree, t0, tM, M,
                                     order=1)
    Theta = feature_tensor(X, f, phi=phi, low_rank=True)
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
    Theta = feature_tensor(X[:, 1:-1], f, phi=None, low_rank=True)
    return np.stack([tt_pi_coeffs(Theta, Xdot[d], threshold, D, J)
                     for d in range(D)])

def rel_err(W, Wtrue):
    """Relative 2-norm error over the full stacked coefficient tensor."""
    return np.linalg.norm(W - Wtrue) / np.linalg.norm(Wtrue)

def _mono_label(idx):
    """Readable label for a monomial multi-index, e.g. (1,2,0,0) -> 'x0 x1^2'."""
    parts = []
    for d, p in enumerate(idx):
        if p == 1:
            parts.append(f"x{d}")
        elif p > 1:
            parts.append(f"x{d}^{p}")
    return "1" if not parts else " ".join(parts)

# -----
# main
# -----
if __name__ == '__main__':

    D = 5
    M = 10000
    dt = 0.015
    r_frac = 1.0 / 120.0
    degree = 16
    threshold = 1e-16

    noise_levels = np.array([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 4e-1])
    n_trials = 40

    f = [lambda x : 1, lambda x : x]
    J = len(f)

    t = np.arange(M) * dt          # exact spacing dt (linspace(0, M*dt, M) is not)
    t0, tM = t[0], t[-1]
    x0 = F * np.ones(D)            # equilibrium: L96(F*ones) == 0 exactly,
    x0[0] += 0.01                  # so perturb to leave it and reach the attractor
    X = gen_L96(x0, t)

    # ----- clean-data check -----
    Wtrue = np.stack(true_coeffs(D, J, F))
    Ww0 = weak_coefficients(X, f, t0, tM, D, J, r_frac=r_frac, degree=degree,
                                threshold=threshold)
    Ws0 = strong_coefficients(X, f, dt, D, J, threshold=threshold)
    print("clean-data relative coefficient error (no noise):")
    print(f"  TT-WSINDy (weak)  : {rel_err(Ww0, Wtrue):.3e}")
    print(f"  MANDy     (strong): {rel_err(Ws0, Wtrue):.3e}")
    print("-" * 66)
    print("x_0 -- true vs. recovered coefficients (nonzero true terms):")
    print(f"  {'monomial':>12}  {'true':>9}  {'weak':>11}  {'strong':>11}")
    for idx in map(tuple, np.argwhere(np.abs(Wtrue[0]) > 1e-12)):
        print(f"  {_mono_label(idx):>12}  {Wtrue[0][idx]:>9.3f}"
                f"  {Ww0[0][idx]:>11.4f}  {Ws0[0][idx]:>11.4f}")
    print("=" * 66)

    # ----- noise sweep -----
    trials_path = "results/weakvstrongformL96_trials.txt"
    # reuse the realizations a previous run already wrote (--fresh to redo)
    weak_err, strong_err, done = errorplot.resume_trials(
        trials_path, noise_levels, n_trials,
        fresh=errorplot.fresh_from_argv())
    xfrob_normalized = np.linalg.norm(X, ord='fro') / np.sqrt(X.size)

    print(f"noise sweep (relative coefficient error, mean over {n_trials} trials):")
    print(f"  {'noise sigma':>12}  {'TT-WSINDy':>12}  {'MANDy':>12}  {'ratio S/W':>10}")
    sweep_st = time()
    for i, sigma in enumerate(noise_levels):
        for tr in range(done[i], n_trials):
            rng = np.random.default_rng(1000 * i + tr)
            Xn = X + sigma * xfrob_normalized * rng.standard_normal(X.shape)
            Ww = weak_coefficients(Xn, f, t0, tM, D, J, r_frac=r_frac,
                                    degree=degree, threshold=threshold)
            Ws = strong_coefficients(Xn, f, dt, D, J, threshold=threshold)
            weak_err[i, tr] = rel_err(Ww, Wtrue)
            strong_err[i, tr] = rel_err(Ws, Wtrue)
        # checkpoint the level just finished, so an interrupted sweep
        # resumes from here rather than from the last full run
        errorplot.save_trials(trials_path, noise_levels, weak_err,
                              strong_err)
        wm, sm = weak_err[i].mean(), strong_err[i].mean()
        print(f"  {sigma:>12.0e}  {wm:>12.3e}  {sm:>12.3e}  {sm / wm:>10.1f}")
    print(f"(noise sweep walltime: {time() - sweep_st:.1f}s)")
    print("=" * 66)

    # ----- save results -----
    os.makedirs("results", exist_ok=True)
    out = np.column_stack([noise_levels,
                            np.nanmean(weak_err, 1), np.nanstd(weak_err, 1),
                            np.nanmean(strong_err, 1), np.nanstd(strong_err, 1)])
    np.savetxt(f"results/weakvstrongformL96.txt", out,
                header=f"D={D} M={M} dt={dt} F={F} "
                        f" r_frac={r_frac:.4f}\n"
                        "noise weak_mean weak_std strong_mean strong_std")

    # raw per-trial errors
    errorplot.save_trials(trials_path,
                          noise_levels, weak_err, strong_err)

    # ----- plot -----
    wm, ws = weak_err.mean(1), weak_err.std(1)
    sm, ss = strong_err.mean(1), strong_err.std(1)
    floor = max(noise_levels[1] / 10, 1e-6)   # x-position for the sigma=0 point
    x_axis = np.where(noise_levels > 0, noise_levels, floor)

    # errorbar (mean +- std) or box (full per-trial spread); see errorplot.STYLE
    # or pass --box / --errorbar on the command line
    style = errorplot.style_from_argv()
    ax = plt.figure(figsize=(7, 5)).gca()
    hw = errorplot.draw_series(ax, x_axis, wm, ws, weak_err, style=style,
                               color='C0', marker='o',
                               label='TT-WSINDy (weak form)',
                               dodge=1.0 / errorplot.DODGE)
    hs = errorplot.draw_series(ax, x_axis, sm, ss, strong_err, style=style,
                               color='C1', marker='s',
                               label='MANDy (strong form, finite diff.)',
                               dodge=errorplot.DODGE)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-6, 1e0)
    plt.xlabel(r'relative noise level $\sigma$')
    plt.ylabel('relative coefficient error')
    plt.title('Weak vs. strong form TT regression on L96 '
                f'(D={D}, M={M}, F={F})')
    plt.legend(handles=[hw, hs])
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"results/weakvstrongformL96.png", dpi=150)
    plt.show()