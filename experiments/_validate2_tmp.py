import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, '../TT-WSINDy')
import numpy as np, copy
from time import time
from scipy.signal import correlate
from feature_tensor import feature_tensor
from test_function import piecewise_polynomial
from _validate_tmp import fput_trajectory, true_coeffs


def norm_product_tensor(Theta, D, J):
    norms = Theta.feature_norms
    s = np.ones((J,) * D)
    for idx in np.ndindex(*([J] * D)):
        p = 1.0
        for d in range(D):
            p *= norms[idx[d], d]
        s[idx] = p
    return s


def tt_pi_coeffs(Theta, y, threshold, D, J):
    """True-space dense coefficient tensor from one TT-PI solve."""
    Th = copy.deepcopy(Theta)
    W = Th.TT_PI(y, threshold=threshold)
    raw = np.asarray(W.full()).reshape((J,) * D)
    return raw / norm_product_tensor(Theta, D, J)


if __name__ == "__main__":
    D, M = 4, 1500
    dt, beta, amp = 0.01, 0.7, 0.6
    f = [lambda x: 1, lambda x: x, lambda x: x ** 2, lambda x: x ** 3]
    J = len(f)

    t, X, V = fput_trajectory(D, M, dt=dt, beta=beta, amplitude=amp, substeps=20, seed=0)
    t0, tM = t[0], t[-1]
    Wtrue = true_coeffs(D, J, beta)
    Wtrue_stack = np.stack(Wtrue)

    def run(Xdata, tol):
        # weak
        phi, dphi = piecewise_polynomial((tM - t0) / 20, 16, t0, tM, M, order=2)
        Tw = feature_tensor(Xdata, f, phi=phi)
        Yw = -1 * correlate(Xdata, np.expand_dims(dphi, 0), mode='valid').transpose()
        Ww = np.stack([tt_pi_coeffs(Tw, Yw[:, d], tol, D, J) for d in range(D)])
        # strong
        Xddot = (Xdata[:, 2:] - 2 * Xdata[:, 1:-1] + Xdata[:, :-2]) / dt ** 2
        Ts = feature_tensor(Xdata[:, 1:-1], f, phi=None)
        Ws = np.stack([tt_pi_coeffs(Ts, Xddot[d], tol, D, J) for d in range(D)])
        ew = np.linalg.norm(Ww - Wtrue_stack) / np.linalg.norm(Wtrue_stack)
        es = np.linalg.norm(Ws - Wtrue_stack) / np.linalg.norm(Wtrue_stack)
        return ew, es

    print("=== clean data, threshold sweep ===")
    for tol in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]:
        ew, es = run(X, tol)
        print(f"tol={tol:.0e}  WEAK={ew:.3e}  STRONG={es:.3e}")

    print("\n=== noise sweep (tol=1e-10) ===")
    rng = np.random.default_rng(1)
    xstd = X.std()
    for sigma in [0.0, 1e-4, 1e-3, 1e-2, 5e-2]:
        Xn = X + sigma * xstd * rng.standard_normal(X.shape)
        ew, es = run(Xn, 1e-10)
        print(f"sigma={sigma:.0e}  WEAK={ew:.3e}  STRONG={es:.3e}")
