"""Temporary validation of the weak vs strong FPUT pipeline."""
import os, sys
sys.path.insert(0, '.')
sys.path.insert(0, '../TT-WSINDy')
sys.path.insert(0, '../../TT-WSINDy')
import numpy as np
from time import time
from scipy.signal import correlate
from feature_tensor import feature_tensor
from test_function import piecewise_polynomial


# ----- FPUT generator (copied from examples/fermipasta) -----
def _fpu_force(x, beta):
    xp = np.concatenate(([0.0], x, [0.0]))
    dr = xp[2:] - xp[1:-1]
    dl = xp[1:-1] - xp[:-2]
    return (dr - dl) + beta * (dr ** 3 - dl ** 3)


def fpu_energy(x, v, beta):
    xp = np.concatenate(([0.0], x, [0.0]))
    d = xp[1:] - xp[:-1]
    U = np.sum(0.5 * d ** 2 + (beta / 4.0) * d ** 4)
    return 0.5 * np.sum(v ** 2) + U


def fput_trajectory(n, m, dt=0.01, beta=0.7, amplitude=0.5, substeps=10, seed=0):
    rng = np.random.default_rng(seed)
    x = 2.0 * amplitude * rng.random(n) - amplitude
    v = np.zeros(n)
    positions = np.zeros((n, m)); velocities = np.zeros((n, m))
    positions[:, 0] = x; velocities[:, 0] = v
    h = dt / substeps
    F = _fpu_force(x, beta)
    for k in range(1, m):
        for _ in range(substeps):
            v = v + 0.5 * h * F
            x = x + h * v
            F = _fpu_force(x, beta)
            v = v + 0.5 * h * F
        positions[:, k] = x; velocities[:, k] = v
    t = np.arange(m) * dt
    return t, positions, velocities


# ----- polynomial algebra for true coefficients -----
def _padd(p, q):
    r = dict(p)
    for k, v in q.items():
        r[k] = r.get(k, 0.0) + v
    return r


def _pscale(p, c):
    return {k: c * v for k, v in p.items()}


def _pmul(p, q):
    r = {}
    for (m1, c1) in p.items():
        for (m2, c2) in q.items():
            md = dict(m1)
            for var, pw in m2:
                pass
            d = {}
            for var, pw in m1: d[var] = d.get(var, 0) + pw
            for var, pw in m2: d[var] = d.get(var, 0) + pw
            key = tuple(sorted(d.items()))
            r[key] = r.get(key, 0.0) + c1 * c2
    return r


def true_coeffs(D, J, beta):
    """Return list of (J,)*D arrays, one per oscillator output dim."""
    Ws = []
    for i in range(D):
        L, R = i - 1, i + 1
        # linear forms dr = x_R - x_i, dl = x_i - x_L  (drop walls)
        dr = {(): 0.0}
        dr = {}
        if R <= D - 1: dr[((R, 1),)] = dr.get(((R, 1),), 0.0) + 1.0
        dr[((i, 1),)] = dr.get(((i, 1),), 0.0) - 1.0
        dl = {}
        dl[((i, 1),)] = dl.get(((i, 1),), 0.0) + 1.0
        if L >= 0: dl[((L, 1),)] = dl.get(((L, 1),), 0.0) - 1.0
        # force = (dr - dl) + beta*(dr^3 - dl^3)
        lin = _padd(dr, _pscale(dl, -1.0))
        dr3 = _pmul(_pmul(dr, dr), dr)
        dl3 = _pmul(_pmul(dl, dl), dl)
        cub = _pscale(_padd(dr3, _pscale(dl3, -1.0)), beta)
        force = _padd(lin, cub)
        # build tensor
        W = np.zeros((J,) * D)
        for mono, coeff in force.items():
            idx = [0] * D
            for var, pw in mono:
                idx[var] = pw
            W[tuple(idx)] += coeff
        Ws.append(W)
    return Ws


def tt_to_dense(W, D, J):
    full = W.full()
    return np.asarray(full).reshape((J,) * D)


if __name__ == "__main__":
    D, M = 4, 1200
    dt, beta, amp = 0.01, 0.7, 0.6
    f = [lambda x: 1, lambda x: x, lambda x: x ** 2, lambda x: x ** 3]
    J = len(f)

    t, X, V = fput_trajectory(D, M, dt=dt, beta=beta, amplitude=amp, substeps=20, seed=0)
    t0, tM = t[0], t[-1]

    # energy conservation check
    e0 = fpu_energy(X[:, 0], V[:, 0], beta)
    eN = fpu_energy(X[:, -1], V[:, -1], beta)
    print(f"energy drift: {abs(eN - e0) / abs(e0):.3e}")
    print(f"X range: [{X.min():.3f}, {X.max():.3f}]")

    Wtrue = true_coeffs(D, J, beta)
    print("true coeff dim0 nonzeros:", np.argwhere(np.abs(Wtrue[0]) > 0).shape[0])

    import copy
    # ---------- WEAK (TT-WSINDy) ----------
    phi, dphi = piecewise_polynomial((tM - t0) / 20, 16, t0, tM, M, order=2)
    Theta_w = feature_tensor(X, f, phi=phi)
    dphi_e = np.expand_dims(dphi, axis=0)
    Yw = -1 * correlate(X, dphi_e, mode='valid').transpose()

    # ---------- STRONG (MANDy, FD) ----------
    Xddot = (X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]) / dt ** 2
    Xint = X[:, 1:-1]
    Theta_s = feature_tensor(Xint, f, phi=None)

    for tol in [0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        werr, serr = [], []
        for d in range(D):
            Th = copy.deepcopy(Theta_w)
            Wd = tt_to_dense(Th.unscale(Th.TT_PI(Yw[:, d], threshold=tol)), D, J)
            werr.append(np.linalg.norm(Wd - Wtrue[d]) / np.linalg.norm(Wtrue[d]))
            Th = copy.deepcopy(Theta_s)
            Wd = tt_to_dense(Th.unscale(Th.TT_PI(Xddot[d], threshold=tol)), D, J)
            serr.append(np.linalg.norm(Wd - Wtrue[d]) / np.linalg.norm(Wtrue[d]))
        print(f"tol={tol:.0e}  WEAK={np.mean(werr):.3e}  STRONG={np.mean(serr):.3e}")
