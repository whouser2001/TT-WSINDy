import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, '../TT-WSINDy')
import numpy as np, copy
from feature_tensor import feature_tensor
from _validate_tmp import fput_trajectory, true_coeffs, tt_to_dense

np.set_printoptions(precision=3, suppress=True, linewidth=200)

D, M = 2, 1500
dt, beta, amp = 0.01, 0.7, 0.6
f = [lambda x: 1, lambda x: x, lambda x: x ** 2, lambda x: x ** 3]
J = len(f)

t, X, V = fput_trajectory(D, M, dt=dt, beta=beta, amplitude=amp, substeps=20, seed=0)
Wtrue = true_coeffs(D, J, beta)

Xddot = (X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]) / dt ** 2
Xint = X[:, 1:-1]
Theta_s = feature_tensor(Xint, f, phi=None)

# product of feature norms s_k for each monomial tuple
norms = Theta_s.feature_norms  # (J, D)
print("feature_norms (J x D):\n", norms)
s = np.ones((J,) * D)
for idx in np.ndindex(*([J] * D)):
    prod = 1.0
    for d in range(D):
        prod *= norms[idx[d], d]
    s[idx] = prod

# raw TT_PI (no unscale)
Th = copy.deepcopy(Theta_s)
Wtt_raw = Th.TT_PI(Xddot[0], threshold=1e-10)
raw = tt_to_dense(Wtt_raw, D, J)

# unscale version
Th2 = copy.deepcopy(Theta_s)
Wtt_un = Th2.unscale(Th2.TT_PI(Xddot[0], threshold=1e-10))
un = tt_to_dense(Wtt_un, D, J)

print("\ntrue dim0:\n", Wtrue[0])
print("\nraw (no unscale):\n", raw)
print("\nraw / s:\n", raw / s)
print("\nraw * s:\n", raw * s)
print("\nunscale():\n", un)
print("\nerr raw/s :", np.linalg.norm(raw / s - Wtrue[0]) / np.linalg.norm(Wtrue[0]))
print("err raw*s :", np.linalg.norm(raw * s - Wtrue[0]) / np.linalg.norm(Wtrue[0]))
print("err unscale:", np.linalg.norm(un - Wtrue[0]) / np.linalg.norm(Wtrue[0]))
