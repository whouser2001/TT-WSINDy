import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, '../TT-WSINDy')
import numpy as np, copy
from scipy.signal import correlate
from feature_tensor import feature_tensor
from test_function import piecewise_polynomial
from _validate_tmp import fput_trajectory, true_coeffs, tt_to_dense

np.set_printoptions(precision=3, suppress=True, linewidth=200)

D, M = 2, 1500
dt, beta, amp = 0.01, 0.7, 0.6
f = [lambda x: 1, lambda x: x, lambda x: x ** 2, lambda x: x ** 3]
J = len(f)

t, X, V = fput_trajectory(D, M, dt=dt, beta=beta, amplitude=amp, substeps=20, seed=0)
t0, tM = t[0], t[-1]
Wtrue = true_coeffs(D, J, beta)
print("Wtrue[0] (dim 0):\n", Wtrue[0])
print("Wtrue[1] (dim 1):\n", Wtrue[1])

# strong form
Xddot = (X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]) / dt ** 2
Xint = X[:, 1:-1]
Theta_s = feature_tensor(Xint, f, phi=None)
print("\nstrong Theta ranks:", Theta_s.ranks, "row_dims:", Theta_s.row_dims, "Mp:", Theta_s.Mp)

for tol in [1e-12, 1e-8, 1e-4]:
    Th = copy.deepcopy(Theta_s)
    Wtt = Th.unscale(Th.TT_PI(Xddot[0], threshold=tol))
    full = Wtt.full()
    print(f"\n--- tol={tol:.0e} ---")
    print("W.full() shape:", np.asarray(full).shape, " W.order:", Wtt.order, " row_dims:", Wtt.row_dims)
    Wd = tt_to_dense(Wtt, D, J)
    print("recovered dim0:\n", Wd)
    print("max |recovered|:", np.abs(Wd).max())
