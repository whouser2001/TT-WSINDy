"""
Fermi-pasta-ulam-tsingou simulation
"""
import os,sys
sys.path.insert(0, '.')
sys.path.insert(0, '../TT-WSINDy')
sys.path.insert(0, '../../TT-WSINDy')
import numpy as np
import copy
from time import time
from feature_tensor import feature_tensor
from test_function import piecewise_polynomial
from ttwsindy import TT_WSINDy
from scipy.signal import correlate
from scipy.integrate import odeint

"""
Trajectory-based Fermi-Pasta-Ulam-Tsingou generator for weak-form
system identification (TT-WSINDy).

Unlike the scattered-sample MANDy generator, this integrates the
first-order system

    x_dot = v
    v_dot = F(x)

with the symplectic velocity-Verlet (Stoermer-Verlet) scheme, producing a
single time-ordered, uniformly-sampled trajectory that the weak form can
integrate against test functions. Boundary conditions are fixed walls
x_{-1} = x_n = 0, matching the original scikit-tt generator.

    F_i(x) = (x_{i+1} - 2 x_i + x_{i-1})
             + beta [ (x_{i+1} - x_i)^3 - (x_i - x_{i-1})^3 ]
"""
import numpy as np


def _fpu_force(x, beta):
    """Acceleration field F(x) with fixed walls x_{-1} = x_n = 0.

    Zero-padding the displacement vector reproduces the boundary rows of the
    original generator exactly (left wall drops x_{-1}, right wall drops x_n).
    """
    xp = np.concatenate(([0.0], x, [0.0]))   # length n + 2
    dr = xp[2:] - xp[1:-1]                    # x_{i+1} - x_i
    dl = xp[1:-1] - xp[:-2]                   # x_i - x_{i-1}
    return (dr - dl) + beta * (dr ** 3 - dl ** 3)


def fpu_energy(x, v, beta):
    """Total Hamiltonian H = 1/2 |v|^2 + U(x) for one configuration.

    U(x) = sum_springs [ 1/2 Delta^2 + beta/4 Delta^4 ],  Delta = x_{i+1} - x_i,
    with the wall springs included via zero-padding.
    """
    xp = np.concatenate(([0.0], x, [0.0]))
    d = xp[1:] - xp[:-1]                       # n + 1 spring stretches
    U = np.sum(0.5 * d ** 2 + (beta / 4.0) * d ** 4)
    T = 0.5 * np.sum(v ** 2)
    return T + U


def fermi_pasta_ulam_trajectory(
    number_of_oscillators,
    number_of_snapshots,
    dt=0.01,
    beta=0.7,
    amplitude=0.1,
    x0=None,
    v0=None,
    substeps=1,
    seed=None,
):
    """Generate a single FPUT trajectory for weak-form identification.

    Parameters
    ----------
    number_of_oscillators : int
        number of oscillators n
    number_of_snapshots : int
        number of recorded time samples m (uniformly spaced by dt)
    dt : float
        spacing between recorded snapshots (the sampling rate the weak form
        sees). Keep it fine enough to resolve the test function and the
        fastest mode.
    beta : float
        cubic coupling strength (0.7 in the reference model)
    amplitude : float
        half-width of the random initial displacement box [-amplitude, amplitude].
        Larger amplitude excites the cubic term more strongly; at amplitude ~ 0.1
        the nonlinearity contributes only a few percent of the linear force, so
        recovering beta from a single low-amplitude run is poorly conditioned.
    x0, v0 : array_like or None
        explicit initial displacement / velocity (length n). If None, x0 is
        drawn uniformly from the amplitude box and v0 is zero (release from rest).
    substeps : int
        number of Verlet substeps per recorded snapshot. Increase to keep the
        integration accurate while sampling coarsely (internal step = dt / substeps).
    seed : int or None
        RNG seed for the random initial condition.

    Returns
    -------
    t : ndarray (number_of_snapshots,)
        sample times
    positions : ndarray (number_of_oscillators, number_of_snapshots)
        x(t) along the trajectory  (use to build the feature library)
    velocities : ndarray (number_of_oscillators, number_of_snapshots)
        v(t) along the trajectory  (the first-order LHS: v_dot weak-projects to
        -<v, phi_dot>)
    """
    n = number_of_oscillators
    m = number_of_snapshots
    rng = np.random.default_rng(seed)

    if x0 is None:
        x = 2.0 * amplitude * rng.random(n) - amplitude
    else:
        x = np.asarray(x0, dtype=float).copy()
    if v0 is None:
        v = np.zeros(n)
    else:
        v = np.asarray(v0, dtype=float).copy()

    positions = np.zeros((n, m))
    velocities = np.zeros((n, m))
    positions[:, 0] = x
    velocities[:, 0] = v

    h = dt / substeps
    F = _fpu_force(x, beta)                    # reused across steps (1 eval/step)
    for k in range(1, m):
        for _ in range(substeps):
            v = v + 0.5 * h * F                # half kick
            x = x + h * v                      # drift
            F = _fpu_force(x, beta)            # force at new position
            v = v + 0.5 * h * F                # half kick
        positions[:, k] = x
        velocities[:, k] = v

    t = np.arange(m) * dt
    return t, positions, velocities


if __name__ == "__main__":

    D = 5       # dimension of system
    M = 2500    # num timepoints

    t, X, V = fermi_pasta_ulam_trajectory(D,M,seed=0)

    f = [
        lambda x : 1,
        lambda x : x,
        lambda x : x**2,
        lambda x : x**3
    ]
    J = len(f)

    nTT = 15
    TTlambs = 10**((2/nTT)*np.arange(0,nTT+1)-3)
    nMat = 25
    Matlambs = 10**((7/nMat)*np.arange(0,nMat+1)-10)

    eps = 10**(-16)
    results = TT_WSINDy(X, t[0], t[-1], f, TTlambs, Matlambs,
                        testfn=('piecewise_polynomial',1/20,16,2),
                        threshold=eps,
                        verbosity=2)