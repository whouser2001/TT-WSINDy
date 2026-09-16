"""
Construction of the piecewise polynomial test function.

References
----------
.. [1] A. Tran and D. M. Bortz, "Weak Form Scientific Machine
        Learning: Test Function Construction for System
        Identification", SIAM Journal on Scientific Computing,
        48 (2026), pp. C890-C915,
        https://doi.org/10.1137/25M1776020
"""
import numpy as np
from scipy.special import binom

def C2_pp(r,p):
    """2-norm normalizing constant of the piecewise polynomial test function."""
    s = 0
    for k in range(0,2*p+1):
        s += binom(2*p, k)*(-1)**k/(2*k+1)
    return np.pow(r,2*p)*np.sqrt(2*r*s)

def piecewise_polynomial(r, p, t0, tn, M, order=1):
    """
    Discretize a piecewise polynomial test function on the data sampling.

    Parameters
    ----------
    r : float
        Radius of support.
    p : int
        Degree.
    t0, tn : int
        Start and end timepoints.
    M : int
        Number of time snapshots.
    order : int
        Number of derivatives taken to compute dphi.

    Returns
    -------
    phi, dphi : np.ndarray
        Discretized test function and derivative, normalized in the
        2-norm with respect to phi.
    """

    def ph(t, r, p): return (np.abs(t) <= r)*np.pow(
                r**2 - t**2, p
            )
    C = C2_pp(r,p)
    def phi(t) : return (1/C)*ph(t,r,p)
    if order == 1:
        def dphi(t): return (-2*t*p/C)*ph(t,r,p-1)
    elif order == 2:
        # sign flipped to account for the second integration by parts
        def dphi(t): return (-1/C)*(-2*p*ph(t,r,p-1) + p*(p-1)*4*t**2*ph(t,r,p-2))

    dt = (tn - t0)/M
    cl = int(np.ceil(r/dt))
    phi_vec = phi(
        np.linspace(-cl*dt, cl*dt, 2*cl+1)
    )
    dphi_vec = dphi(
        np.linspace(-cl*dt, cl*dt, 2*cl+1)
    )
    return phi_vec, dphi_vec
