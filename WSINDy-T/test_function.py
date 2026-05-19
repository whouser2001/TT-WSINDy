"""
Utils for the weak form. 

TODO: cite April's paper.
"""
import numpy as np
from scipy.special import binom

def C2_pp(r,p):
    """
    2-norm normalizing constant of the piecewise
    polynomial test function

    TODO: cite April's paper
    """
    s = 0
    for k in range(0,2*p+1):
        s += binom(2*p, k)*(-1)**k/(2*k+1)
    return np.pow(r,2*p)*np.sqrt(2*r*s)

def piecewise_polynomial(r, p, t0, tn, M):
    """
    Generate and discretize a piecewise polynomial
    test function, given the sampling of the
    data.

    Parameters
    ----------
    r : float
        radius of support of test fn
    p : int
        specifies degree of test fn
    t0, tn : int
        Start and end timepoints
    M : int
        number of time snapshots

    Returns
    -------
    phi, dphi : 
        discretization of the piecewise polynomial
        test funtion and its derivative.
        normalized in 2-norm wrt phi.
    """

    def ph(t, r, p): return (np.abs(t) <= r)*np.pow(
                r**2 - t**2, p
            )
    C = C2_pp(r,p)
    def phi(t) : return (1/C)*ph(t,r,p)
    def dphi(t): return (-2*t*p/C)*ph(t,r,p-1)

    dt = (tn - t0)/M
    cl = int(np.ceil(r/dt))
    phi_vec = phi(
        np.linspace(-cl*dt, cl*dt, 2*cl+1)
    )
    dphi_vec = dphi(
        np.linspace(-cl*dt, cl*dt, 2*cl+1)
    )
    return phi_vec, dphi_vec