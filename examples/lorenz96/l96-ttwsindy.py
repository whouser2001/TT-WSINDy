"""
Lorenz96 simulation, arbitrarily high-dimensional system.
Data gen via https://en.wikipedia.org/wiki/Lorenz_96_model
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

if __name__ == '__main__':
    D = 5       # Dimension of system
    F = 8       # Forcing
    M = 500    # num timepoints

    def L96(x, t):
        """Lorenz 96 model with constant forcing"""
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F 

    x0 = F * np.ones(D)         # Initial state (equilibrium)
    x0[0] += 0.01               # Add small perturbation to the first variable
    tn = 10
    t = np.linspace(0.0, tn, M)

    X = odeint(L96, x0, t).T
    f = [
        lambda x : 1,
        lambda x : x,
        lambda x : np.sin(x)        # superfluous basis feature
    ]
    J = len(f)

    # Lambdas range
    denom = 10
    TTlambs = 10**(
        (3.7/(denom))*np.arange(0,denom+1) - 4
    )
    num = 25
    flatlambs = np.linspace(
        10**(-10), 2.5*10**(-8), num
    )

    TT_WSINDy(X, 0, tn, f, TTlambs, flatlambs, verbose=True)
        