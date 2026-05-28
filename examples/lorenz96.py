"""
Lorenz96 simulation, arbitrarily high-dimensional system.
Data gen via https://en.wikipedia.org/wiki/Lorenz_96_model
"""
import os,sys
sys.path.insert(0, '.')
sys.path.insert(0, '/WSINDy-T')
sys.path.insert(0, '../WSINDy-T')
import numpy as np
import copy
from feature_tensor import feature_tensor
from test_function import piecewise_polynomial
from scipy.signal import correlate
from scipy.integrate import odeint

if __name__ == '__main__':
    D = 5  # Dimension of system
    F = 8  # Forcing
    M = 1000 # num timepoints

    def L96(x, t):
        """Lorenz 96 model with constant forcing"""
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F 

    x0 = F * np.ones(D)  # Initial state (equilibrium)
    x0[0] += 0.01  # Add small perturbation to the first variable
    t = np.linspace(0.0, 30.0, M)

    X = odeint(L96, x0, t).T
    f = [
        lambda x : 1,
        lambda x : x,
        lambda x : x**2, # superfluous basis feature
    ]

    # Make (weak) feature tensor
    phi, dphi = piecewise_polynomial(
        0.5, 16, 0, 10, M
    )
    Mp = M - len(phi) + 1
    ThetaX = feature_tensor(X,f,phi=phi)

    thresh = 0
    lamb = 10**-5
    Theta = feature_tensor(X,f,
                           threshold=thresh,
                           phi=phi,
                           verbose=True)
    
    # Weak form LHS
    dphi = np.expand_dims(dphi, axis=0)
    Y = correlate(X, dphi, mode='valid').transpose() # (Mp, D)

    # Apply TT-STLS, for each dimension separately
    W = []
    for d in range(Y.shape[1]):
        ThetaD = copy.deepcopy(Theta)
        Yd = Y[:,d]
        W.append(
            ThetaD.TT_STLS(Yd, lamb)
        )
        suppd = ThetaD.all_active_features()
        print(suppd)