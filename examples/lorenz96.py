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
    D = 20  # Dimension of system
    F = 8  # Forcing
    M = 10000 # num timepoints

    def L96(x, t):
        """Lorenz 96 model with constant forcing"""
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F 

    x0 = F * np.ones(D)  # Initial state (equilibrium)
    x0[0] += 0.01  # Add small perturbation to the first variable
    tn = 10
    t = np.linspace(0.0, tn, M)

    X = odeint(L96, x0, t).T
    f = [
        lambda x : 1,
        lambda x : x,
        lambda x : x**2 # superfluous basis feature
    ]

    # Make (weak) feature tensor
    phi, dphi = piecewise_polynomial(
        tn/20, 16, 0, tn, M
    )
    Mp = M - len(phi) + 1
    ThetaX = feature_tensor(X,f,phi=phi)

    thresh = 0
    lamb = 0.25
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
        print('--------')
        print('--------')
        print('dim = {}'.format(d+1))
        print('--------')
        print('--------')

        ThetaD = copy.deepcopy(Theta)
        Yd = Y[:,d]
        W.append(
            ThetaD.TT_STLS(Yd, lamb)
        )
        suppd = ThetaD.all_active_features()
        print(suppd)