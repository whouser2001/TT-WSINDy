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
from sparsification import TT_MSTLS
from scipy.signal import correlate
from scipy.integrate import odeint

if __name__ == '__main__':
    D = 5      # Dimension of system
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
        lambda x : np.sin(x)       # superfluous basis feature
    ]
    J = len(f)

    # Make (weak) feature tensor
    phi, dphi = piecewise_polynomial(
        tn/20, 16, 0, tn, M
    )
    Mp = M - len(phi) + 1
    thresh = 0
    Theta = feature_tensor(X,f,
                           threshold=thresh,
                           phi=phi,
                           verbose=True)
    
    # Weak form LHS
    dphi = np.expand_dims(dphi, axis=0)
    Y = correlate(X, dphi, mode='valid').transpose() # (Mp, D)

    # Lambdas range, from seth code
    denom = 20
    lambs = 10**(
        (3.7/(denom))*np.arange(0,denom+1) - 4
    )

    # Apply TT-STLS, for each dimension separately
    W = []
    for d in range(Y.shape[1]):
    #for d in range(1):
        print('--------')
        print('--------')
        print('dim = {}'.format(d+1))
        print('--------')
        print('--------')

        st_mstls = time()

        ThetaD = copy.deepcopy(Theta)
        Yd = Y[:,d]
        Theta_star, Wstar, suppStar = TT_MSTLS(ThetaD, Yd, lambs, J**D, verbose=True)

        end_mstls = time()

        print('--------')
        print('TT-MSTLS complete.')
        print(f'Runtime: {end_mstls - st_mstls}')
        print(f'Optimal support : {suppStar}')
        print('--------')
        