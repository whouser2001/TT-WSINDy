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
    M = 5000     # num timepoints

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
    denom = 20
    TTlambs = 10**(
       (8/(denom))*np.arange(0,denom+1) - 8
    )
    #TTlambs = np.linspace(0.1, 1, 50)
    num = 25
    flatlambs = np.linspace(
        10**(-11), 2.5*10**(-6), num
    )

    eps = 10**(-22)
    results = TT_WSINDy(X, 0, tn, f, TTlambs, flatlambs, verbosity=2,
              threshold=eps)
    
    supps = results[1]
    feature_maps = results[2]
    print(results[-1])  #coarse supps
    # print(supp)
    # print(feature_maps)

    fstr = [lambda n : '',
            lambda n : f'x_{n}',
            lambda n : f'x_{n}^2']
    for d1 in range(D):

        supp = supps[d1]
        feature_map = feature_maps[d1]
        str = f'x_{d1 + 1}\' = '

        for k in supp:
            substr = ''
            for d2 in range(D):
                substr += fstr[feature_map[k][d2]](d2+1)
            if substr == '': substr += '1'
            str += substr
            if k != supp[-1]: str += ' + '
        
        print(str)
    # for d in range(D):
    #     str = f'x_{d}\' = '
    #     for k in range(len(supp[d])):
    #         substr = ''
    #         for l in range(len(feature_maps[d][supp[d][k]])):
    #             for dj in range(D):
    #                 substr += fstr[feature_maps[d][supp[d][k]][l]](dj)
    #         if substr == '': substr += '1'
    #         str += substr
    #     str += f' + '
    #     print(str)