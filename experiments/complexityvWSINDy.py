"""
Testing walltime and memory against D, in comparison to
flat WSINDy.

TODO. Implement caching to TT-WSINDy to speed this up
"""
import os,sys
sys.path.insert(0, '.')
sys.path.insert(0, '../TT-WSINDy')
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sparsification import MSTLS
from ttwsindy import TT_WSINDy
import test_function
from scipy.signal import correlate
from scipy.integrate import odeint
import itertools

if __name__ == '__main__':

    # Whether to rerun simluation or read from file
    recompute_data = True

    F = 8       # Forcing function
    M = 10000     # num timepoints

    def L96(x,t):
        """Lorenz 96 model with constant forcing"""
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    
    t0 = 0
    tM = 10
    t = np.linspace(t0, tM, M)

    f = [
        lambda x : 1,
        lambda x : x,
        lambda x : np.sin(x)
    ]
    J = len(f)

    numTT = 15
    numflat = 25
    TTlambs = 10**((4/(numTT))*np.arange(0,numTT+1) - 4)
    flatlambs = np.linspace(10**(-10), 10**(-1), numflat)
    threshold = 10**(-16)

    D_min = 5
    D_max = 8   # inclusive
    walltimes = np.zeros((D_max - D_min + 1, 4))
    for D in range(D_min, D_max + 1): # Increase to ~[4,15] once operational

        if not recompute_data: break
        print(f'D={D}')
        print(f'Initial problem size: {D*(J**D)}')

        # Generate D-dimensional data
        x0 = F * np.ones(D)
        x0[0] += 0.01
        X = odeint(L96, x0, t).T

        # TT-WSINDy
        ttwsindy_ret = TT_WSINDy(
            X, t0, tM, f, TTlambs, flatlambs, verbosity=0, threshold=threshold
        )
        walltimes[D-D_min][0] = ttwsindy_ret[3]
        walltimes[D-D_min][2] = ttwsindy_ret[4]
        walltimes[D-D_min][3] = ttwsindy_ret[5]
        coarse_supps = ttwsindy_ret[6]
        reduced_size = np.sum(
            [np.prod([np.max([s.size, 1]) for s in coarse_supps[d]])
             for d in range(D)]
        )
        print(f'Reduced problem size: {reduced_size}')
        print('Beginning flat WSINDy.')

        # WSINDy
        wsindy_st = time()

        # construct full WSINDy problem
        basis_data = np.zeros((J,D,M))
        for j in range(J):
            basis_data[j,:,:] = np.vectorize(f[j])(X)

        feature_map = list(itertools.product(
            *[np.arange(J) for _ in range(D)]
        ))
        
        Jprime = J**D
        G = np.zeros((Jprime, M))
        for k in range(Jprime):
            gk = np.ones(M)
            for d in range(D):
                gk *= basis_data[feature_map[k][d], d, :]
            G[k, :] = gk
        
        phi, dphi = test_function.piecewise_polynomial(
            (tM - t0)/20, 16, t0, tM, M
        )
        phi = np.expand_dims(phi, axis=0)
        dphi = np.expand_dims(dphi, axis=0)
        Y = correlate(X, dphi, mode='valid').transpose()
        G = correlate(G, phi, mode='valid').transpose()

        # flat MSTLS
        for d in range(D):
            wsindy_ret = MSTLS(
                G, Y[:,d], flatlambs, verbose=False
            )
        
        wsindy_end = time()
        walltimes[D-D_min][1] = wsindy_end - wsindy_st
    
    # dump or load data
    if recompute_data: np.savetxt('results/complexityvWSINDy.txt', walltimes)
    else: walltimes = np.loadtxt('results/complexityvWSINDy.txt')

    # plot results
    Ds = range(D_min, D_max + 1)

    plt.plot(Ds, walltimes[:,0], label='TT-WSINDy', marker='o')
    plt.plot(Ds, walltimes[:,1], label='matrix WSINDy', marker='o')
    plt.plot(Ds, walltimes[:,2], label='(TTW) TT-MSTLS', marker='o')
    plt.plot(Ds, walltimes[:,3], label='(TTW) MSTLS', marker='o')

    plt.legend()
    plt.xlabel('number of dimensions')
    plt.ylabel('walltime')
    
    plt.savefig('results/complexityvWSINDy.png')
    plt.show()






