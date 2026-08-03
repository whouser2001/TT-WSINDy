"""
Rank truncation demonstration.
"""
import os,sys
sys.path.insert(0, '.')
sys.path.insert(0, '../TT-WSINDy')
sys.path.insert(0, '../WSINDy')
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sparsification import MSTLS
from ttwsindy import TT_WSINDy
import test_function
from scipy.signal import correlate
from scipy.integrate import odeint
import itertools

def print_supp(fstr, supps, feature_maps):

    D = len(supps)
    for d1 in range(D):

        supp = supps[d1]
        feature_map = feature_maps[d1]
        str = f'x_{d1 + 1}\' : '

        if supp is not None:
            for k in supp:
                substr = ''
                for d2 in range(D):
                    substr += fstr[feature_map[k][d2]](d2+1)
                if substr == '': substr += '1'
                str += substr
                if k != supp[-1]: str += '  '

        print(str)

if __name__ == '__main__':
    
    recompute_data = False
    plot_walltimes = True

    F = 8

    def L96(x,t):
        """Lorenz 96 model with constant forcing"""
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    
    f = [lambda x : 1, lambda x : x]
    fstr = [lambda n : '', lambda n : f'x_{n}']
    J = len(f)
    if len(fstr) != J: raise ValueError('fstr does not correspond to f')

    nTT = 10
    nFlat = 10
    TTlambs = np.linspace(10**(-5), 10**(-1), nTT)
    flatlambs = np.linspace(10**(-8), 10**(-1), nFlat)

    Ds = [4,5,6,7,8,9]
    nAvg = 1
    thresholds = [0, 1e-16]

    M = 2000

    walltimes0 = np.zeros((len(Ds),nAvg))
    walltimes1 = np.zeros((len(Ds),nAvg))
    for k in range(len(Ds)):

        D = Ds[k]
        t0 = 0
        tM = 30
        t = np.linspace(t0,tM,M)

        if not recompute_data: break
        print()
        print(f'D = {D}')

        x0 = F*np.ones(D)
        x0[0] += 0.01
        X = odeint(L96, x0, t).T

        print('Exact')
        for i in range(nAvg):
            ttwsindy_ret = TT_WSINDy(
                X, t0, tM, f, TTlambs, flatlambs, verbosity=0, threshold=thresholds[0],
                low_rank=False, one_pass=True
            )
            walltimes0[k][i] = ttwsindy_ret[3]

        supps = ttwsindy_ret[1]
        feature_maps = ttwsindy_ret[2]
        print('Exact Discovered support:')
        print_supp(fstr, supps, feature_maps)

        print('Truncated')
        for i in range(nAvg):
            ttwsindy_ret = TT_WSINDy(
                X, t0, tM, f, TTlambs, flatlambs, verbosity=0, threshold=thresholds[1],
                low_rank=True, one_pass=True
            )
            walltimes1[k][i] = ttwsindy_ret[3]
            
        supps = ttwsindy_ret[1]
        feature_maps = ttwsindy_ret[2]
        print('truncated Discovered support:')
        print_supp(fstr, supps, feature_maps)

        np.savetxt('results/ranktruncation20.txt', walltimes0)
        np.savetxt('results/ranktruncation21.txt', walltimes1)

    if plot_walltimes:
        if not recompute_data: 
            walltimes0 = np.loadtxt('results/ranktruncation20.txt')
            walltimes1 = np.loadtxt('results/ranktruncation21.txt')

        plt.plot(Ds, walltimes0, label='Original construction walltime', marker='o', color='blue')
        plt.plot(Ds, walltimes1, label=r'$\text{Low rank}, \epsilon=10^{-16}$ walltime', marker='o', color='orange')
        
        plt.legend()
        plt.title('Exact vs. low rank walltimes, Lorenz96, M=2000')
        plt.xlabel('D')
        plt.ylabel('walltime')
        plt.yscale('log')
        plt.grid(True,which='both',ls=':',alpha=0.5)
        plt.tight_layout()
        plt.savefig('results/ranktruncation2.png')
        plt.show()