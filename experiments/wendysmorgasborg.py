"""
Accuracy test of ttwsindy vs. wsindy on problems featured in
the WENDy paper

logistic
lotka-volterra
fitzhugh-nagumo
hindmarsh-rose
protein transduction benchmark (PTB)
"""
import os,sys
sys.path.insert(0,'.')
sys.path.insert(0,'../TT-WSINDy')
import numpy as np
import itertools
from scipy.signal import correlate
from scipy.integrate import odeint
from ttwsindy import TT_WSINDy
from ttwsindy_finepass import TT_WSINDy as TT_WSINDy2
from sparsification import MSTLS
import test_function

def gen_logistic(
        M: int = 100,
        t0: float = 0,
        tM: float = 10,
        u0: float = 0.01,
        w: tuple[float,float] = (1,-1)
    ):
    """ logistic """
    t = np.linspace(t0,tM,M)
    exp_term = np.exp(w[0]*(t - t0))
    u = (w[0]*u0*exp_term)/(w[0]+w[1]*u0*(exp_term-1))
    return u, t

def gen_lotka_volterra(
        M: int = 100,
        t0: float = 0,
        tM: float = 5,
        u0: tuple[float,float] = (1,1),
        w: tuple[float,float,float,float] = (3,-1,-6,1)
    ):
    """ lotka volterra """
    t = np.linspace(t0,tM,M)
    def lotka_volterra(x,t):
        return [
            w[0]*x[0] + w[1]*x[0]*x[1],
            w[2]*x[1] + w[3]*x[0]*x[1]
        ]
    return odeint(lotka_volterra,u0,t).T, t

def gen_fitzhugh_nagumo(
        M: int = 1000,
        t0: float = 0,
        tM: float = 25,
        u0: tuple[float,float] = (0,0.1),
        w: tuple[float,float,float,float,float,float] = (
            3,-3,3,-1/3,17/150,1/15
        )
    ):
    """ fitzhugh nagumo """
    t = np.linspace(t0,tM,M)
    def fitzhugh_nagumo(x,t):
        return [
            w[0]*x[0] + w[1]*x[0]**3 + w[2]*x[1],
            w[3]*x[0] + w[4] + w[5]*x[1]
        ]
    return odeint(fitzhugh_nagumo,u0,t).T, t

def gen_hindmarsh_rose(
        M: int = 20000,
        t0: float = 0,
        tM: float = 10,
        u0: tuple[float,float,float] = (-1.31,-7.6,-0.2),
        w: tuple[float,...] = (
            10,-10,30,-10,10,-50,-10,0.04,0.0319,-0.01
        )
    ):
    """ hindmarsh rose """
    t = np.linspace(t0,tM,M)
    def hindmarsh_rose(x,t):
        return [
            w[0]*x[1] + w[1]*x[0]**3 + w[2]*x[0]**2 + w[3]*x[2],
            w[4] + w[5]*x[0]**2 + w[6]*x[1],
            w[7]*x[0] + w[8] + w[9]*x[2]
        ]
    return odeint(hindmarsh_rose, u0, t).T, t

def gen_ptb(
        M: int = 5000,
        t0: float = 0,
        tM: float = 25,
        u0: tuple[float,...] = (1,0,1,0,1),
        w: tuple[float,...] = (
            -0.07,-0.6,0.35,0.07,-0.6,0.05,0.17,0.6,
            -0.35,0.3,-0.017
        )
    ):
    """ protein transduction benchmark (PTB) """
    t = np.linspace(t0,tM,M)
    def ptb(x,t):
        rat = x[4]/(0.3 + x[4])
        return [
            w[0]*x[0] + w[1]*x[0]*x[2] + w[2]*x[3],
            w[3]*x[0],
            w[4]*x[0]*x[2] + w[5]*x[3] + w[6]*rat,
            w[7]*x[0]*x[2] + w[8]*x[3],
            w[9]*x[3] + w[10]*rat
        ]
    return odeint(ptb, u0, t).T, t

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

def run_flat(X, D, M, f, t0, tM):
    J = len(f)
    basis = np.stack([np.vectorize(f[j])(X).astype(float) for j in range(J)])
    G = np.ones((J ** D, M))
    fmap = list(itertools.product(*[range(J)] * D))
    for k, tup in enumerate(fmap):
        for d in range(D):
            G[k] *= basis[tup[d], d]
    phi, dphi = test_function.piecewise_polynomial((tM - t0) / 20, 16, t0, tM, M)
    # first-order weak form: <x_dot, phi> = -<x, phi'> (the -1 was missing before,
    # which sign-flipped the recovered coefficients)
    if D > 1: dphi = np.expand_dims(dphi, 0)
    Y = -1 * correlate(X, dphi, mode='valid').T
    G = correlate(G, np.expand_dims(phi, 0), mode='valid').T
    supps = []
    feature_maps = [fmap for _ in range(D)]
    for d in range(D):
        y_d = Y[:,d] if D > 1 else Y
        mstls_ret = MSTLS(G, y_d, flatlambs, verbose=False)
        supps.append(mstls_ret[1])
    return supps, feature_maps
    

if __name__=='__main__':

    data_dict = {}
    data_dict['logistic'] = (
        gen_logistic,
        [
            lambda x : 1,
            lambda x : x,
            lambda x : x**2
        ],
        [
            lambda n : '',
            lambda n : f'x_{n}',
            lambda n : f'x_{n}^2'
        ],
        np.linspace(10**(-11), 10**(-1), 10),
        np.linspace(10**(-11), 10**(-1), 10),
        10**(-16)
    )
    data_dict['lotka-volterra'] = (
        gen_lotka_volterra,
        [
            lambda x : 1,
            lambda x : x
        ],
        [
            lambda n : '',
            lambda n : f'x_{n}',
        ],
        np.linspace(10**(-8), 10**(-1), 7),
        np.linspace(10**(-8), 10**(-1), 7),
        10**(-16)
    )
    data_dict['fitzhugh-nagumo'] = (
        gen_fitzhugh_nagumo,
        [
            lambda x : 1,
            lambda x : x,
            lambda x : x**2,
            lambda x : x**3
        ],
        [
            lambda n : '',
            lambda n : f'x_{n}',
            lambda n : f'x_{n}^2',
            lambda n : f'x_{n}^3'
        ],
        np.linspace(10**(-11), 10**(-1), 10),
        np.linspace(10**(-11), 10**(-1), 10),
        10**(-16)
    )
    data_dict['hindmarsh-rose'] = (
        gen_hindmarsh_rose,
        [
            lambda x : 1,
            lambda x : x,
            lambda x : x**2,
            lambda x : x**3
        ],
        [
            lambda n : '',
            lambda n : f'x_{n}',
            lambda n : f'x_{n}^2',
            lambda n : f'x_{n}^3'
        ],
        np.linspace(10**(-2), 10**(-1), 10),
        np.linspace(10**(-11), 5*10**(-2), 10),
        10**(-16)
    )
    data_dict['ptb'] = (
        gen_ptb,
        [
            lambda x : 1,
            lambda x : x,
            lambda x : x/(0.3 + x)
        ],
        [
            lambda n : '',
            lambda n : f'x_{n}',
            lambda n : f'x_{n}/(0.3 + x_{n})'
        ],
        np.linspace(10**(-8), 10**(-1), 7),
        np.linspace(10**(-8), 10**(-1), 7),
        10**(-16)
    )

    keys = ['logistic', 'lotka-volterra',
            'fitzhugh-nagumo', 'hindmarsh-rose',
            'ptb']
    
    for key in ['hindmarsh-rose']:

        print()
        print(key)

        data = data_dict[key]
        X,t = data[0]()
        f = data[1]
        fstr = data[2]
        TTlambs = data[3]
        flatlambs = data[4]
        eps = data[5]

        ttwsindy_ret = TT_WSINDy(
            X, t[0], t[-1], f, TTlambs, flatlambs, verbosity=1,
            threshold=eps, low_rank=False
        )
        supps = ttwsindy_ret[1]
        feature_maps = ttwsindy_ret[2]
        print('TT-WSINDy discovered support:')
        print_supp(fstr, supps, feature_maps)

        # WSINDy
        D,M = X.shape if X.ndim > 1 else (1,X.size)
        wsindy_ret = run_flat(
            X, D, M, f, t[0], t[-1]
        )
        supps = wsindy_ret[0]
        feature_maps = wsindy_ret[1]
        print('WSINDy discovered support:')
        print_supp(fstr, supps, feature_maps)





        