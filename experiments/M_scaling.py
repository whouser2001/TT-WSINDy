"""
M scaling
"""
import os,sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, 'TT-WSINDy'))
sys.path.insert(0, os.path.join(_ROOT, 'WSINDy'))
RESULTS = os.path.join(_HERE, 'results')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from ttwsindy import TT_WSINDy

# ------------------------------- problem setup -------------------------------
F = 8.0
C = 1.0
D = 8
t0 = 0.0
DT = 0.1
Ms = [100, 1000, 3000, 5000, 10000, 20000]

# candidate library f = {1, x}  (J = 2); its tensor product spans the true model
f = [lambda x: 1.0, lambda x: x]
J = len(f)

DEGREE = 16
EPS16 = 1e-16

TTlambs = np.linspace(1e-5, 5e-1, 10)
flatlambs = np.linsapce(1e-5, 1e-1, 10)

def l96_data(D, sigma, M, tM, seed=0, burn=20.0):
    """Simulate Lorenz-96, landing on the attractor before sampling.

    With NOISE_LEVEL > 0, additive Gaussian noise of standard deviation
    NOISE_LEVEL*||X||_F/sqrt(X.size) is applied to the sampled trajectory. The
    initial condition is drawn first, so a given seed puts the same trajectory
    under every noise level.
    """
    rhs = lambda x, t: (np.roll(x, -1) - np.roll(x, 2))*np.roll(x, 1) - C*x + F
    rng = np.random.default_rng(seed)
    x0 = (F/C)*np.ones(D) + 0.01*rng.standard_normal(D)
    x0 = odeint(rhs, x0, np.linspace(0, burn, 1000))[-1]     # burn-in to attractor
    X = odeint(rhs, x0, np.linspace(t0, tM, M)).T             # (D, M)
    if sigma:
        X = X + sigma*(np.linalg.norm(X, 'fro')/np.sqrt(X.size)) \
            * rng.standard_normal(X.shape)
    return X

def l96_true(D):
    """Per-equation true support as {tuple-of-powers over {1,x}: coefficient}."""
    true = []
    for d in range(D):
        terms = {tuple([0]*D): F}                              # constant  F
        t = [0]*D; t[d] = 1;                           terms[tuple(t)] = -C
        t = [0]*D; t[(d+1) % D] = 1; t[(d-1) % D] = 1; terms[tuple(t)] = 1.0
        t = [0]*D; t[(d-2) % D] = 1; t[(d-1) % D] = 1; terms[tuple(t)] = -1.0
        true.append(terms)
    return true

def coeff_error(W, supps, fmaps, D, true):
    """
    Coefficient error over each of the D dimensions.
    """
    errs = []
    for d in range(D):
        rec = {}
        if supps[d] is not None:
            for i, k in enumerate(supps[d]):
                rec[tuple(int(p) for p in fmaps[d][k])] = float(W[d][i])
        keys = set(true[d]) | set(rec)
        wt = np.array([true[d].get(k, 0.0) for k in keys])
        wh = np.array([rec.get(k, 0.0) for k in keys])
        denom = np.linalg.norm(wt) or 1.0
        errs.append(np.linalg.norm(wh - wt)/denom)
    return errs

def run_tt(X, D, eps, tM, r_frac):
    """One-pass low-rank TT-WSINDy at TT-PI SVD tolerance `eps`."""
    r = TT_WSINDy(X, t0, tM, f, TTlambs, flatlambs,
                  testfn=('piecewise_polynomial', r_frac, DEGREE, 1),
                  threshold=eps, verbosity=0, low_rank=True, one_pass=True)
    # walltime, supp, feature_maps, coefficients, coarse_supps
    return r[3], r[1], r[2], r[0], r[6]

def make_figures():
    raise NotImplementedError

if __name__=='__main__':
    # have claude do the execution.
    hi = 0