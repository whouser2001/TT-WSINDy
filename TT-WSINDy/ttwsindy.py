"""
TT-WSINDy
"""
import os,sys
sys.path.insert(0, '.')
sys.path.insert(0, '../TT-WSINDy')
import numpy as np
import copy
from scipy.signal import correlate
from time import time
import test_function
from feature_tensor import feature_tensor
import sparsification
import itertools

def TT_WSINDy(X, t0, tM, f, TTlambs, flatlambs,
                testfn=('piecewise_polynomial', 1/40, 16, 1), 
                loss='default',
                threshold=0.0,
                verbosity=0,
                low_rank=False,
                one_pass=False):
    """
    TT-WSINDy.
 
    Discover the governing equations of a dynamical system from data using
    weak-form SINDy with tensor-train sparsification: a coarse TT-MSTLS pass
    per dimension, followed by a fine matrix MSTLS solve.
 
    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape D x M (D system dimensions, M time points).
    t0, tM : float
        Initial and final time point. Data is assumed equispaced in time.
    f : list of callable
        Candidate functions fj : R -> R.
    TTlambs : iterable
        Threshold values to test in TT-MSTLS.
    flatlambs : iterable
        Threshold values to test in the flat MSTLS.
    testfn : tuple
        Test-function specification, structured as (name, *parameters).
        Currently supported:
            1.  name : piecewise_polynomial
                r : float
                    real number in (0, 1] giving the fraction of the time
                    interval the test function spans
                p : int
                    degree of the polynomial
                o : int
                    order of the ODE to be discovered. An oth-order ODE
                    requires 'dphi' to be the oth-order derivative.
            2.  name : manual
                phi : np.array
                    discretized phi data
                dphi: np.array
                    discretized phi derivative data
    loss : str
        Loss function. Currently supported:
            1. name : default
    threshold : float
        Truncation parameter applied to the matrix SVDs in TT-PI.
        threshold=0.0 computes pseudoinverses exactly.
    verbosity : int
        Print verbosity during execution:
            0 : print nothing
            1 : print coarse supports, final support, and walltime
                information (verbose)
            2 : additionally print weights, support, and loss at every
                tested lambda (debug)
    low_rank : bool
        If true, builds feature tensor directly in compressed
        form by a left-to-right SVD sweep over a small "carry" matrix.
        Preferred when M is large.
    one_pass : bool
        If true, performs TT-STLS non-iteratively; only performing
        a single regression/sparsification step.
        Preferred roughly when J^D is of the order 10^3 or smaller.

    Returns
    -------
    W : list of np.ndarray
        Per-dimension coefficient vectors on the surviving features.
    supp : list of np.ndarray
        Per-dimension surviving column indices into each feature map.
    feature_maps : list of list of tuple
        feature_maps[d][k] gives the original candidate-function index in
        each dimension for the k-th column of dimension d's library.
    ttwsindy_time : float
        Total wall-clock runtime.
    tt_mstls_time : float
        Cumulative TT-MSTLS runtime across dimensions.
    mstls_time : float
        Cumulative flat-MSTLS runtime across dimensions.
    coarse_supps : list of list of np.ndarray
        Per-dimension coarse supports from TT-MSTLS.
    """
    if X.ndim == 1: D,M = (1,X.size)
    else: D,M = X.shape
    J = len(f)
    problemSize = J**D

    verbose = (verbosity >= 1)
    debug = (verbosity//2 >= 1)

    st = time()

    if testfn[0] == 'piecewise_polynomial':
        radius = (tM - t0)*testfn[1]
        degree = testfn[2]
        phi, dphi = test_function.piecewise_polynomial(
            radius, degree, t0, tM, M, order=testfn[3]
        )
    elif testfn[0] == 'manual':
        phi, dphi = testfn[1:]
    else:
        return NotImplementedError('Test function string ' \
        'not supported. Enter one of piecewise_polynomial, manual.')
    
    Theta = feature_tensor(X, f, 
                            threshold=threshold,
                            phi=phi,
                            verbose=debug,
                            low_rank=low_rank)
    
    # compute the weak-form left-hand side
    phi = np.expand_dims(phi, axis=0)
    if D > 1: dphi = np.expand_dims(dphi, axis=0)
    Y = -1*correlate(X, dphi, mode='valid').transpose()    # (Mp, D)

    # store per-dimension results
    Ws = []
    supps = []
    feature_maps = []
    coarse_supps = []

    # If one_pass, compute the single TT SVD here
    pi_factors = Theta.svd(D, threshold=Theta.threshold,
                        ortho_l=True, ortho_r=True, overwrite=False) if one_pass else None

    tt_mstls_time = 0
    mstls_time = 0
    for d in range(D):

        # coarse pass: TT-MSTLS
        if d < D-1: Theta_d = copy.deepcopy(Theta)
        else: Theta_d = Theta

        tt_mstls_st = time()

        y_d = Y[:,d] if D > 1 else Y
        Theta_star, wStar, suppStar = sparsification.TT_MSTLS(
            Theta_d, y_d, TTlambs, problemSize, verbose=debug,
            one_pass=one_pass, pi_factors=pi_factors
        )
        
        tt_mstls_end = time()
        if verbose:
            print('------------------')
            print(f'TT-MSTLS concluded for d = {d}')
            print(f'coarse support: {suppStar}')
            print('Beginning flat MSTLS')

        coarse_supp = Theta_star.all_active_features()
        coarse_supps.append(coarse_supp)
        feature_map = list(itertools.product(*coarse_supp))
        basis = np.unique(
            np.concatenate(coarse_supp)
        )
        Jtilde = len(feature_map)

        # inverse index map
        inx = {basis[i]:i for i in range(len(basis))}

        # reconstruct WSINDy matrix G from surviving features
        basis_data = np.zeros((basis.size, D, M))
        for i in range(basis.size):
            j = basis[i]
            basis_data[i, :, :] = np.vectorize(f[j])(X)
        
        G = np.zeros((Jtilde, M))           # prod(Jd) x M
        for k in range(Jtilde):
            gk = np.ones(M)
            if D > 1:
                for dee in range(D):
                    gk *= basis_data[inx[feature_map[k][dee]], dee, :]
            else: gk = basis_data[inx[feature_map[k][0]], :, :]
            G[k, :] = gk

        G = correlate(G, phi, mode='valid').transpose()

        # fine pass: MSTLS
        mstls_st = time()
        wStar, suppStar = sparsification.MSTLS(
            G, y_d, flatlambs, verbose=debug
        )
        mstls_end = time()

        Ws.append(wStar)
        supps.append(suppStar)
        feature_maps.append(feature_map)
        
        tt_mstls_time += tt_mstls_end - tt_mstls_st
        mstls_time += mstls_end - mstls_st

    ttwsindy_time = time() - st
    if verbose:
        print('------------------')
        print('TT-WSINDy complete.')
        print(f'Total runtime: {ttwsindy_time}')
        print(f'TT-MSTLS runtime: {tt_mstls_time}')
        print(f'MSTLS runtime: {mstls_time}')
        print('------------------')

    return Ws, supps, feature_maps, ttwsindy_time, tt_mstls_time, mstls_time, coarse_supps