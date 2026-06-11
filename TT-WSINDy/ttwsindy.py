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
from scikit_tt import tensor_train
import test_function
from feature_tensor import feature_tensor
import sparsification
import itertools

def TT_WSINDy(X, t0, tM, f, TTlambs, flatlambs,
                testfn=('piecewise_polynomial', 1/20, 16), 
                loss='default',
                threshold=0.0,
                verbose=False):
    """
    TT-WSINDy.

    Parameters
    ----------
    X : np.array
        Data matrix, size M x D
    t0,tM : float
        initial and final timepoint. Data is assumed to be
        equispaced in time.
    f : list
        List of basis features fj : R -> R
    TTlambs : iterable
        lambda values to test over in TT-MSTLS
    flatlambs : iterable
        lambda values to test over in MSTLS
    testfn : tuple
        information that informs construction of the test function
        tuple structured like (name, *parameters). Currently suported:
            1.  name : piecewise_polynomial
                r : float
                    real number in (0,1] that gives the ratio of time
                    interval that test function spans
                p : int
                    degree of the polynomial
    loss : string
        specifies loss function. Currently supported
            1. name : default
    threshold : float
        truncation parameter to be applied to matrix SVDs of TT-PI. 
        setting threshold=0.0 means pseudoinverses are computed exactly.
    verbose : bool
        toggle print statements during execution
    """
    D = X.shape[0]
    M = X.shape[1]
    J = len(f)
    problemSize = J**D

    if verbose: st = time()

    # testfn = (name, *args)
    if testfn[0] == 'piecewise_polynomial':
        radius = (tM - t0)*testfn[1]
        degree = testfn[2]
        phi, dphi = test_function.piecewise_polynomial(
            radius, degree, t0, tM, M
        )
    elif testfn == 'Cinfty_bump':
        return NotImplementedError
    else:
        return NotImplementedError
    
    Theta = feature_tensor(X, f, 
                            threshold=threshold,
                            phi=phi,
                            verbose=verbose)
    
    # Compute LHS
    phi = np.expand_dims(phi, axis=0)
    dphi = np.expand_dims(dphi,axis=0)
    Y = correlate(X, dphi, mode='valid').transpose()    # (Mp, D)

    # store results
    W = []
    supp = []

    # TODO add functionality to run these loops in parallel
    # TODO currently doing just 1 dim for debugging
    tt_mstls_time = 0
    mstls_time = 0
    for d in range(D):
    #for d in range(1):

        if verbose:
            print('--------')
            print('dim = {}'.format(d))
            print('--------')
        
        # TT-MSTLS
        Theta_d = copy.deepcopy(Theta)

        if verbose: tt_mstls_st = time()

        y_d = Y[:,d]
        Theta_star, wStar, suppStar = sparsification.TT_MSTLS(
            Theta_d, y_d, TTlambs, problemSize, verbose=verbose
        )

        if verbose:
            tt_mstls_end = time()
            print('--------')
            print('TT-MSTLS concluded.')
            print(f'coarse support: {suppStar}')
            print('Beginning flat MSTLS')
            print('--------')

        coarse_supp = Theta_star.all_active_features()
        feature_map = list(itertools.product(*coarse_supp))
        basis = np.unique(
            np.concatenate(coarse_supp)
        )
        Jtilde = len(feature_map)

        # Reconstruct WSINDy matrix G from surviving features
        basis_data = np.zeros((basis.size, D, M))
        for i in range(basis.size):
            j = basis[i]
            basis_data[i, :, :] = np.vectorize(f[j])(X)
        
        G = np.zeros((Jtilde, M))           # prod(Jd) x M
        for k in range(Jtilde):
            gk = np.ones(M)
            for d in range(D):
                gk *= basis_data[feature_map[k][d], d, :]
            G[k, :] = gk

        G = correlate(G, phi, mode='valid').transpose()

        if verbose: mstls_st = time()

        # MSTLS
        wStar, suppStar = sparsification.MSTLS(
            G, y_d, flatlambs, verbose=verbose
        )

        if verbose: mstls_end = time()

        W.append(wStar)
        supp.append(suppStar)

        print([[base.item() for base in feature] for feature in feature_map])
        tt_mstls_time += tt_mstls_end - tt_mstls_st
        mstls_time += mstls_end - mstls_st


    if verbose:
        end = time()
        print('------------------')
        print('TT-WSINDy complete.')
        print(f'Total runtime: {end - st}')
        print(f'TT-MSTLS runtime: {tt_mstls_time}')
        print(f'MSTLS runtime: {mstls_time}')
        for d in range(D):
        #for d in range(1):
            # TODO: print in terms of original features, instead
            print(f'dim {d} support: {supp[d]}')
        print('------------------')

    return wStar, suppStar