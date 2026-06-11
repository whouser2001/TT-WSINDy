"""
Matrix STLS and TT-MSTLS
"""
import numpy as np
import numpy.linalg as la
import copy
from scikit_tt.tensor_train import TT
from feature_tensor import feature_tensor
import ttutils as utils
from time import time


def tensor_loss(W, W0, Theta, supp_ratio, W0Theta_norm):
    """
    Choice of loss function for MSTLS. Penalizes distance
    in quality of estimate from W0 against Theta, while rewarding
    small support size.

    Parameters
    ----------
    W : TT
        current estimate of coefficients
        J x ... x J
    W0 : TT
        initial estimate of coefficients, with no thresholding
        J x ... x J
    Theta : TT
        feature tensor against which we are regressing
        J x ... x J x Mp
    supp_ratio : float
        ratio of support size to total J^D features, computed outside
    w0Theta_norm : float
        norm of W0 @ Theta, precomputed for efficiency

    Returns
    -------
    loss : float
        value of loss function for current W

    Raises
    ------
    ValueError
        If W.shape != W0.shape
    """

    # Check shape
    if W.row_dims != W0.row_dims:
        raise ValueError(
            f'Input to loss function has shape {W.row_dims}, but \
                non-thresholded coefficient tensor has shape {W0.row_dims}'
        )

    # Compute W x Theta, and subtract W0 x Theta
    # Warning: W - W0 does NOT do elementwise subtraction,
    #   So (W - W0) x Theta does not work as might be expected
    Wprod = utils.W_contract(W, Theta)
    W0prod = utils.W_contract(W0, Theta)
    s1 = np.linalg.norm(Wprod - W0prod)/W0Theta_norm
    print(f'diff norm = {s1}')
    return s1 + supp_ratio

def TT_MSTLS(Theta, x, lambs, total_size, verbose=False):
    """
    Tensor Train MSTLS (TT-MSTLS) algorithm.

    Parameters
    ----------
    Theta : TT
        feature tensor
    x : np.ndarray
        target values
    lambs : list of float
        list of threshold values
    total_size : int
        total number of features

    Returns
    -------
    ThetaStar : feature_tensor
        feature tensor associated with the lowest loss.
        contains coarse_supp information
    """

    # Compute initial coefficient estimate, and
    # norm for loss.
    W0 = Theta.unscale(Theta.TT_PI(x))
    W0Theta_norm = np.linalg.norm(
        utils.W_contract(W0, Theta)
    ) # Vector 2-norm

    # track argmin
    min_loss = np.inf    # Non thresholded loss is 1
    ThetaStar = None
    Wstar = None
    suppStar = None

    Theta.verbose = False

    # Iterate over threshold values
    for i in range(len(lambs)):

        lamb = lambs[i]
        ThetaLa = copy.deepcopy(Theta)

        if i == 0: ThetaLa.verbose = verbose # To see weights

        # TT-STLS, get coeffs + support
        WLa = ThetaLa.TT_STLS(x, lamb)
        suppLa = ThetaLa.all_active_features()

        # Cast W back up to full size ([J]^D), for loss 
        WLa = utils.embed_full(WLa, suppLa, W0.row_dims)
        suppLa_size = np.prod([s.size for s in suppLa])
        suppLa_ratio = suppLa_size/total_size
        loss = tensor_loss(WLa, W0, Theta, suppLa_ratio, W0Theta_norm)

        if loss <= min_loss:
            min_loss = loss
            ThetaStar = ThetaLa
            Wstar = WLa
            suppStar = suppLa

        if verbose:
            print('-----------')
            print(f'iteration {i}')
            print(f'support = {suppLa}')
            print(f'supp size = {suppLa_size}')
            print(f'supp ratio = {suppLa_ratio}')
            print(f'lambda = {lamb}, loss = {loss}')
            print(f'min_loss = {min_loss}')
        

    return ThetaStar, Wstar, suppStar

def STLS(G, b, lamb, w_LS):
    """
    Sequential thresholding least squares with index tracking
    
    Returns (w, surviving_indices).
    """
    n = G.shape[1]
    max_its = n

    col_norms = la.norm(G, axis=0)
    col_norms[col_norms == 0] = 1.0                 
    bound = la.norm(b) / col_norms                  
    LB = lamb * np.maximum(1.0, bound)              
    UB = (1.0 / lamb) * np.minimum(1.0, bound)    

    w = np.asarray(w_LS, dtype=float).copy()
    active_prev = None
    for _ in range(max_its + 1):
        aw = np.abs(w)
        active = (aw >= LB) & (aw <= UB)            

        if active_prev is not None and np.array_equal(active, active_prev):
            break                                   
        if not active.any():
            return None, None, None

        w = np.zeros(n)
        sol, *_ = la.lstsq(G[:, active], b, rcond=None)
        w[active] = sol
        active_prev = active
    
    return sol, G[:, active_prev], np.where(active_prev)[0]

def matrix_loss(w, w0, G, supp_ratio, w0G_norm):
    """
    Choice of loss function for MSTLS. Penalizes distance
    in quality of estimate from W0 against Theta, while rewarding
    small support size.

    Parameters
    ----------
    W : np.array
        current estimate of coefficients
    W0 : np.array
        initial estimate of coefficients, with no thresholding
    Theta_flat : np.ndarray
        feature tensor against which we are regressing
    supp_ratio : float
        ratio of support size to total J^D features, computed outside
    w0Theta_norm : float
        norm of W0 @ Theta, precomputed for efficiency

    Returns
    -------
    loss : float
        value of loss function for current W
    """
    return la.norm((w - w0)@G)/w0G_norm + supp_ratio

def MSTLS(G, b, lambs, verbose=False):
    """
    TODO
    """
    Jtilde = G.shape[1] # G : M' x prod(Jd)
    w0, *_ = la.lstsq(G, b, rcond=None)
    Gw0 = G @ w0
    Gw0_norm = la.norm(Gw0)

    min_loss = np.inf
    wStar = None
    suppStar = None

    # Iterate over threshold values
    for i in range(len(lambs)):

        lamb = lambs[i]
        wLa, GLa, suppLa = STLS(G, b, lamb, w0)

        if wLa is None:
            loss = 1   # Support is empty
        else:
            # Compute G @ w and pad with zeros, to compare w/ G @ w0
            # TODO do I even need to pad?
            #   both should be estimate for x \in R^m'?
            GwLa = GLa @ wLa
            #GwLa_full = np.zeros(Jtilde)
            #GwLa_full[suppLa] = GwLa

            # Compute loss
            diffnorm = la.norm(GwLa - Gw0)/Gw0_norm
            suppratio = len(suppLa)/Jtilde
            loss = diffnorm + suppratio

        if loss <= min_loss:
            min_loss = loss
            wStar = wLa
            suppStar = suppLa

        if verbose:
            print('-----------')
            print(f'iteration {i+1}')
            print(f'lambda = {lamb}')
            print(f'support = {suppLa}')
            if wLa is not None:
                print(f'diff norm = {diffnorm}')
                print(f'supp ratio = {suppratio}')
                print(f'loss = {loss}')
            print(f'min loss = {min_loss}')

    return wStar, suppStar