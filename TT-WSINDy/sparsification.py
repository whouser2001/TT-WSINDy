"""
Sparsification: TT-MSTLS (tensor train) and the flat matrix MSTLS/STLS
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
    MSTLS loss for a tensor-train coefficient estimate.

    Penalizes the change in fit relative to the non-thresholded estimate W0,
    while rewarding a smaller support.

    Parameters
    ----------
    W : TT
        Current coefficient estimate, embedded to full size [J]^D.
    W0 : TT
        Non-thresholded coefficient estimate, full size [J]^D.
    Theta : TT
        Feature tensor being regressed against, [J]^D x Mp.
    supp_ratio : float
        Support size as a fraction of the J^D candidate features.
    W0Theta_norm : float
        Precomputed norm of W0 contracted with Theta.

    Returns
    -------
    loss : float
        Relative fit difference plus supp_ratio.
    diff_norm : float
        The relative fit-difference term on its own.

    Raises
    ------
    ValueError
        If W and W0 have different shapes.
    """

    # Check shape
    if W.row_dims != W0.row_dims:
        raise ValueError(
            f'Input to loss function has shape {W.row_dims}, but \
                non-thresholded coefficient tensor has shape {W0.row_dims}'
        )

    # W contracted with Theta, minus W0 contracted with Theta.
    # Note: W - W0 is NOT elementwise subtraction on a TT, so
    # (W - W0) x Theta would not give the intended difference.
    Wprod = utils.W_contract(W, Theta)
    W0prod = utils.W_contract(W0, Theta)
    s1 = np.linalg.norm(Wprod - W0prod)/W0Theta_norm
    return s1 + supp_ratio, s1

def TT_MSTLS(Theta, x, lambs, total_size, verbose=False):
    """
    Tensor-train MSTLS (TT-MSTLS).

    Sweep over threshold values, run TT-STLS at each, and keep the support
    with the lowest loss. Weights are cached across thresholds, so each
    distinct support is solved only once.

    Parameters
    ----------
    Theta : feature_tensor
        Feature tensor to sparsify.
    x : np.ndarray
        Target values.
    lambs : list of float
        Threshold values to test.
    total_size : int
        Total number of candidate features (J^D).
    verbose : bool
        If True, print per-threshold diagnostics.

    Returns
    -------
    ThetaStar : feature_tensor
        Feature tensor achieving the lowest loss; carries the coarse support
        in its supp_indices.
    Wstar : TT
        Coefficient estimate at the lowest loss, embedded to full size.
    suppStar : list of np.ndarray
        Surviving candidate-function indices per dimension.
    """

   # initial, non-thresholded estimate and its fit norm (for the loss)
    W0 = Theta.unscale(Theta.TT_PI(x))
    W0Theta_norm = np.linalg.norm(utils.W_contract(W0, Theta))

    min_loss = 1        # Non thresholded loss is 1
    ThetaStar = None
    Wstar = None
    suppStar = None

    Theta.verbose = False

    weight_cache = {}
    for i in range(len(lambs)):

        lamb = lambs[i]
        ThetaLa = copy.deepcopy(Theta)

        if i == 0: ThetaLa.verbose = verbose

        # TT-STLS: coefficient estimate and surviving support
        WLa = ThetaLa.TT_STLS(x, lamb, weight_cache=weight_cache)
        suppLa = ThetaLa.all_active_features()

        # embed W back to full size [J]^D for the loss 
        WLa = utils.embed_full(WLa, suppLa, W0.row_dims)
        suppLa_size = np.prod([s.size for s in suppLa])
        suppLa_ratio = suppLa_size/total_size
        loss, diff_norm = tensor_loss(WLa, W0, Theta, suppLa_ratio, W0Theta_norm)

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
            print(f'diff norm = {diff_norm}')
            print(f'lambda = {lamb}, loss = {loss}')
            print(f'min_loss = {min_loss}')
        

    return ThetaStar, Wstar, suppStar

def STLS(G, b, lamb, w_LS):
    """
    Sequential thresholding least squares (STLS) with index tracking.

    Iteratively keeps coefficients whose magnitude falls within a per-column
    band [LB, UB] and re-solves on the surviving columns until the support
    stabilizes.

    Parameters
    ----------
    G : np.ndarray
        Library matrix, Mp x prod(Jd).
    b : np.ndarray
        Target values, length Mp.
    lamb : float
        Thresholding parameter setting the band width.
    w_LS : np.ndarray
        Initial (least-squares) coefficient estimate.

    Returns
    -------
    w : np.ndarray or None
        Coefficients on the surviving columns; None if the support empties.
    G_supp : np.ndarray or None
        Columns of G for the surviving features; None if the support empties.
    surv : np.ndarray or None
        Indices of the surviving columns into G; None if the support empties.
    """
    n = G.shape[1]
    max_its = n

    col_norms = la.norm(G, axis=0)
    col_norms[col_norms == 0] = 1.0                 # guard against zero-norm columns
    bound = la.norm(b) / col_norms                  # ||b|| / ||G_k|| per column
    LB = lamb * np.maximum(1.0, bound)              # per-column lower band
    UB = (1.0 / lamb) * np.minimum(1.0, bound)      # per-column upper band   

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

def MSTLS(G, b, lambs, verbose=False):
    """
    Flat (matrix) MSTLS on the reduced library from TT-MSTLS.

    Sweep over threshold values, run STLS at each, and keep the support with
    the lowest loss (relative fit difference plus support ratio).

    Parameters
    ----------
    G : np.ndarray
        Reduced library matrix, Mp x prod(Jd).
    b : np.ndarray
        Target values, length Mp.
    lambs : list of float
        Threshold values to test.
    verbose : bool
        If True, print per-threshold diagnostics.

    Returns
    -------
    wStar : np.ndarray
        Coefficients on the surviving columns at the lowest loss.
    suppStar : np.ndarray
        Indices of the surviving columns into G.
    """
    Jtilde = G.shape[1] # G : M' x prod(Jd)
    w0, *_ = la.lstsq(G, b, rcond=None)
    Gw0 = G @ w0
    Gw0_norm = la.norm(Gw0)

    min_loss = np.inf
    wStar = None
    suppStar = None

    for i in range(len(lambs)):

        lamb = lambs[i]
        wLa, GLa, suppLa = STLS(G, b, lamb, w0)

        if wLa is None:
            loss = 1   # empty supprt
        else:
            GwLa = GLa @ wLa

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
            #print(f'support = {suppLa}')
            if wLa is not None:
                print(f'diff norm = {diffnorm}')
                print(f'supp ratio = {suppratio}')
                print(f'loss = {loss}')
            print(f'min loss = {min_loss}')

    return wStar, suppStar