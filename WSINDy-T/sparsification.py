"""
TT-MSTLS, TT-STLS, and associated helper methods
"""
import numpy as np
import numpy.linalg as la
from scikit_tt.tensor_train import TT

def coarse_supp(W, lamb):
    """
    Use coefficient tensor W and thresholding parameter
    lamb to reduce Theta

    Parameters
    ----------
    TODO
    """
    D = W.order
    supp = [None]*(D)
    for d in range(D):
        core = np.abs(W.cores[d])
        supp[d] = np.all(
            core >= lamb and core <= lamb**-1,
            axis=1
        )
    return supp

def apply_supp(Theta, supp):
    """
    TODO
    """
    # Use supp to reduce Theta
    D = Theta.order
    Tp_cores = [None]*(D)
    Tp_cores[D-1] = Theta.cores[D-1] #weak core
    for d in range(D - 1):
        # R_d-1 x len(supp[d]) x 1 x R_d
        Tp_cores[d] = Theta.cores[d][:, supp[d], :, :]
    Tp = TT(Tp_cores)

    return Tp

def TT_regress(Theta, x, threshold=0.0):
    """
    compute pseudoinverse of feature matrix, apply x
    and threshold

    Parameters
    ----------
    TODO
    """
    D = Theta.order
    W = Theta.pinv(D-1, threshold=threshold) #D-1 is dim of system
    W.cores[D-1] = (
        W.cores[D-1].reshape([W.ranks[D-1], Theta.shape[D-1]])
    ).dot(x).reshape(W.ranks[D-1], 1, 1, 1)
    W.row_dims[D-1] = 1
    
    return W

def TT_STLS(Theta, x, lamb, eps):
    """
    Tensor train sequential thresholding least squares
    (TT-STLS)

    Parameters
    ----------
    Theta : scikit_tt.tensor_train.TT
        feature tensor
        modes   J x ... x J x M, or
                D x ... x D x M
        ranks   (M, ..., M)
                
    x : np.array
        LHS vector, with size M
    lambda : float
        thresholding parameter
    eps : float
        truncation parameter for TT regression step

    Returns
    -------
    W : scikit_tt.tensor_train.TT
        coefficient tensor estimate
        modes   J x ... x J or
                D x ... x D
        ranks   (M, ..., M)
    W0 : scikit_tt.tensor_train.TT
        initial tensor estimate (for loss function)
        modes   J x ... x J or
                D x ... x D
        ranks   (M, ..., M)
    """

    # Compute initial coefficient estimate and support
    Psi = Theta
    W0 = TT_regress(Psi, x)
    Sl = 0
    Slp1 = coarse_supp(W0, lamb)
    M = Theta.modes[-1]

    # TODO size condition, as well
    #   maybe write mask data structure??, that stores size
    #   and has an apply/hadamard function
    # TODO reduce the size of the tensor problem, both via
    #   truncation and masking.
    while sum(len(s) for s in Slp1) <= \
        sum(len(s) for s in Sl): #TODO revise this condition
        Sl = Slp1
        Psi = apply_supp(Psi, Sl)
        W = TT_regress(Psi, x, threshold=eps)
        Slp1 = coarse_supp(W, lamb)

    return W, W0

def aux_loss(W, W0, Theta, J, D):
    """
    TODO
    """
    return NotImplementedError

def STLS(A, b, lamb):
    """
    Sequential thresholding least squares (STLS)

    #TODO cite seth (WSINDy for PDEs code base,
    MSTLSiterate)
    """
    max_its = A.shape[1]
    # TODO update with more specific bounds,
    #   more Seth's code and Dan's paper



    return NotImplementedError

def TT_MSTLS(Theta, x, lambs, L, eps):
    """
    TODO
    """
    Lmin = np.inf
    Wstar = None
    for lamb in lambs:
        W, W0 = TT_STLS(Theta, x, lamb, eps)
        # Right-unfold W
        # (assuming W is somewhat small via TT-STLS)
        Wfull = W.squeeze().full()
        W = Wfull.reshape(-1, Wfull.shape[-1])
        W = STLS(W, x, lamb)
        Llamb = NotImplementedError
        if Llamb < Lmin:
            Lmin = Llamb
            Wstar = W

    return Wstar
