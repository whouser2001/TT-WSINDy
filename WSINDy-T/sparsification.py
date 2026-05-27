"""
TT-STLS and TT-MSTLS
"""
import numpy as np
import numpy.linalg as la
from scikit_tt.tensor_train import TT

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

def TT_MSTLS(Theta, x, lambs, eps):
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
