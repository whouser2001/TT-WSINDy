"""
Utility functions for TT-WSINDy tensor computations
"""
import numpy as np
import numpy.linalg as la
import copy
from scikit_tt.tensor_train import TT
from feature_tensor import feature_tensor

def W_contract(W, Theta):
    """
    Helper method to fully contract a coefficient estimate W
    into the feature tensor Theta, to produce a vector x in R^Mp.

    Superior to calling W.tensordot directly, as that method
    stores prohibitively large (O(M^4)) intermediate tensors

    Parameters
    ----------
    W : TT
        coefficient tensor, with shape J x ... x J
    Theta : TT
        feature tensor, with shape J x ... x J x Mp

    Returns
    -------
    x : np.array
        contracted vector in R^Mp
    """

    E = np.ones((1,1))                      # intermediate matrix

    # Accumulate E
    for d in range(W.order):
        Wd = W.cores[d][:,:,0,:]            # col dims all = 1
        Thetad = Theta.cores[d][:,:,0,:]    # ""
        E = np.tensordot(E, Thetad, axes = ([1], [0]))
        E = np.tensordot(Wd, E, axes = ([0,1], [0,1]))


    tail = Theta.cores[-1][:,:,0,:]
    return (E @ tail[:,:,0]).ravel()

def embed_full(W, active_features, full_dims):
    """
    Helper method to embed a reduced TT coefficient tensor back into the 
    full feature space. For compatibility with W0 in tensor loss function

    Parameters
    ----------
    W : TT
        Reduced coefficient tensor. Cores have shape
        (r_d, len(active_features[d]), 1, r_{d+1}).
    active_features : list of np.array
        active_features[d] holds the original feature indices surviving in
        dimension d (i.e. feature_tensor.all_active_features()).
    full_dims : list of int
        Original number of features per dimension (e.g. W0.row_dims).

    Returns
    -------
    W_full : TT
        Coefficient tensor in the full space, zero on thresholded features.
        row_dims == full_dims, ranks unchanged from W.

    Raises
    ------
    ValueError
        If W.order != len(active_features) or W.order != len(full_dims)
    ValueError
        If active_features doesn't track with the shape of W cores, in
        any dimension.
    """
    if W.order != len(active_features) or W.order != len(full_dims):
        raise ValueError("active_features and full_dims must have length W.order")

    cores_full = []
    for d in range(W.order):
        r_left, n_d, c_d, r_right = W.cores[d].shape
        if n_d != len(active_features[d]):
            raise ValueError(
                f"dim {d}: core has {n_d} features but "
                f"{len(active_features[d])} active indices"
            )
        core = np.zeros((r_left, full_dims[d], c_d, r_right))
        core[:, active_features[d], :, :] = W.cores[d]
        cores_full.append(core)

    return TT(cores_full)