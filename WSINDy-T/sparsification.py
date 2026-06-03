"""
Matrix STLS and TT-MSTLS
"""
import numpy as np
import numpy.linalg as la
import copy
from scikit_tt.tensor_train import TT
from feature_tensor import feature_tensor

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

    # Compute W - W0, and fully contract into Theta.
    # This should output a length Mp vector
    contr = Theta.tensordot(
        other=W - W0, num_axes=W.order, mode='first-last'
    )

    return np.linalg.norm(contr)/W0Theta_norm + supp_ratio

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

def TT_MSTLS(Theta, x, lambs, total_size):
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
    Wstar : TT
        optimal coefficient tensor wrt coarse TT-STLS
        algorithm
    """

    # Compute initial coefficient estimate, and
    # norm for loss.
    W0 = Theta.unscale(Theta.TT_PI(x))
    W0Theta_norm = np.linalg.norm(
        Theta.tensordot(other=W0, num_axes=W0.order, mode='first-last')
    )

    # track argmin
    min_loss = np.inf
    Wstar = None

    # Iterate over threshold values
    for i in range(len(lambs)):

        lamb = lambs[i]
        ThetaLa = copy.deepcopy(Theta)

        # TT-STLS, get coeffs + support
        WLa = ThetaLa.TT_STLS(x, lamb)
        suppLa = ThetaLa.all_active_features()

        # Cast W back up to full size ([J]^D), for loss 
        WLa = embed_full(WLa, suppLa, W0.row_dims)
        suppLa_ratio = np.prod([
            s.size for s in suppLa
        ])/total_size

        loss = tensor_loss(WLa, W0, Theta, suppLa_ratio, W0Theta_norm)
        if loss < min_loss:
            loss = min_loss
            Wstar = WLa

    return Wstar

def STLS(A, b, lamb, w_LS):
    """
    Sequential thresholding least squares (STLS)

    #TODO cite seth (WSINDy for PDEs code base,
    MSTLSiterate)
    """
    max_its = A.shape[1]

    bound = la.norm(b)/la.norm(A, dim=0)
    LB = lamb*np.max([1,bound])
    UB = (1/lamb)*np.min([1,bound])

    i = 0
    w = np.abs(w_LS.copy())
    
    nonzero_inds = None
    while i <= max_its:
        ib_inds = np.where(w >= LB & w <= UB)
        oob_inds = np.where(w < LB | w > UB)

        if np.all(np.equal(ib_inds, nonzero_inds)) or not ib_inds.shape[0]:
            break

        w[ib_inds] = la.lstsq(A[:,ib_inds],b).solution
        w[oob_inds] = 0
        nonzero_inds = ib_inds
        i += 1

    return w

def matrix_loss(W, W0, Theta_flat, supp_ratio, W0Theta_norm):
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
    return np.linalg.norm((W - W0)@Theta_flat)/W0Theta_norm + supp_ratio

def MSTLS():

    return NotImplementedError
