"""
Utility functions for TT-WSINDy tensor computations
"""
import numpy as np
import numpy.linalg as la
import copy
from scikit_tt.tensor_train import TT

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

def mask_coeffs(W, supp):
    """
    Restrict a full-support coefficient TT to a product support by zeroing the
    feature slices that fall outside the per-dimension masks.

    Zeroing feature slice j in dimension d kills every contraction term that
    uses feature j in that dimension, so the result equals W on the product
    support (j_0,...,j_{D-1} with each j_d kept) and 0 elsewhere. Used by
    one-pass TT-MSTLS to score a candidate support without re-solving.

    Parameters
    ----------
    W : TT
        Full-support coefficient tensor (row_dims all J).
    supp : list of np.ndarray
        Per-dimension boolean masks; supp[d] has length W.row_dims[d].

    Returns
    -------
    W_masked : TT
        Copy of W with out-of-support feature slices set to zero; same shape.
    """
    cores = []
    for d in range(W.order):
        core = W.cores[d].copy()
        core[:, ~supp[d], :, :] = 0.0
        cores.append(core)
    return TT(cores)

def truncated_svd(A, threshold, small=700, n_oversamples=12, n_iter=2):
    """
    Thin SVD keeping the singular triplets with s > tol (= rel*s[0]), where
    rel = threshold (or 1e-13 if threshold==0).

    For a tall/wide matrix whose effective rank is far below min(A.shape) -- as
    happens at the weak feature tensor's time-core bond, where convolving with
    the test function collapses the rank -- a full SVD wastes almost all its
    work. This uses a randomized range finder [1] with an adaptive target rank.

    Parameters
    ----------
    A : np.array
        Matrix to be SVDed
    threshold : float
        SVD thresholding parameter
    small : int
        Rough size at which full SVD is computationally preferable
    n_oversamples : int
        Small number of Monte Carlo oversamples. Drastically mproves expected 
        accuracy of the SVD.
    n_iter : int
        Number of QR iterations per rank searched over.
    
    Returns 
    -------
    U : np.array
        Left-orthonormal columns
    s : np.array
        diagonal entries of Sigma
    Vt : np.array
        Right-orthonormal columns

    References
    ----------
    .. [1] N. Halko, P. G. Martinsson, and J. A. Tropp, "Finding Structure with
            Randomness: Probabilistic Algorithms for Constructing Approximate
            Matrix Decompositions", SIAM Review, 53 (2011) pp. 217-288, 
            https://doi.org/10.1137/090771806
    """
    m, n = A.shape
    p = min(m, n)
    rel = threshold if threshold > 0 else 1e-13

    if p <= small:                                    # full SVD already cheap
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
    else:
        rng = np.random.default_rng(0)
        target = 256
        while True:
            ell = min(target + n_oversamples, p)
            Q, _ = np.linalg.qr(A @ rng.standard_normal((n, ell)))
            for _ in range(n_iter):                   # power iters (sharpen gap)
                Q, _ = np.linalg.qr(A @ (A.T @ Q))
            B = Q.T @ A                               # (ell, n), small
            Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
            U = Q @ Ub
            s0 = s[0] if s.size and s[0] > 0 else 1.0
            if ell >= p or s[-1] <= rel * s0:         # captured all significant
                break
            target = min(target * 2, p)               # else widen and retry

    s0 = s[0] if s.size and s[0] > 0 else 1.0
    k = max(int(np.sum(s > rel * s0)), 1)
    return U[:, :k], s[:k], Vt[:k]