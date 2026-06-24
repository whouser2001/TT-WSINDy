"""
Constructing the weak/strong feature tensor
"""
import numpy as np
from time import time
from scikit_tt.tensor_train import TT
from scipy.signal import correlate

class feature_tensor(TT):
    """
    Build a tensor train of feature cores from data and a list of
    candidate functions. Extends the TT (tensor train) class from
    scikit-tt.

    Parameters
    ----------
    X : np.ndarray
        Raw data, shape D x M (D system dimensions, M time points).
    f : list of callable
        Candidate functions fj : R -> R.
    phi : np.ndarray or None
        Discretization of a compactly supported test function, as a
        vector. None selects the strong form.

    Methods
    -------
    all_active_features()
        Active candidate functions in each dimension.
    support_key()
        Hashable key identifying the current support state.
    compute_weights(W)
        Per-mode importance weights from a coefficient tensor
        (independent of the threshold lamb).
    threshold_weights(weights, lamb)
        Threshold cached weights into a support mask.
    coarse_supp(W, lamb)
        Coarse support of W at threshold lamb.
    apply_supp(supp)
        Reduce the feature tensor to the features in supp.
    TT_PI(x, threshold)
        TT pseudoinverse regression against x, with optional SVD
        rank truncation.
    TT_STLS(x, lamb, threshold, weight_cache)
        Tensor-train sequential thresholding least squares.
    unscale(W)
        Undo the per-feature normalization of a coefficient estimate.
    supp_size(supp)
        Size of the support induced by a mask.
    """

    def __init__(self, X, f, threshold=0, phi=None, verbose=False, low_rank=True):
        """
        Parameters
        ----------
        X : np.ndarray
            Raw data, shape D x M (D system dimensions, M time points).
        f : list of callable
            Candidate functions fj : R -> R.
        phi : np.ndarray or None
            Discretization of a compactly supported test function, as a
            vector. None selects the strong form.
        verbose : bool
            If True, print progress/debugging information.
        low_rank : bool
            Construction method. The naive construction stores diagonal cores
            of shape (M, J, 1, M), i.e. O(D J M^2) memory, which is infeasible
            for large M. With low_rank=True (default) the same feature tensor is
            built directly in compressed form by a left-to-right SVD sweep over
            a small "carry" matrix: the internal bonds collapse to the true
            ranks (<= J^min(d, D-d)) and only the final (time) core scales with
            M, giving O(J^D + r M) memory -- linear in M. The two paths are
            numerically equivalent (to the SVD threshold). Set low_rank=False to
            force the original dense construction.

        Attributes (beyond those provided by TT)
        ----------
        verbose : bool
            If True, print progress/debugging information.
        snapshots : int
            Number of time points (M) in the data.
        Mp : int
            Length of the final (weak/strong) mode. Equals M for the
            strong form, or M - len(phi) + 1 for the weak form.
        feature_norms : np.ndarray
            Per-(candidate function, dimension) normalization factors,
            shape (J, D).
        supp_indices : list of np.ndarray
            supp_indices[d] maps each surviving feature in dimension d
            back to its index in the original candidate-function list,
            for d = 0, ..., D-1 (the weak/strong core is excluded).
            Initialized to the identity (all J features active per dim).
        """
        J = len(f)
        D,self.snapshots = X.shape if X.ndim > 1 else (1,X.size)

        M = self.snapshots

        # normalized basis evaluations B[j, d, :] = f_j(X[d, :]) / ||.||,
        # and the per-(feature, dim) norms (shared by both constructions)
        norms = np.zeros((J, D))
        B = np.zeros((J, D, M))
        for j in range(J):
            fX = np.vectorize(f[j])(X).astype(float)        # (D, M)
            for d in range(D):
                nrm = np.linalg.norm(fX[d, :]) if D > 1 else np.linalg.norm(fX)
                norms[j, d] = nrm if nrm > 0 else 1.0
            B[j] = fX / norms[j][:, None]

        if low_rank:
            # Build the feature tensor directly in compressed TT form. The naive
            # tensor is  Theta[j_0..j_{D-1}, m] = prod_d B[j_d, d, m]  with the
            # time index m carried on every bond (rank M). Here we carry a
            # compressed state C (shape r x M) and, at each mode, expand by the
            # next factor and re-compress with an SVD, so the bonds shrink to the
            # true ranks. Only the final (time) core scales with M.
            #
            # True ranks are bounded by J^D, so use this when J^D >= M to save
            # time & memory
            cores = []
            C = np.ones((1, M))                              # carry: (r, M)
            for d in range(D):
                r = C.shape[0]
                # E[(a,j), m] = C[a, m] * B[j, d, m]
                E = (C[:, None, :] * B[:, d, :][None, :, :]).reshape(r * J, M)
                U, s, Vt = np.linalg.svd(E, full_matrices=False)
                s0 = s[0] if s.size and s[0] > 0 else 1.0
                tol = (threshold if threshold > 0 else 1e-13) * s0
                k = max(int(np.sum(s > tol)), 1)
                cores.append(U[:, :k].reshape(r, J, 1, k))
                C = s[:k, None] * Vt[:k, :]                  # new carry: (k, M)

            # final (time) core: strong form keeps every snapshot; the weak form
            # convolves the carry with phi (this is the only M-sized object)
            if phi is None:
                cores.append(C.reshape(C.shape[0], M, 1, 1))
            else:
                Cw = correlate(C, np.expand_dims(phi, axis=0), mode='valid')
                cores.append(Cw.reshape(C.shape[0], Cw.shape[1], 1, 1))

            # cores are already compressed; avoid a second global rounding
            super().__init__(cores, threshold=0)
        else:
            # original dense construction: O(D J M^2) memory (diagonal cores)
            cores = [np.zeros([1, J, 1, M])] + \
                    [np.zeros([M, J, 1, M]) for _ in range(D - 1)]
            for j in range(J):
                cores[0][0, j, 0, :] = B[j, 0, :]
                for d in range(1, D):
                    for m in range(M):
                        cores[d][m, j, 0, m] = B[j, d, m]
            if phi is None:
                cores.append(np.eye(M).reshape(M, M, 1, 1))
            else:
                Iphi = correlate(np.eye(M), np.expand_dims(phi, axis=0),
                                 mode='valid')
                cores.append(Iphi.reshape(M, Iphi.shape[1], 1, 1))
            super().__init__(cores, threshold=threshold)

        # other metadata
        self.verbose = verbose
        self.Mp = M - len(phi) + 1 if phi is not None else M # final mode
        self.feature_norms = norms
        self.supp_indices = [np.arange(J) for _ in range(D)]
        
    def all_active_features(self):
        """
        Active candidate functions in each dimension, as indices into the
        original candidate-function list. Use this to recover features
        after TT_STLS.

        Returns
        -------
        active_features : list of np.ndarray
            active_features[d] holds the surviving feature indices in
            dimension d.
        """
        D = self.order - 1
        active_features = [self.supp_indices[d] for d in range(D)]
        return active_features
    
    def support_key(self):
        """
        Hashable key identifying the current support state.

        Returns
        -------
        key : tuple of tuple of int
            Per-dimension tuples of surviving original candidate-function
            indices. Two feature tensors at the same support share a key,
            so their weights can be cached and reused across thresholds.
        """
        D = self.order - 1
        return tuple(tuple(int(i) for i in self.supp_indices[d]) for d in range(D))

    def compute_weights(self, W):
        """
        Per-mode importance weights for a coefficient tensor.

        These are the quantities thresholded inside coarse_supp. They depend
        only on the support and the regression target (through W), not on the
        threshold lamb, so they can be cached per support and reused across
        the lambda sweep.

        Parameters
        ----------
        W : TT
            Coefficient tensor estimate, with the weak/strong core already
            contracted into the train.

        Returns
        -------
        weights : list of np.ndarray
            weights[d] holds the normalized weight of each feature in
            dimension d (scaled so the largest weight is 1).
        """
        D = self.order - 1
        Wcores, Wmodes, Wranks = W.cores, W.row_dims, W.ranks

        # Accumulate right density matrices
        DRs = [None]*(D-1)
        for d in range(D-2, -1, -1):
            DR = np.zeros((Wranks[d+1], Wranks[d+1]))
            for j in range(Wmodes[d+1]):
                Wdj = Wcores[d+1][:, j, :, :].squeeze(axis=1)
                DR += (Wdj @ Wdj.T) if d == D-2 else (Wdj @ DRs[d+1] @ Wdj.T)
            DRs[d] = DR

        weights = [None]*D
        DLm1 = None
        for d in range(D):

            DL = np.zeros((Wranks[d+1], Wranks[d+1]))
            w = np.zeros(Wmodes[d])

            for j in range(Wmodes[d]):

                Wdj = Wcores[d][:, j, :, :].squeeze(axis=1)
                S = (Wdj.T @ Wdj) if DLm1 is None else (Wdj.T @ DLm1 @ Wdj)
                DL += S
                w[j] = max(0.0, float(np.sum(S))) if d == D-1 \
                    else max(0.0, float(np.sum(S * DRs[d])))
                
            DLm1 = DL
            w /= w.max()
            weights[d] = w

        return weights

    def threshold_weights(self, weights, lamb):
        """
        Threshold cached weights into a per-dimension support mask.

        Parameters
        ----------
        weights : list of np.ndarray
            Normalized per-feature weights, as returned by compute_weights.
        lamb : float
            Threshold. A feature is kept where its weight exceeds lamb; if a
            dimension would be emptied, its strongest feature is kept.

        Returns
        -------
        supp : list of np.ndarray
            supp[d] is a boolean mask over the features in dimension d.
        """
        supp = [None]*len(weights)
        for d, w in enumerate(weights):
            keep = w > lamb
            if not keep.any() and w.size > 0:
                keep[int(np.argmax(w))] = True
            if self.verbose:
                print(f"  d={d}: weights={w}, LB={lamb:.3e}")
            supp[d] = keep
        return supp

    def coarse_supp(self, W, lamb):
        """
        Coarse support of W at threshold lamb.

        Convenience wrapper that computes the (lamb-independent) weights and
        thresholds them in one call. Prefer compute_weights / threshold_weights
        directly when caching weights across lambda values.

        Parameters
        ----------
        W : TT
            Coefficient tensor estimate.
        lamb : float
            Threshold.

        Returns
        -------
        supp : list of np.ndarray
            Per-dimension boolean support masks.
        """
        return self.threshold_weights(self.compute_weights(W), lamb)
                
    def apply_supp(self, supp):
        """
        Reduce the feature tensor to the features in supp.

        Slices the feature axis of each feature core, updates the index map
        (supp_indices) to keep only surviving features, and updates the
        corresponding row dimensions. The weak/strong core is left untouched.

        Parameters
        ----------
        supp : list of np.ndarray
            supp[d] is a boolean mask over the features in dimension d.
        """
        D = self.order - 1
        cores_prime = [None]*(D+1)
        cores_prime[D] = self.cores[D] # weak/strong core

        for d in range(D):

            # slce feature axis to only contain features in support
            cores_prime[d] = self.cores[d][:, supp[d], :, :]  

            # update index map: keep only suriving indices
            self.supp_indices[d] = self.supp_indices[d][supp[d]]

            # update row dim to new number of features
            self.row_dims[d] = sum(supp[d])
        
        self.cores = cores_prime


    def TT_PI(self, x, threshold=0):
        """
        TT pseudoinverse regression.

        Form the pseudoinverse of the feature tensor (optionally truncating
        its constituent SVDs by `threshold`), regress against x, and collapse
        the result into a coefficient tensor.

        Parameters
        ----------
        x : np.ndarray
            Target values to regress against, length self.Mp.
        threshold : float
            SVD truncation parameter. threshold=0 computes the exact
            pseudoinverse.

        Returns
        -------
        W : TT
            Coefficient tensor estimate (the weak/strong core has been
            contracted with x and collapsed into the train).

        Raises
        ------
        ValueError
            If x does not have length self.Mp.
        """
        if x.shape[0] != self.Mp:
            raise ValueError(
                f"Vector x must have length {self.Mp}, but has length {x.shape[0]}"
            )

        D = self.order - 1

        # self.pinv
        U, Sigma, V = self.svd(D, threshold=threshold,
                       ortho_l=True, ortho_r=True, overwrite=False)
        s = Sigma[0]
        Wcores = U.cores + V.cores
        Wcores[D] = np.tensordot(np.diag(np.reciprocal(Sigma)),
                                 Wcores[D], axes=(1,0))
        W = TT(Wcores)
        
        # contract vector with last core of W
        last = (W.cores[-1].reshape(W.ranks[-2], W.row_dims[-1])).dot(x.T).reshape(W.ranks[-2],1)

        # Collapse the final core into rest of tensor
        next_last = W.cores[-2]@last
        W = TT(W.cores[:-2] + [next_last]) # auto update attributes

        #print(f'DEBUG ranks: {W.ranks}')
        return W

    def TT_STLS(self, x, lamb, threshold=0, weight_cache=None):
        """
        Tensor-train sequential thresholding least squares (TT-STLS).

        Repeatedly estimate the coefficient tensor (TT_PI), threshold it to a
        coarse support, and reduce the feature tensor, until the support stops
        shrinking (or collapses to a single feature). The final estimate is
        recomputed on the converged support and unscaled.

        Parameters
        ----------
        x : np.ndarray
            Target values to regress against.
        lamb : float
            Thresholding parameter.
        threshold : float
            SVD truncation parameter passed to TT_PI.
        weight_cache : dict, optional
            Maps a support key (see support_key) to its precomputed weights,
            so the weights for a given support are computed only once across a
            lambda sweep. A fresh cache is used if none is given.

        Returns
        -------
        W : TT
            Coefficient tensor estimate on the converged support, unscaled by
            the feature norms.
        """
        if weight_cache is None:
            weight_cache = {}

        # initialize support
        J = self.row_dims[0] # row dims are (J,...,J,M')
        supp = [np.ones(J, dtype=bool) for _ in range(self.order - 1)]

        if self.verbose:
            print(f'Starting TT-STLS')
            print('----------------')
            print(f'Number of basis functions: {J}')
            print(f'Initial support size: {self.supp_size(supp)}')
            print(f'Number of time points: {self.snapshots}')
            print('----------------')
            st = time()

        i = 1
        while True:

            supp_prev = supp

            key = self.support_key()
            weights = weight_cache.get(key)

            if weights is None: 
                W = self.TT_PI(x, threshold)
                weights = self.compute_weights(W)
                weight_cache[key] = weights

            # compute & apply supp
            supp = self.threshold_weights(weights, lamb)
            self.apply_supp(supp)

            if self.verbose:
                print(f'Iteration {i}:')
                print(f'Support size: {self.supp_size(supp)}')
                print('----------------')

            # supp is montonically decreasing. So only need to compare to size of previous support
            if self.supp_size(supp) == self.supp_size(supp_prev) or \
                self.supp_size(supp) == 1:
                break
            i += 1

        if self.verbose:
            print(f'Finished TT-STLS in {i} iterations, with final support size {self.supp_size(supp)}')
            print(f'Time elapsed: {time() - st:.2f} seconds')

        # Recompute W on converged support and unscale
        W = self.TT_PI(x, threshold)
        W = self.unscale(W)

        return W
    
    def unscale(self, W):
        """
        Undo the per-feature normalization of a coefficient estimate.

        Each feature slice was scaled by its norm during construction; this
        rescales the corresponding coefficient slices back, using the index
        map to recover the original candidate function for each survivor.

        Parameters
        ----------
        W : TT
            Coefficient tensor estimate with normalized feature scaling.

        Returns
        -------
        W : TT
            The same tensor with the normalization undone.
        """
        for d in range(W.order):
            for j in range(W.row_dims[d]):
                orig_j = self.supp_indices[d][j]
                W.cores[d][:,j,:,:] /= self.feature_norms[orig_j,d]

        return W

    def supp_size(self, supp):
        """
        Size of the support induced by a mask.

        Parameters
        ----------
        supp : list of np.ndarray
            Per-dimension boolean support masks.

        Returns
        -------
        size : int
            Product over dimensions of the number of active features (each
            dimension counted as at least 1).
        """
        return np.prod([np.max([s.sum(),1]) for s in supp])
