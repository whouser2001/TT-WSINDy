"""
Constructing the weak/strong feature tensor
"""
import numpy as np
from time import time
from scikit_tt.tensor_train import TT
from scipy.signal import correlate
from ttutils import truncated_svd

class feature_tensor(TT):
    """
    Tensor train of feature cores from data and a list of candidate functions.
    Extends the TT (tensor train) class from scikit-tt.

    Methods
    -------
    all_active_features()
        Active candidate functions in each dimension.
    support_key()
        Hashable key identifying the current support state.
    compute_weights(W)
        Per-mode importance weights from a coefficient tensor.
    threshold_weights(weights, lamb)
        Threshold cached weights into a support mask.
    apply_supp(supp)
        Reduce the feature tensor to the features in supp.
    TT_PI(x, factors)
        TT pseudoinverse regression against x.
    TT_STLS(x, lamb, weight_cache)
        Tensor-train sequential thresholding least squares.
    supp_size(supp)
        Size of the support induced by a mask.
    """

    def __init__(self, X, f, threshold=0.0, phi=None, verbose=False, low_rank=True,
                 n_traj=1, construction='dimension_major'):
        """
        Parameters
        ----------
        X : np.ndarray
            Raw data, shape D x M. With n_traj > 1, the columns hold n_traj
            equal-length trajectories concatenated along time.
        f : list of callable
            Candidate functions fj : R -> R.
        threshold : float
            Truncation parameter applied to the matrix SVDs in TT-PI.
            threshold=0.0 computes pseudoinverses exactly.
        phi : np.ndarray or None
            Discretization of a compactly supported test function, as a
            vector. None selects the strong form.
        verbose : bool
            If True, print progress/debugging information.
        low_rank : bool
            Build the train directly in compressed form, by a left-to-right
            SVD sweep over a small carry matrix. Preferred for large M, and
            when the dense feature tensor does not fit in memory.
        n_traj : int
            Number of independent trajectories concatenated along the time
            axis of X, each of length M / n_traj. Only the weak-form
            convolution is trajectory-aware: phi is correlated within each
            trajectory separately, so no row of the final mode straddles a
            boundary.
        construction : str
            Which set of candidates the train enumerates.

            'dimension_major' (default): one mode per state dimension, of size
            J, giving J^D candidates. Two functions can never act on the same
            dimension.

            'function_major': one mode per non-constant candidate function, of
            size D+1, giving (D+1)^(J-1) candidates. Mode j picks the dimension
            f_j acts on, or the extra slot meaning absent. Same-dimension
            products are in the span; a single function cannot be reused.
            Requires low_rank=True.

        Attributes (beyond those provided by TT)
        ----------
        verbose : bool
            If True, print progress/debugging information.
        snapshots : int
            Total number of time points (M) in the data, over all
            trajectories.
        n_traj : int
            Number of trajectories concatenated along the time axis.
        Mp : int
            Length of the final (weak/strong) mode. Equals M for the
            strong form, or n_traj * (M/n_traj - len(phi) + 1) for the
            weak form.
        supp_indices : list of np.ndarray
            supp_indices[d] maps each surviving feature in dimension d
            back to its index in the original candidate-function list,
            for d = 0, ..., D-1 (the weak/strong core is excluded).
            Initialized to the identity (all J features active per dim).
        threshold : float
            Truncation parameter applied to the matrix SVDs in TT-PI.

        Raises
        ------
        ValueError
            If the number of time points is not a multiple of n_traj, or if an
            invalid construction string is given.
        NotImplementedError
            If construction='function_major' is used without low_rank.
        """
        J = len(f)
        D,self.snapshots = X.shape if X.ndim > 1 else (1,X.size)
        M = self.snapshots

        if M % n_traj != 0:
            raise ValueError(
                f"Number of time points ({M}) must be a multiple of "
                f"n_traj ({n_traj})"
            )
        m_traj = M // n_traj                             # points per trajectory

        # basis evaluations B[j, d, :] = f_j(X[d, :])
        B = np.zeros((J, D, M))
        for j in range(J):
            B[j] = np.vectorize(f[j])(X).astype(float)        # (D, M)

        # dimension or function major construction
        if construction == 'dimension_major':
            slices = [B[:, d, :] for d in range(D)]
        elif construction == 'function_major':
            slices = [np.vstack([B[j], np.ones((1, M))]) for j in range(1, J)]
        else:
            raise ValueError(
                f"construction must be 'dimension_major' or 'function_major' "
                f"(got {construction!r})"
            )
        mode_sizes = [S.shape[0] for S in slices]

        if low_rank:
            # Build directly in compressed form
            cores = []
            C = np.ones((1, M))                              # carry: (r, M)
            for i, S in enumerate(slices):
                r, Ji = C.shape[0], mode_sizes[i]
                # E[(a,j), m] = C[a, m] * S[j, m]
                E = (C[:, None, :] * S[None, :, :]).reshape(r * Ji, M)
                if i < len(slices) - 1:
                    # interior mode: the bond rank is exact, so nothing can be
                    # truncated and a full SVD is fine
                    U, s, Vt = np.linalg.svd(E, full_matrices=False)
                    s0 = s[0] if s.size and s[0] > 0 else 1.0
                    tol = (threshold if threshold > 0 else 1e-13) * s0
                    k = max(int(np.sum(s > tol)), 1)
                    cores.append(U[:, :k].reshape(r, Ji, 1, k))
                    C = s[:k, None] * Vt[:k, :]              # new carry: (k, M)
                else:
                    # convolve with the test function first, which sharply drops
                    # the rank, then extract the last two cores
                    Ec = E if phi is None else \
                        correlate(E.reshape(E.shape[0], n_traj, M // n_traj),
                            phi[None, None, :], mode='valid').reshape(
                                E.shape[0], -1
                            )                                  # (rJ, Mp)
                    U, s, Vt = truncated_svd(Ec, threshold)
                    cores.append(U.reshape(r, Ji, 1, U.shape[1]))       # feature core
                    cores.append((s[:, None] * Vt).reshape(            # time core
                        U.shape[1], Ec.shape[1], 1, 1))

            super().__init__(cores, threshold=0)
        elif construction != 'dimension_major':
            # the dense path builds uniform (M, J, 1, M) cores directly, and is
            # only implemented for the default construction
            raise NotImplementedError(
                "construction='function_major' requires low_rank=True"
            )
        else:
            # Original construction
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
                # per-trajectory convolution operator, repeated block-diagonally
                # so that no column mixes two trajectories
                Iphi = correlate(np.eye(m_traj), np.expand_dims(phi, axis=0),
                                 mode='valid')
                if n_traj > 1:
                    Iphi = np.kron(np.eye(n_traj), Iphi)
                cores.append(Iphi.reshape(M, Iphi.shape[1], 1, 1))
            super().__init__(cores, threshold=threshold)

        # other metadata
        self.verbose = verbose
        self.n_traj = n_traj
        self.Mp = n_traj * (m_traj - len(phi) + 1) if phi is not None else M # final mode
        self.construction = construction
        self.supp_indices = [np.arange(sz) for sz in mode_sizes]
        self.threshold = threshold
        
    def all_active_features(self):
        """
        Active candidate functions in each dimension, as indices into the
        original candidate-function list.

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

        These are the quantities thresholded by threshold_weights. They depend
        only on the support and on W, not on the threshold, so they can be
        cached per support and reused across a lambda sweep.

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
                
    def apply_supp(self, supp):
        """
        Reduce the feature tensor to the features in supp.

        Slices the feature axis of each feature core, and updates the index map
        (supp_indices) and the row dimensions. The weak/strong core is left
        untouched.

        Parameters
        ----------
        supp : list of np.ndarray
            supp[d] is a boolean mask over the features in dimension d.
        """
        D = self.order - 1
        cores_prime = [None]*(D+1)
        cores_prime[D] = self.cores[D] # weak/strong core

        for d in range(D):

            # slice feature axis to only contain features in support
            cores_prime[d] = self.cores[d][:, supp[d], :, :]

            # update index map: keep only surviving indices
            self.supp_indices[d] = self.supp_indices[d][supp[d]]

            # update row dim to new number of features
            self.row_dims[d] = sum(supp[d])
        
        self.cores = cores_prime

    def TT_PI(self, x, factors=None):
        """
        TT pseudoinverse regression.

        Form the pseudoinverse of the feature tensor (optionally truncating
        its constituent SVDs by self.threshold), regress against x, and
        collapse the result into a coefficient tensor.

        Parameters
        ----------
        x : np.ndarray
            Target values to regress against, length self.Mp.
        factors : tuple (TT, np.ndarray, TT), optional
            Precomputed pseudoinverse SVD factors. When given, the
            target-independent global SVD is skipped and reused. The factor
            cores are copied, so the returned W can be mutated without
            corrupting the shared factors.

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

        if factors is None:
            U, Sigma, V = self.svd(D, threshold=self.threshold,
                           ortho_l=True, ortho_r=True, overwrite=False)
            Wcores = U.cores + V.cores
        else:
            U, Sigma, V = factors
            Wcores = [c.copy() for c in U.cores] + [c.copy() for c in V.cores]
        Wcores[D] = np.tensordot(np.diag(np.reciprocal(Sigma)),
                                 Wcores[D], axes=(1,0))
        W = TT(Wcores)
        
        # contract vector with last core of W
        last = (W.cores[-1].reshape(W.ranks[-2], W.row_dims[-1])).dot(x.T).reshape(W.ranks[-2],1)

        # Collapse the final core into rest of tensor
        next_last = W.cores[-2]@last
        W = TT(W.cores[:-2] + [next_last])

        return W

    def TT_STLS(self, x, lamb, weight_cache=None):
        """
        Tensor-train sequential thresholding least squares (TT-STLS).

        Repeatedly estimate the coefficient tensor (TT_PI), threshold it to a
        coarse support, and reduce the feature tensor, until the support stops
        shrinking (or collapses to a single feature). The final estimate is
        recomputed on the converged support.

        Parameters
        ----------
        x : np.ndarray
            Target values to regress against.
        lamb : float
            Thresholding parameter.
        weight_cache : dict, optional
            Maps a support key (see support_key) to its precomputed weights,
            so the weights for a given support are computed only once across a
            lambda sweep. A fresh cache is used if none is given.

        Returns
        -------
        W : TT
            Coefficient tensor estimate on the converged support.
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
                W = self.TT_PI(x)
                weights = self.compute_weights(W)
                weight_cache[key] = weights

            # compute & apply supp
            supp = self.threshold_weights(weights, lamb)
            self.apply_supp(supp)

            if self.verbose:
                print(f'Iteration {i}:')
                print(f'Support size: {self.supp_size(supp)}')
                print('----------------')

            # supp is monotonically decreasing, so comparing sizes suffices
            if self.supp_size(supp) == self.supp_size(supp_prev) or \
                self.supp_size(supp) == 1:
                break
            i += 1

        if self.verbose:
            print(f'Finished TT-STLS in {i} iterations, with final support size {self.supp_size(supp)}')
            print(f'Time elapsed: {time() - st:.2f} seconds')

        # Recompute W on converged support
        W = self.TT_PI(x)
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
