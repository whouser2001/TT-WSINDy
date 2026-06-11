"""
Constructing the weak/strong feature tensor
"""
import numpy as np
from time import time
from scikit_tt.tensor_train import TT
from scipy.signal import correlate

class feature_tensor(TT):
    """
    Construct list of feature cores from data 
    and candidate functions. Extends the TT 
    (tensor train) class from scikit-tt

    Parameters
    ----------
    X : np.array
        Raw data with shape D x M
    f : list
        List of candidate functions
        fj : R -> R
    phi : np.array
        discretization of compactly supported test function
        as a vector. If strong form, phi = None

    Methods
    -------
    all_active_features
        Return list of active basis features in each dimension
    coarse_supp(W, lambda)
        Compute coarse support of W
    apply_supp(supp)
        Given coarse support, reduce feature tensor to only those
        features
    TT_PI(x, threshold)
        Perform TT pseudoinverse regression against x, and truncate
        constituent SVDs according to threshold
    TT_STLS(x, lambda, threshold)
        Tensor-train sequential thresholding least squares
    unscale(W)
        Utility function to unscale coefficient estimate.
        For numerical stability
    eject
        Return flattened feature tensor and space of induced 
        feature functions
    supp_size(supp)
        Utility function to get the size of the induced support,
        given a basis

    References
    ----------
    """

    def __init__(self, X, f, threshold=0, phi=None, verbose=False):
        """
        Parameters
        ----------
        X : np.array
            Raw data with shape D x M
        f : list
            List of candidate functions
            fj : R -> R
        phi : np.array
            discretization of compactly supported test function
            as a vector. If strong form, phi = None
        verbose : bool
            If True, print out debugging statements during construction

        Attributes (in addition to TT attributes)
        ----------
        supp_indices : list of np.array
            supp_indices[d] maps shrunk feature index back to original
            for d = 0, ..., D-2 (excluding weak/strong core)
            initialized to identity, all J features active in each dimension
        verbose : bool
            If True, print out debugging statements during construction
        snapshots : int
            number of time points in data
        dims : int
            number of dimensions in data
        feature_norms : np.array
            tracking of per-slice norms
        """
        J = len(f)
        D,self.snapshots = X.shape

        M = self.snapshots

        # initialize blank cores
        cores = [np.zeros([1, J, 1, M])] + [np.zeros([M, J, 1, M]) for _ in range(D-1)]

        # norm tracking
        norms = np.zeros((J,D))

        # fill in cores with candidate functions
        for j in range(J):

            # evaluate dataset on fj
            fX = np.vectorize(f[j])(X)
            fX = fX.astype(float)

            # Perform per-dim normalization and store norms for later
            for d in range(D):
                nrm = np.linalg.norm(fX[d,:])
                norms[j,d] = nrm if nrm > 0 else 1.0
                fX[d,:] /= norms[j,d]

            # fill in slice of first core
            cores[0][0, j, 0, :] = fX[0, :]

            # fill in other cores
            for d in range(1,D):
                for m in range(M):
                    cores[d][m, j, 0, m] = fX[d,m]

        # append strong or weak core
        if phi is None: cores.append(np.eye(M).reshape(M,M,1,1))
        else: 
            Iphi = correlate(
                np.eye(M), np.expand_dims(phi, axis=0), mode='valid'
            )
            cores.append(Iphi.reshape(M,Iphi.shape[1],1,1))

        # initialize TT
        super().__init__(cores, threshold=threshold)

        # index tracking
        # supp_indices[d] maps shrunk feature index back to original
        # for d = 0, ..., D-2 (excluding weak/strong core)
        # initialized to identity, all J features active in each dimension
        self.supp_indices = [np.arange(J) for _ in range(D)]

        # other metadata
        self.verbose = verbose
        self.Mp = M - len(phi) + 1 if phi is not None else M # final mode
        self.feature_norms = norms
        
    def all_active_features(self):
        """
        Return list of all active features in each dimension, as lists of
        indices into original feature list. Use this to recover features
        after TT_STLS.

        Returns
        -------
        active_features : list of np.array
            active_features[d] is array of indices of active features in dimension d
        """
        D = self.order - 1
        active_features = [self.supp_indices[d] for d in range(D)]
        return active_features
    
    def coarse_supp(self, W, lamb):
        """
        Compute coarse support of W

        Parameters
        ----------
        W : TT
            Coefficient tensor estimate
            modes   J_1 x ... x J_D or
                    D_1 x ... x D_J
            ranks   (M_1, ..., M_D)
            W has had the core corresponding to the weak core
            contracting into the rest of the tensor train
        lamb : float
            Thresholding parameter
        bound : float
            ||x||_2/||Theta||_2

        Returns
        -------
        supp : list of np.array
            supp[d] is boolean array of length J_d, indicating which features
            are in the support for dimension d
        """
        D = self.order - 1
        supp = [None]*(D)
        Wcores = W.cores
        Wmodes = W.row_dims
        Wranks = W.ranks
        
        # Accumulate right density matrices
        DRs = [None]*(D-1) #D_R^1, ..., D_R^(D-1), D_R^(D)

        for d in range(D-2,-1,-1):

            DR = np.zeros((Wranks[d+1], Wranks[d+1])) # running sum
            for j in range(Wmodes[d+1]):

                Wdj = Wcores[d+1][:,j,:,:].squeeze(axis=1)
                if d == D-2: DR += Wdj @ Wdj.T
                else: DR += Wdj @ DRs[d+1] @ Wdj.T

            DRs[d] = DR

        # Compute left density matrices and traces in unison
        DLm1 = None
        for d in range(D):
            
            DL = np.zeros((Wranks[d+1], Wranks[d+1]))
            weights = np.zeros(Wmodes[d])

            supp[d] = np.zeros(Wmodes[d], dtype=bool)
            for j in range(Wmodes[d]):

                Wdj = Wcores[d][:,j,:,:].squeeze(axis=1)
                S = np.zeros((Wranks[d+1],Wranks[d+1]))
                if DLm1 is None: S += Wdj.T @ Wdj
                else: S += Wdj.T @ DLm1 @ Wdj
                DL += S

                if d == D - 1: 
                    weights[j] = max(0.0,float(np.sum(S)))
                else: 
                    weights[j] = max(0.0, float(np.sum(S * DRs[d])))

            DLm1 = DL

            # Scalar threshold against the band
            LB = lamb
            #weights **= 2
            weights /= weights.max()
            keep = (weights > LB)

            # Safeguard: never empty a dimension — keep the argmax
            if not keep.any() and Wmodes[d] > 0:
                keep[int(np.argmax(weights))] = True
            supp[d] = keep

            if self.verbose:
                print(f"  d={d}: weights={weights}, LB={LB:.3e}")

        return supp
                

    def apply_supp(self, supp):
        """
        Given support of W, reduce feature tensor to only those features.

        Parameters
        ----------
        supp : list of list
            supp[d] is boolean array of length J_d, indicating which features
            are in the support for dimension d
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
        Find pseudoinverse of feature tensor, regress
        against x and threshold down ranks, if applicable.

        Parameters
        ----------
        x : np.array
            Data to regress against

        Returns
        -------
        W : TT
            Pseudoinverse of feature tensor
        smax : float
            largest singular value of self, which
            is also the 2-norm of the right-unfolding

        Raises
        ------
        ValueError
            if vector x does not have length self.shape[-1]
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
        return W

    def TT_STLS(self, x, lamb, threshold=0):
        """
        Tensor-train squential thresholding least squares (TT-STLS)

        Parameters
        ----------
        x : np.array
            Data to regress against
        lamb : float
            Thresholding parameter

        Returns
        -------
        W : TT
            Coefficient tensor estimate after TT-STLS
        TODO
        """

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


        # iteratively apply STLS step until support converges
        i = 1
        while True:

            supp_prev = supp
            
            # compute coefficient estimate
            W = self.TT_PI(x, threshold)

            # Compute & apply supp
            supp = self.coarse_supp(W, lamb)
            self.apply_supp(supp)

            if self.verbose:
                print(f'Iteration {i}:')
                print(f'Support size: {self.supp_size(supp)}')
                print('----------------')

            # supp is montonically decreasing. So only need to compare
            #   to size of previous support
            if self.supp_size(supp) == self.supp_size(supp_prev) or \
                self.supp_size(supp) == 1:
                break
            i += 1

        if self.verbose:
            print(f'Finished TT-STLS in {i} iterations, with final support size {self.supp_size(supp)}')
            print(f'Time elapsed: {time() - st:.2f} seconds')
            #print(supp)

        # Recompute W on converged support and unscale
        W = self.TT_PI(x, threshold)
        W = self.unscale(W)

        return W
    
    def unscale(self, W):
        """
        Unscale each core slice of W by the norms of the corresponding features.

        Parameters
        ----------
        W : TT
            Coefficient tensor estimate, with cores scaled by feature norms

        Returns
        -------
        W_unscaled : TT
            Coefficient tensor estimate, with cores unscaled by feature norms
        """
        for d in range(W.order):
            for j in range(W.row_dims[d]):
                orig_j = self.supp_indices[d][j]
                W.cores[d][:,j,:,:] *= self.feature_norms[orig_j,d]

        return W

    def supp_size(self, supp):
        """
        Utility function, gives support size
        """
        return np.prod([np.max([s.sum(),1]) for s in supp])
