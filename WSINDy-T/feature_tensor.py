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
    TODO

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
        threshold : float
            Threshold for pseudoinverse of feature tensor
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
        """
        J = len(f)
        D,self.snapshots = X.shape

        M = self.snapshots

        # initialize blank cores
        cores = [np.zeros([1, J, 1, M])] + [np.zeros([M, J, 1, M]) for _ in range(D-1)]

        # fill in cores with candidate functions
        for j in range(J):

            # evaluate dataset on fj
            fX = np.vectorize(f[j])(X)

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
        self.threshold = threshold
        self.verbose = verbose
        self.Mp = M - len(phi) + 1 if phi is not None else M # final mode
        
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
        Compute coarse support of W, by thresholding each core.

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

            supp[d] = np.zeros(Wmodes[d], dtype=bool)
            for j in range(Wmodes[d]):

                Wdj = Wcores[d][:,j,:,:].squeeze(axis=1)
                S = np.zeros((Wranks[d+1],Wranks[d+1]))
                if DLm1 is None: S += Wdj.T @ Wdj
                else: S += Wdj.T @ DLm1 @ Wdj
                DL += S

                if d == D - 1: A = S.copy()
                else: A = S * DRs[d]

                # threshold intermediate matrix A
                A[(np.abs(A) <= lamb) | (np.abs(A) >= 1/lamb)] = 0
                if np.sum(A) > 0: supp[d][j] = 1

            DLm1 = DL

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


    def TT_PI(self, x):
        """
        Find pseudoinverse of feature tensor, regress
        against x and threshold down ranks, if applicable.

        Computes W^T

        Parameters
        ----------
        x : np.array
            Data to regress against

        Returns
        -------
        W : TT
            Pseudoinverse of feature tensor

        Raises
        ------
        ValueError
            if vector x does not have length self.shape[-1]
        """
        D = self.order - 1
        W = self.pinv(D, threshold=self.threshold) #D is dim of system

        if x.shape[0] != self.Mp:
            raise ValueError(
                f"Vector x must have length {self.Mp}, but has length {x.shape[0]}"
            )
        
        # contract vector with last core of W
        W.cores[-1] = (W.cores[-1].reshape(W.ranks[-2], W.row_dims[-1])).dot(x.T).reshape(W.ranks[-2],1)
        W.row_dims[-1] = 1

        # Collapse the final core into rest of tensor
        W.cores[-2] = W.cores[-2]@W.cores[-1]
        W.cores.pop(-1)
        W.row_dims.pop(-1)
        return W

    def TT_STLS(self, x, lamb):
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
            W = self.TT_PI(x)
            supp = self.coarse_supp(W, lamb)
            self.apply_supp(supp)
            #print(supp)

            if self.verbose:
                print(f'Iteration {i}:')
                print(f'Support size: {self.supp_size(supp)}')
                print('----------------')

            # supp is montonically decreasing. So only need to compare
            #   to size of previous support
            if self.supp_size(supp) == self.supp_size(supp_prev):
                break
            i += 1

        if self.verbose:
            print(f'Finished TT-STLS in {i} iterations, with final support size {self.supp_size(supp)}')
            print(f'Time elapsed: {time() - st:.2f} seconds')
            print(supp)

        return W

    def flatten(self):
        """
        Flatten feature tensor to a 2D array, with shape
        (prod(J_d), M)

        Also flatten the index map, so that we can track which features are active
         after flattening.

        Returns
        -------
        Theta_flat : np.array
            Flattened feature tensor, with shape (prod(J_d), M)
        
        """
        D = self.order - 1
        Theta_full = self.full().squeeze() # (J_0, ..., J_{D-1}, M)
        Theta_flat = np.moveaxis(Theta_full, D, 0).reshape(
            self.shape[-1], -1
        ).transpose() # (prod(J_d), M)

        return Theta_flat
    
    def supp_size(self, supp):
        """
        Utility function, gives support size
        """
        return np.prod([s.sum() for s in supp])
