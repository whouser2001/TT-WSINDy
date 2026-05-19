"""
Constructing the weak/strong feature tensor
"""
import numpy as np
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

    def __init__(self, X, f, threshold=0, phi=None):
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
        """
        J = len(f)
        D,M = X.shape

        # initialize blank cores
        cores = [np.zeros([1, J, 1, M])] + [np.zeros([M, J, 1, M])]*(D-1)

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
        self.supp_indices = [np.arange(J)]*(D-1)

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
        D = self.order
        active_features = [self.supp_indices[d] for d in range(D-1)]
        return active_features

    def apply_supp(self, supp):
        """
        Given support of W, reduce feature tensor to only those features.

        Parameters
        ----------
        supp : list of np.array
            supp[d] is boolean array of length J_d, indicating which features
            are in the support for dimension d
        """
        D = self.order
        cores_prime = [None]*(D)
        cores_prime[D-1] = self.cores[D-1] # weak/strong core
        for d in range(D - 1):

            # slce feature axis to only contain features in support
            cores_prime[d] = self.cores[d][:, supp[d], :, :]  

            # update index map: keep only suriving indices
            self.supp_indices[d] = self.supp_indices[d][supp[d]]
        
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
        D = self.order
        W = self.pinv(D-1, threshold=self.threshold) #D-1 is dim of system

        if x.shape[0] != self.shape[-1]:
            raise ValueError(
                f"Vector x must have length {self.shape[-1]}, but has length {x.shape[0]}"
            )
        
        # contract vector with self across first mode index of PI
        #   (last index of self)
        W.cores[0] = np.tensordot(x, W.cores[0], axes=(0,1))[:, np.newaxis, :, :]
        W.row_dims[0] = 1

        return W.squeeze() # collapse first core into the second
    
    def TT_SLTS_step(self, x, lamb):
        """
        Perform one step of sequential thresholding least squares (STLS)
        on the feature tensor.

        Parameters
        ----------
        x : np.array
            Data to regress against
        lamb : float
            Thresholding parameter
        eps : float
            Convergence parameter
        """

        # compute coefficient estimate
        W = self.TT_PI(x)

        # compute supp
        D = self.order
        supp = [None]*(D-1)
        for d in range(D-1):

            Wd = np.abs(W.cores[d])
            
            # compute coarse support of Wd
            # TODO: revise condition to incorporate scaling in Dan's paper?
            supp[d] = np.any(
                Wd >= lamb & Wd <= 1/lamb,
                axis = (0,2,3)
            )

        # apply supp
        self.apply_supp(supp)

    def flatten(self):
        """
        Flatten feature tensor to a 2D array, with shape
        (prod(J_d), M)

        Returns
        -------
        Theta_flat : np.array
            Flattened feature tensor, with shape (prod(J_d), M)
        """
        D = self.order
        Theta_full = self.full().squeeze() # (J_0, ..., J_{D-1}, M)
        Theta_flat = np.moveaxis(Theta_full, D-1, 0).reshape(
            self.shape[-1], -1
        ).transpose() # (prod(J_d), M)
        return Theta_flat

