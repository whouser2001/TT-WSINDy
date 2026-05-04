"""
Constructing the weak/strong feature tensor
"""
import numpy as np
from scikit_tt.tensor_train import TT
from scipy.signal import correlate

def feature_cores(X, f):
    """
    Construct list of feature cores from data 
    and candidate functions.

    Parameters
    ----------
    X : np.array
        Raw data with shape D x M
    f : list
        List of candidate functions
        fj : R -> R

    Returns
    -------
    Theta : list
        Length-D list of 
        R_d-1 x J x 1 x R_d np arrays,
        with R_0 = R_D = 1, and
        R_1 = ... = R_D-1 = M
    """
    J = len(f)
    D,M = X.shape

    # initialize blank cores
    cores = [np.zeros([1, J, 1, M])] + [np.zeros([M, J, 1, M])]*(D-1)

    # fill in cores with candidate function
    # evaluations of the data
    for j in range(J):

        # evaluate dataset on fj
        fX = np.vectorize(f[j])(X)

        # fill in slice of first core
        cores[0][0, j, 0, :] = fX[0, :]

        # fill in other cores
        for d in range(1,D):
            for m in range(M):
                cores[d][m, j, 0, m] = fX[d,m]

    return cores

def strong_core(M):
    return np.eye(M).reshape(M,M,1,1)

def weak_core(M, phi):
    Iphi = correlate(
        np.eye(M), np.reshape(1,phi.size)
    )
    return Iphi.reshape(M,Iphi.shape[1],1,1)

def feature_tensor(X,f,phi=None):
    """
    Construct list of feature cores from data 
    and candidate functions.

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

    Returns
    -------
    feature_tensor : scikit.tensor_train
    """
    M = X.shape[1]
    cores = feature_cores(X,f)
    if phi == None:
        return TT(cores.append(strong_core(M)))
    return TT(cores.append(weak_core(M,phi)))