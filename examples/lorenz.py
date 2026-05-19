
import os,sys
sys.path.insert(0, '.')
sys.path.insert(0, '../WSINDy-T')
import numpy as np
from feature_tensor import feature_tensor
from test_function import piecewise_polynomial
from sparsification import TT_STLS
from scipy.signal import correlate
from dysts.flows import Lorenz

if __name__ == '__main__':
    D = 3
    M = 1000 #note that TT-SVD is slower, for this many time points
    model = Lorenz(
        parameters = {
            'beta' : 10,
            'rho' : 28,
            'sigma' : 8/3
        },
        ic = [2,1,1]
    )
    X = np.array(model.make_trajectory(M)).transpose() # (3, 10)
    f = [
        lambda x : 1,
        lambda x : x,
        lambda x : x**2
    ]
    
    # Make (weak) feature tensor
    phi, dphi = piecewise_polynomial(
        0.5, 16, 0, 10, M
    )
    Mp = M - len(phi) + 1
    ThetaX = feature_tensor(X,f,phi=phi)

    # Weak form LHS
    dphi = np.expand_dims(dphi, axis=0)
    Y = correlate(X, dphi, mode='valid').transpose() # (Mp, D)

    # Apply TT-STLS
    
    for d in len(Y.shape[1]):
        Yd = Y[:,d]
        Wd = TT_STLS(ThetaX, Yd, lamb=0.1, eps=1e-10)


    W, W0 = TT_STLS(ThetaX, Y, lamb=0.1, eps=1e-6)

    # See results
    # How to convert to just printing the ODE?
    Wfull = W.full().squeeze()
    Wflat = np.moveaxis(Wfull, D, 0).reshape(
        W.row_dims[D], -1
    )
    print(Wflat.shape)
    print(Wflat)
    
    # Wflat = Wfull.flatten().reshape(
    #     np.prod(Wfull.shape[:-1]), Wfull.shape[-1]
    # )