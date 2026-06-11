"""
3D Lorenz
"""
import os,sys
sys.path.insert(0, '.')
sys.path.insert(0, '/TT-WSINDy')
sys.path.insert(0, '../TT-WSINDy')
import numpy as np
import copy
from time import time
from feature_tensor import feature_tensor
from test_function import piecewise_polynomial
from sparsification import TT_MSTLS
from scipy.signal import correlate
from dysts.flows import Lorenz

if __name__ == '__main__':
    D = 3
    M = 100 #note that TT-SVD is slower, for this many time points
    model = Lorenz(
        parameters = {
            'beta' : 10,
            'rho' : 28,
            'sigma' : 8/3
        },
        ic = [2,1,1]
    )
    X = np.array(model.make_trajectory(M, verbose=True)).transpose() # (3, 10)
    f = [
        lambda x : 1,
        lambda x : x,
        lambda x : x**2,        # spurious basis feature
        lambda x : np.sin(x)    # ""
    ]
    J = len(f)
    
    # Make (weak) feature tensor
    phi, dphi = piecewise_polynomial(
        0.5, 16, 0, 10, M
    )
    Mp = M - len(phi) + 1
    ThetaX = feature_tensor(X,f,phi=phi)

    thresh = 0
    Theta = feature_tensor(X,f,
                           threshold=thresh,
                           phi=phi,
                           verbose=True)

    # Weak form LHS
    dphi = np.expand_dims(dphi, axis=0)
    Y = correlate(X, dphi, mode='valid').transpose() # (Mp, D)

    # Lambdas range, from seth code
    denom = 20
    lambs = 10**(
        (4/(denom))*np.arange(0,denom+1) - 4
    )

    # Apply TT-STLS, for each dimension separately
    W = []
    for d in range(Y.shape[1]):
    #for d in range(1):
        print('--------')
        print('--------')
        print('dim = {}'.format(d+1))
        print('--------')
        print('--------')

        st_mstls = time()

        ThetaD = copy.deepcopy(Theta)
        Yd = Y[:,d]
        ThetaStar = TT_MSTLS(ThetaD, Yd, lambs, J**D, verbose=True)
        #W.append(Wstar)

        end_mstls = time()

        print('--------')
        print('TT-MSTLS complete.')
        print(f'Runtime: {end_mstls - st_mstls}')
        #print(f'Support: {suppStar}')
        print('--------')