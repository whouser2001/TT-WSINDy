import os,sys
sys.path.insert(0, '.')
sys.path.insert(0, '/WSINDy-T')
sys.path.insert(0, '../WSINDy-T')
import numpy as np
import copy
from feature_tensor import feature_tensor
from test_function import piecewise_polynomial
from scipy.signal import correlate
from dysts.flows import LorenzCoupled

if __name__ == '__main__':
    D = 6
    M = 100 #note that TT-SVD is slower, for this many time points
    model = LorenzCoupled()
    X = np.array(model.make_trajectory(n=M)).transpose() # (6, 100)
    f = [
        lambda x : 1,
        lambda x : x,
        lambda x : x**2, # superfluous basis feature
    ]
    
    # Make (weak) feature tensor
    phi, dphi = piecewise_polynomial(
        0.5, 16, 0, 10, M
    )
    Mp = M - len(phi) + 1
    ThetaX = feature_tensor(X,f,phi=phi)

    thresh = 0.0
    lamb = 0.01
    Theta = feature_tensor(X,f,
                           threshold=thresh,
                           phi=phi,
                           verbose=True)

    # Weak form LHS
    dphi = np.expand_dims(dphi, axis=0)
    Y = correlate(X, dphi, mode='valid').transpose() # (Mp, D)

    # Apply TT-STLS, for each dimension separately
    W = []
    for d in range(Y.shape[1]):
        ThetaD = copy.deepcopy(Theta)
        Yd = Y[:,d]
        W.append(
            ThetaD.TT_STLS(Yd, lamb)
        )
        suppd = ThetaD.all_active_features()

    # See results