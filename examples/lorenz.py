
import os,sys
sys.path.insert(0, '.')
sys.path.insert(0, '../WSINDy-T')
import numpy as np
from feature_tensor import feature_tensor
import TT_STLS
from dysts.flows import Lorenz

if __name__ == '__main__':
    model = Lorenz(
        parameters = {
            'beta' : 10,
            'rho' : 28,
            'sigma' : 8/3
        },
        ic = [2,1,1]
    )
    X = np.array(model.make_trajectory(10)).transpose() # (3, 10)
    f = [
        lambda x : 1,
        lambda x : x,
        lambda x : x**2
    ]
    ThetaX = feature_tensor(X,f)
    print(ThetaX)