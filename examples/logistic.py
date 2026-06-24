import os,sys
sys.path.insert(0, '.')
sys.path.insert(0, '../WSINDy-T')
import numpy as np
import copy
from feature_tensor import feature_tensor
from test_function import piecewise_polynomial
from scipy.signal import correlate
from dysts.flows import Lorenz

def generate_data(
    n_samples: int = 100,
    t_start: float = 0.0,
    t_end: float = 10.0,
    u0: float = 0.01,
    w1: float = 1.0,
    w2: float = -1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the analytical logistic solution at equispaced times.
 
    Parameters
    ----------
    n_samples : int
        Number of equispaced time points.
    t_start : float
        Start time.
    t_end : float
        End time.
    u0 : float
        Initial condition u(t_start).
    w1 : float
        Linear coefficient in du/dt = w1*u + w2*u^2.
    w2 : float
        Quadratic coefficient.
 
    Returns
    -------
    t : np.ndarray, shape (n_samples,)
        Sample times.
    u : np.ndarray, shape (n_samples,)
        Exact solution values at sample times.
    """
    t = np.linspace(t_start, t_end, n_samples)
    exp_term = np.exp(w1 * (t - t_start))
    u = (w1 * u0 * exp_term) / (w1 + w2 * u0 * (exp_term - 1))
    return t, u

if __name__ == '__main__':
    D = 1
    M = 1000
    X = np.expand_dims(generate_data(n_samples=M),axis=0) # (1, M)
    f = [
        lambda x: 1,
        lambda x: x,
        lambda x: x**2
    ]
    

    phi, dphi = piecewise_polynomial(
        0.5, 16, 0, 10, M
    )
    Mp = M - len(phi) + 1
    Theta = feature_tensor(X,f,phi=phi)

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

    print(W[0].full())
    print(suppd)