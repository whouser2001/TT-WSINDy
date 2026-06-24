"""
TT-WSINDy with a PyWLaSDI (wsindy.py) fine pass.

This is a variant of ``TT-WSINDy/ttwsindy.py``. The COARSE pass is identical -- a
per-dimension TT-MSTLS sweep that reduces the J^D candidate library to a small set
of surviving cross-terms -- but the FINE pass is delegated to the MathBioCU
PyWLaSDI ``wsindy`` package (``WSINDy/wsindy.py``) instead of the local flat
``sparsification.MSTLS``.

The fine pass is carried out by ``ReducedTensorProductWsindy``, a version of
``examples/lorenz97.py``'s ``TensorProductWsindy`` whose candidate library is not
the full {1,x}^D tensor product but only the cross-terms that survived the coarse
pass for that equation. PyWLaSDI then builds its own weak form (a uniform grid of
piecewise-polynomial test functions), applies its GLS covariance handling, and runs
its sequential-thresholding least-squares solve -- i.e. the wsindy.py code does the
fine pass of TT-WSINDy.

Because each equation d has its own coarse support, the fine pass is run once per
equation on that equation's reduced library (``fine_pass_equation``), which is the
single-equation extract of PyWLaSDI's ``getWSindyUniform1``.

Run with the env that has numpy/scipy/scikit_tt (the miniconda3 python):
    python TT-WSINDy/ttwsindy_finepass.py
"""
import os
import sys
import copy
import itertools
from time import time

import numpy as np
from scipy.signal import correlate
from scipy.linalg import lstsq

# Resolve sibling packages relative to THIS file so the module imports cleanly
# from any working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE,                                   # TT-WSINDy (local modules)
           os.path.join(_HERE, "..", "examples"),   # lorenz97.TensorProductWsindy
           os.path.join(_HERE, "..", "WSINDy")):     # PyWLaSDI wsindy (also added by lorenz97)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_function
from feature_tensor import feature_tensor
import sparsification
from lorenz97 import TensorProductWsindy


class ReducedTensorProductWsindy(TensorProductWsindy):
    """
    PyWLaSDI ``TensorProductWsindy`` restricted to a *reduced* candidate library.

    ``examples/lorenz97.py``'s ``TensorProductWsindy`` builds the full {1,x}^D
    tensor product (2^D square-free cross terms). Here the library is instead the
    explicit list of surviving cross-terms produced by TT-WSINDy's coarse pass,
    evaluated through the user's candidate functions ``f`` (so it is not limited to
    monomials). Everything else -- the weak-form test functions, the GLS handling
    and the sequential-thresholding solve -- is inherited from PyWLaSDI ``wsindy``.

    Parameters
    ----------
    f : list of callable
        Candidate functions fj : R -> R (the same list passed to TT_WSINDy).
    feature_map : list of tuple of int
        feature_map[k] is a length-D tuple giving, for column k, the index into
        ``f`` used in each dimension. Typically
        ``list(itertools.product(*coarse_supp))``.
    **kwargs
        Forwarded to ``wsindy.__init__`` (polys, trigs, scaled_theta, ld, gamma,
        multiple_tracjectories, useGLS).
    """

    def __init__(self, f, feature_map, **kwargs):
        super().__init__(**kwargs)
        self.f = f
        self.feature_map = feature_map

    def poolDatagen(self, xobs):
        # xobs: (M, D). Column k is prod_dim f[feature_map[k][dim]](x[:, dim]),
        # i.e. the surviving cross terms instead of TensorProductWsindy's full
        # {1,x}^D product. Candidate evaluations are cached per (feature index,
        # dimension) so each f_j(x_dim) is computed once even when it appears in
        # many cross terms.
        n, d = xobs.shape
        cache = {}

        def evalfd(j, dim):
            key = (j, dim)
            if key not in cache:
                cache[key] = np.vectorize(self.f[j])(xobs[:, dim]).astype(float)
            return cache[key]

        Jtilde = len(self.feature_map)
        theta = np.empty((n, Jtilde))
        for k, combo in enumerate(self.feature_map):
            col = np.ones(n)
            for dim in range(d):
                col = col * evalfd(combo[dim], dim)
            theta[:, k] = col

        # tags[k] = per-dimension candidate-function indices for column k. For a
        # monomial basis ({1, x, x^2, ...}) these coincide with monomial powers,
        # so lorenz97's tag_to_str / discovered_terms remain meaningful.
        tags = np.array(self.feature_map, dtype=float)
        return theta, tags

    def fine_pass_equation(self, xobs, tobs, target, L=30, overlap=0.5):
        """
        Weak-form sequential-thresholding solve for a single equation.

        This is the per-equation body of PyWLaSDI ``getWSindyUniform1`` extracted
        so that only equation ``target`` is solved: TT-WSINDy gives each equation
        its own reduced library, and solving one equation at a time keeps the
        thresholding parameter ``ld`` pristine (``sparsifyDynamics`` may otherwise
        shrink it across equations). The weak-form library (``buildTheta``), the
        uniform test-function grid (``Uniform_grid``), the GLS covariance handling
        and the sparsifying solve (``sparsifyDynamics``) are all PyWLaSDI's.

        Parameters
        ----------
        xobs : np.ndarray
            Data, shape (M, D) (PyWLaSDI (time, state) convention).
        tobs : np.ndarray
            Sample times, shape (M,).
        target : int
            Index of the equation (state dimension) to regress.
        L : int
            Uniform test-function support, in samples.
        overlap : float
            Test-function overlap fraction.

        Returns
        -------
        w : np.ndarray
            Coefficient vector over this model's reduced library, length
            len(feature_map). Zeroed entries are exactly 0 (sparsified out).
        """
        Theta_0, tags, M_diag = self.buildTheta(xobs)
        V, Vp, grid = self.Uniform_grid(tobs, L, overlap, [0, np.inf, 0])

        if self.useGLS > 0:
            Cov = Vp.dot(Vp.T) + self.useGLS * np.identity(V.shape[0])
            RT = np.linalg.cholesky(Cov)
            G = lstsq(RT, V.dot(Theta_0))[0]
            b = lstsq(RT, Vp.dot(xobs[:, target]))[0]
        else:
            RT = 1 / np.linalg.norm(Vp, 2, 1)
            RT = np.reshape(RT, (RT.size, 1))
            G = np.multiply(V.dot(Theta_0), RT)
            b = RT.T * Vp.dot(xobs[:, target])

        if self.scale_theta > 0:
            w_temp = self.sparsifyDynamics(np.multiply(G, (1 / M_diag.T)), b, 1)
            w = np.ndarray.flatten(np.multiply((1 / M_diag), w_temp))
        else:
            w = np.ndarray.flatten(self.sparsifyDynamics(G, b, 1))

        # stash weak-form artifacts for inspection (mirrors getWSindyUniform1)
        self.tags = tags
        self.mats = [[V, Vp]]
        self.ts_grids = [grid]
        return w


def TT_WSINDy(X, t0, tM, f, TTlambs,
              testfn=('piecewise_polynomial', 1/20, 16, 1),
              loss='default',
              threshold=0.0,
              verbosity=0,
              low_rank=False,
              ld=0.25, L=30, overlap=0.5, scale_theta=2,
              useGLS=1e-12, gamma=10**(-np.inf),
              coef_tol=1e-8):
    """
    TT-WSINDy with a PyWLaSDI fine pass.

    Discover the governing equations of a dynamical system from data using
    weak-form SINDy with tensor-train sparsification: a coarse TT-MSTLS pass per
    dimension (identical to ``ttwsindy.TT_WSINDy``), followed by a fine pass done
    by PyWLaSDI ``wsindy`` on the reduced library (a ``ReducedTensorProductWsindy``
    weak-form sequential-thresholding solve) in place of the local
    ``sparsification.MSTLS``.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape D x M (D system dimensions, M time points).
    t0, tM : float
        Initial and final time point. Data is assumed equispaced in time.
    f : list of callable
        Candidate functions fj : R -> R.
    TTlambs : iterable
        Threshold values to test in the coarse TT-MSTLS pass.
    testfn : tuple
        Coarse-pass test-function spec, (name, *parameters); see ttwsindy.py.
        (The fine pass uses PyWLaSDI's own uniform test functions, set by L /
        overlap, not this test function.)
    loss : str
        Coarse-pass loss function. Currently only 'default'.
    threshold : float
        SVD truncation parameter for the coarse feature tensor / TT-PI.
    verbosity : int
        0 : silent; 1 : per-dimension supports and walltimes (verbose);
        2 : additionally coarse-pass debug output.
    low_rank : bool
        Feature-tensor construction mode (see feature_tensor); False matches
        ttwsindy.TT_WSINDy's default. Set True for large M.
    ld : float
        PyWLaSDI sequential-thresholding (sparsity) parameter for the fine pass.
    L : int
        PyWLaSDI uniform test-function support, in samples.
    overlap : float
        PyWLaSDI test-function overlap fraction. With M these set the number of
        fine-pass test functions N; recovery needs N >= (reduced library size).
    scale_theta : int
        PyWLaSDI column normalization (0 = none, 2 = l2).
    useGLS : float
        PyWLaSDI generalized-least-squares covariance regularization.
    gamma : float
        PyWLaSDI Tikhonov regularization parameter (default: none).
    coef_tol : float
        Magnitude above which a fine-pass coefficient counts as surviving.

    Returns
    -------
    W : list of np.ndarray
        Per-dimension coefficient vectors on the surviving features.
    supp : list of np.ndarray
        Per-dimension surviving column indices into each feature_map.
    feature_maps : list of list of tuple
        feature_maps[d][k] gives the candidate-function index in each dimension
        for the k-th column of dimension d's reduced library.
    ttwsindy_time : float
        Total wall-clock runtime.
    tt_mstls_time : float
        Cumulative coarse TT-MSTLS runtime across dimensions.
    fine_time : float
        Cumulative PyWLaSDI fine-pass runtime across dimensions.
    coarse_supps : list of list of np.ndarray
        Per-dimension coarse supports from TT-MSTLS.
    """
    if X.ndim == 1: D, M = (1, X.size)
    else: D, M = X.shape
    J = len(f)
    problemSize = J**D

    verbose = (verbosity >= 1)
    debug = (verbosity // 2 >= 1)

    st = time()

    if testfn[0] == 'piecewise_polynomial':
        radius = (tM - t0)*testfn[1]
        degree = testfn[2]
        phi, dphi = test_function.piecewise_polynomial(
            radius, degree, t0, tM, M, order=testfn[3]
        )
    else:
        raise NotImplementedError

    # coarse feature tensor (weak form, via the TT-WSINDy test function)
    Theta = feature_tensor(X, f,
                           threshold=threshold,
                           phi=phi,
                           verbose=debug,
                           low_rank=low_rank)

    # weak-form left-hand side for the COARSE pass (TT-MSTLS)
    if D > 1: dphi = np.expand_dims(dphi, axis=0)
    Y = -1*correlate(X, dphi, mode='valid').transpose()    # (Mp, D)

    # data in PyWLaSDI (time, state) convention for the fine pass
    Xmat = X if X.ndim > 1 else X.reshape(1, -1)
    xobs = np.ascontiguousarray(Xmat.T)                    # (M, D)
    tobs = np.linspace(t0, tM, M)

    # store per-dimension results
    W = []
    supp = []
    feature_maps = []
    coarse_supps = []

    tt_mstls_time = 0.0
    fine_time = 0.0
    for d in range(D):

        # coarse pass: TT-MSTLS (unchanged from ttwsindy.py)
        if d < D-1: Theta_d = copy.deepcopy(Theta)
        else: Theta_d = Theta

        tt_mstls_st = time()
        y_d = Y[:, d] if D > 1 else Y
        Theta_star, wStar, suppStar = sparsification.TT_MSTLS(
            Theta_d, y_d, TTlambs, problemSize, verbose=debug
        )
        tt_mstls_time += time() - tt_mstls_st

        coarse_supp = Theta_star.all_active_features()
        coarse_supps.append(coarse_supp)
        feature_map = list(itertools.product(*coarse_supp))
        feature_maps.append(feature_map)

        if verbose:
            print('------------------')
            print(f'TT-MSTLS concluded for d = {d}')
            print(f'coarse support: {suppStar}')
            print(f'reduced features: {len(feature_map)}')
            print('Beginning PyWLaSDI fine pass')

        # fine pass: PyWLaSDI wsindy on the reduced library. A fresh model per
        # equation keeps each equation's reduced library and ld independent.
        fine_st = time()
        model = ReducedTensorProductWsindy(
            f, feature_map,
            polys=np.arange(0, 2), trigs=[],          # unused: poolDatagen overridden
            scaled_theta=scale_theta, ld=ld, gamma=gamma,
            multiple_tracjectories=False, useGLS=useGLS,
        )
        w = model.fine_pass_equation(xobs, tobs, d, L=L, overlap=overlap)
        fine_time += time() - fine_st

        suppStar = np.where(np.abs(w) > coef_tol)[0]
        wStar = w[suppStar]
        W.append(wStar)
        supp.append(suppStar)

        if verbose:
            N = model.mats[0][0].shape[0]
            Jt = len(feature_map)
            print(f'fine pass: {N} test functions vs {Jt} reduced features '
                  f'({"over" if N >= Jt else "UNDER"}-determined)')
            print(f'surviving support (indices into feature_map): {suppStar}')

    ttwsindy_time = time() - st
    if verbose:
        print('------------------')
        print('TT-WSINDy (PyWLaSDI fine pass) complete.')
        print(f'Total runtime: {ttwsindy_time}')
        print(f'TT-MSTLS runtime: {tt_mstls_time}')
        print(f'Fine-pass runtime: {fine_time}')
        print('------------------')

    return W, supp, feature_maps, ttwsindy_time, tt_mstls_time, fine_time, coarse_supps


if __name__ == '__main__':
    # Lorenz-96 sanity check (mirrors complexityvWSINDy / lorenz97). D=8 is the
    # known-good regime (chaos -> full-rank coarse pass); recovery quality is set
    # by the coarse TT-MSTLS pass, which under-recovers at smaller D. ~45 s.
    from scipy.integrate import odeint

    F = 8.0
    D = 8
    M = 5000
    t0, tM = 0.0, 30.0
    t = np.linspace(t0, tM, M)

    def L96(x, _t):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

    x0 = F * np.ones(D); x0[0] += 0.01
    X = odeint(L96, x0, t).T                      # (D, M)

    f = [lambda x: 1, lambda x: x]                # {1, x} per dimension
    fstr = [lambda n: '', lambda n: f'x_{n}']
    TTlambs = np.linspace(1e-3, 5e-1, 25)

    ret = TT_WSINDy(X, t0, tM, f, TTlambs, threshold=1e-16, verbosity=1)
    supps, feature_maps = ret[1], ret[2]

    print('\nTT-WSINDy (PyWLaSDI fine pass) discovered support:')
    for d in range(D):
        terms = []
        for k in supps[d]:
            s = ''.join(fstr[feature_maps[d][k][d2]](d2 + 1) for d2 in range(D))
            terms.append(s if s else '1')
        print(f"x_{d + 1}' : " + '  '.join(terms))
