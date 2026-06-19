"""
lorenz97.py

WSINDy discovery of the Lorenz-96 system, mirroring the discovery experiment in
``experiments/complexityvWSINDy.py`` -- generate a D-dimensional Lorenz-96
trajectory, run WSINDy against the FULL {1, x} cross-term library, and print the
discovered terms -- driven by the MathBioCU **PyWLaSDI** ``wsindy`` package
instead of the local ``WSINDy-T`` (TT-WSINDy / flat-WSINDy) implementation.

    Lorenz-96:  dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
                        =  x_{i-1} x_{i+1} - x_{i-2} x_{i-1} - x_i + F

Library
-------
The full tensor product of the per-dimension basis {1, x} across all D state
variables -- i.e. every square-free monomial / cross term

    1, x_i, x_i x_j, x_i x_j x_k, ..., x_1 x_2 ... x_D

for a total of 2^D candidate functions, shared across every equation. This is
exactly the flat (J=2) library of complexityvWSINDy.py.

PyWLaSDI's built-in library generator (``wsindy.poolDatagen``) only produces
total-degree monomials (which include squares x_i^2 and cap the total degree),
so it cannot represent this square-free tensor product. We therefore subclass
``wsindy`` as ``TensorProductWsindy`` and override ONLY ``poolDatagen`` to build
the 2^D cross-term library; everything else -- the uniform grid of
piecewise-polynomial test functions (weak form), the GLS covariance handling and
the sequential-thresholding least-squares solve -- is PyWLaSDI's.

Problem size
------------
The uniform test-function grid lays down a fixed number of rows N (set by L,
overlap, M); the regression is well posed only when N >= 2^D. The defaults
(D=8, M=2000, L=30, overlap=0.5) give N=282 >= 2^8=256. For D >= 9 raise
``overlap`` (toward ~0.9) and/or ``M`` so N keeps up; ``run`` reports N vs 2^D
and warns when the system is underdetermined.

Run with the env that has numpy/scipy (e.g. the miniconda3 python):
    python examples/lorenz97.py
"""
import os
import sys
import io
import contextlib
import itertools

import numpy as np
from scipy.integrate import odeint

# Use the vendored PyWLaSDI wsindy package (../PyWLaSDI), resolved relative to
# this file so the script runs from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PyWLaSDI"))
from wsindy import wsindy  # noqa: E402


class TensorProductWsindy(wsindy):
    """
    PyWLaSDI ``wsindy`` whose candidate library is the full tensor product of
    the per-dimension basis {1, x} across all D state variables (2^D square-free
    cross terms), instead of PyWLaSDI's total-degree monomials.

    Only ``poolDatagen`` (library construction) is overridden; the weak-form
    test functions, GLS handling and sparsification are inherited unchanged.
    """
    def poolDatagen(self, xobs):
        # xobs: (n_timepoints, D). Build every product of a subset of the state
        # variables -- one factor in {1, x_dim} per dimension -- i.e. all 0/1
        # power vectors. itertools.product yields the constant (all-zeros) first,
        # matching the feature ordering of complexityvWSINDy's flat library.
        n, d = xobs.shape
        combos = list(itertools.product([0, 1], repeat=d))   # 2^d cross terms
        theta = np.empty((n, len(combos)))
        for k, powers in enumerate(combos):
            col = np.ones(n)
            for dim, p in enumerate(powers):
                if p:
                    col = col * xobs[:, dim]
            theta[:, k] = col
        tags = np.array(combos, dtype=float)                 # 0/1 power vectors
        return theta, tags


def lorenz96_rhs(F):
    """Return the Lorenz-96 RHS f(x, t) with constant forcing F."""
    def L96(x, t):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    return L96


def generate_data(D, F, M, t0, tM):
    """
    Integrate a D-dimensional Lorenz-96 trajectory.

    Returns t (M,) and X (M, D) in PyWLaSDI's (time, state) convention -- the
    transpose of the local WSINDy-T convention (D, M).
    """
    x0 = F * np.ones(D)
    x0[0] += 0.01                       # small perturbation off equilibrium
    t = np.linspace(t0, tM, M)
    X = odeint(lorenz96_rhs(F), x0, t)  # odeint returns (M, D) -> no transpose
    return t, X


def tag_to_str(tag):
    """Turn a library tag (vector of monomial powers) into a readable string."""
    powers = [int(round(p.real)) for p in tag]
    if not any(powers):
        return "1"
    factors = []
    for k, p in enumerate(powers):
        if p == 1:
            factors.append(f"x_{k + 1}")
        elif p > 1:
            factors.append(f"x_{k + 1}^{p}")
    return " ".join(factors)


def discovered_terms(model, tol=1e-8):
    """
    Extract discovered terms from a fitted model as a list (length D) of lists
    of (coefficient, powers_tuple, term_string), one per equation, sorted by
    increasing total degree.
    """
    coef, tags = model.coef, model.tags        # (n_terms, D), (n_terms, D)
    D = coef.shape[1]
    per_eq = []
    for d in range(D):
        terms = []
        for k in range(tags.shape[0]):
            c = coef[k, d]
            if abs(c) > tol:
                powers = tuple(int(round(p.real)) for p in tags[k])
                terms.append((float(c), powers, tag_to_str(tags[k])))
        terms.sort(key=lambda t: (sum(t[1]), t[1]))   # degree, then lexicographic
        per_eq.append(terms)
    return per_eq


def print_discovered(per_eq):
    """Print the discovered governing equations (analog of print_supp)."""
    for d, terms in enumerate(per_eq):
        rhs = "  ".join(f"{c:+.3f} ({s})" for c, _, s in terms)
        print(f"x_{d + 1}' = {rhs}")


def true_lorenz96_terms(D, F):
    """Exact Lorenz-96 terms per equation as {powers_tuple: coefficient}."""
    eqs = []
    for i in range(D):
        terms = {}
        terms[tuple([0] * D)] = F                                   # + F
        p = [0] * D; p[i] = 1; terms[tuple(p)] = -1.0              # - x_i
        p = [0] * D; p[(i - 1) % D] += 1; p[(i + 1) % D] += 1       # + x_{i-1} x_{i+1}
        terms[tuple(p)] = terms.get(tuple(p), 0.0) + 1.0
        p = [0] * D; p[(i - 2) % D] += 1; p[(i - 1) % D] += 1       # - x_{i-2} x_{i-1}
        terms[tuple(p)] = terms.get(tuple(p), 0.0) - 1.0
        eqs.append(terms)
    return eqs


def verify(per_eq, D, F, coef_tol=0.05):
    """Compare discovered support/coefficients to the true Lorenz-96 system."""
    truth = true_lorenz96_terms(D, F)
    all_ok = True
    for d in range(D):
        found = {powers: c for c, powers, _ in per_eq[d]}
        true_supp, found_supp = set(truth[d]), set(found)
        missing = true_supp - found_supp
        extra = found_supp - true_supp
        bad_coef = [
            k for k in true_supp & found_supp
            if abs(found[k] - truth[d][k]) > coef_tol * max(1.0, abs(truth[d][k]))
        ]
        ok = not (missing or extra or bad_coef)
        all_ok &= ok
        note = ""
        if missing: note += f" missing={[tag_to_str(np.array(m)) for m in missing]}"
        if extra: note += f" extra={[tag_to_str(np.array(e)) for e in extra]}"
        if bad_coef: note += f" wrong_coef={[tag_to_str(np.array(b)) for b in bad_coef]}"
        print(f"  x_{d + 1}': {'OK' if ok else 'MISMATCH'}{note}")
    return all_ok


def run(D=8, F=8.0, M=2000, t0=0.0, tM=20.0,
        ld=0.25, L=30, overlap=0.5, scale_theta=2, useGLS=1e-12, verbose=False):
    """
    Generate Lorenz-96 data, run PyWLaSDI WSINDy against the full {1, x}
    cross-term library (2^D terms), print and return the discovered terms.

    Parameters
    ----------
    D, F, M, t0, tM : Lorenz-96 problem size / forcing / time grid.
    ld : sequential-thresholding parameter (sparsity knob).
    L, overlap : uniform test-function support (in samples) and overlap fraction;
        together with M these set the number of test functions N. Need N >= 2^D.
    scale_theta : column normalization (0 = none, 2 = l2). Defaults to l2 because
        the cross-term columns span many orders of magnitude.
    useGLS : generalized-least-squares covariance regularization.

    Returns
    -------
    per_eq : list (length D) of discovered terms, each a list of
             (coefficient, powers_tuple, term_string).
    """
    t, X = generate_data(D, F, M, t0, tM)

    model = TensorProductWsindy(
        # polys/trigs feed PyWLaSDI's default library generator, which
        # TensorProductWsindy overrides; the library is the full {1, x}^D product.
        polys=np.arange(0, 2),
        trigs=[],
        scaled_theta=scale_theta,
        ld=ld,
        gamma=10 ** (-np.inf),          # no Tikhonov regularization
        multiple_tracjectories=False,
        useGLS=useGLS,
    )

    # getWSindyUniform1: single trajectory, uniform grid of test functions.
    if verbose:
        model.getWSindyUniform1(X, t, L=L, overlap=overlap)
    else:
        with contextlib.redirect_stdout(io.StringIO()):   # mute internal prints
            model.getWSindyUniform1(X, t, L=L, overlap=overlap)

    n_terms = model.tags.shape[0]                 # = 2^D
    n_testfn = model.mats[0][0].shape[0]          # rows in the weak-form regression
    print(f"Lorenz-96 WSINDy via PyWLaSDI  (D={D}, F={F:g}, M={M}, "
          f"t in [{t0:g},{tM:g}])")
    print(f"Library: full {{1,x}} tensor product = 2^{D} = {n_terms} cross terms; "
          f"{n_testfn} test functions "
          f"({'over' if n_testfn >= n_terms else 'UNDER'}-determined)")
    if n_testfn < n_terms:
        print(f"  WARNING: underdetermined ({n_testfn} < {n_terms}); recovery may "
              f"fail. Increase overlap (toward ~0.9), increase M, or decrease L.")

    per_eq = discovered_terms(model)
    print("Discovered terms:")
    print_discovered(per_eq)
    print("Verification against true Lorenz-96:")

    return per_eq


if __name__ == "__main__":
    run(D=9, tM=30, M=5000)