"""
Utilities shared by the experiment scripts in this directory.
"""
import itertools
import os

import numpy as np

TERM_SEP = ' '
SUPP_SEP = '  '
RULE_WIDTH = 66
MIN_LABEL_COL = 10

def term_label(idx, labels):
    """Readable name for one tensor-product candidate index.
    """
    parts = [labels[j].format(d) for d, j in enumerate(idx) if j]
    return '1' if not parts else TERM_SEP.join(parts)


def function_major_label(idx, labels, D):
    """Readable name for one function-major candidate index.
    """
    parts = [labels[k].format(d) for k, d in enumerate(idx) if d < D]
    return '1' if not parts else TERM_SEP.join(parts)


def check_labels(f, labels, function_major=False):
    """Fail early if LABELS does not line up with the library f."""
    want = len(f) - 1 if function_major else len(f)
    if len(labels) != want:
        kind = 'non-constant library functions' if function_major else 'library functions'
        raise ValueError(f'LABELS has {len(labels)} entries for {want} {kind}')

def rule(char='='):
    """Print one horizontal section rule."""
    print(char * RULE_WIDTH)


def print_supp(supps, feature_maps, labels, label_fn=term_label):
    """Print one discovered support per equation, one line each."""
    for d, (supp, fmap) in enumerate(zip(supps, feature_maps)):
        if supp is None:
            terms = '(not solved)'
        elif len(supp) == 0:
            terms = '(empty)'
        else:
            terms = SUPP_SEP.join(label_fn(tuple(fmap[k]), labels)
                                  for k in supp)
        print(f"  x{d}' : {terms}")


def print_coeff_table(Wtrue, Wweak, Wstrong, labels, eqs=None,
                      label_fn=term_label):
    """Print true vs. recovered coefficients for every nonzero true term."""
    eqs = (0,) if eqs is None else list(eqs)
    rows = []
    for d in eqs:
        for idx in map(tuple, np.argwhere(np.abs(Wtrue[d]) > 1e-12)):
            rows.append((d, label_fn(idx, labels),
                         Wtrue[d][idx], Wweak[d][idx], Wstrong[d][idx]))
    w = max([MIN_LABEL_COL] + [len(r[1]) for r in rows])
    which = 'all equations' if len(eqs) > 1 else f"equation x{eqs[0]}'"
    print(f'true vs. recovered coefficients, nonzero true terms ({which}):')
    print(f"  {'eq':>4}  {'candidate':>{w}}  {'true':>9}  {'weak':>11}"
          f"  {'strong':>11}")
    for d, lab, tr, wk, st in rows:
        print(f"  {f'x{d}' + chr(39):>4}  {lab:>{w}}  {tr:>9.4f}"
              f"  {wk:>11.4f}  {st:>11.4f}")

def tt_pi_coeffs(Theta, y, shape):
    """One TT-PI solve, returned as a dense coefficient tensor of `shape`.

    shape is (J,)*D for a tensor-product library and (D+1)^(J-1) for a
    function-major one, matching the experiment's true_coeffs.

    TT_PI uses overwrite=False, so Theta is not mutated.
    """
    W = Theta.TT_PI(y)
    return np.asarray(W.full()).reshape(shape)

def rel_err(W, Wtrue):
    """Relative 2-norm error over the full stacked coefficient tensor."""
    return np.linalg.norm(W - Wtrue) / np.linalg.norm(Wtrue)

def flatten_trajectories(Xs):
    """Concatenate trajectories along time: (P, D, M) -> (D, P*M).

    The blocking is trajectory-major, as feature_tensor's n_traj assumes.
    """
    return np.concatenate(list(Xs), axis=1)

def library_rank(X, f, J, tol=1e-10):
    """Numerical rank of the strong (pointwise) tensor-product library, and J^D.

    Pass the flattened (D, P*M) data for the rank pooled over all
    trajectories, which is what the regression sees.
    """
    D, M = X.shape
    fmap = list(itertools.product(*[range(J)] * D))
    bdata = np.stack([np.vectorize(f[j])(X).astype(float) for j in range(J)])
    G = np.ones((len(fmap), M))
    for k, tup in enumerate(fmap):
        for d in range(D):
            G[k] *= bdata[tup[d], d]
    s = np.linalg.svd(G, compute_uv=False)
    return int(np.sum(s / s[0] > tol)), len(fmap)

def library_rank_function_major(X, f, D, J, tol=1e-10):
    """Numerical rank of the function-major candidate library, and its size.

    Enumerates the (D+1)^(J-1) candidates -- one placement per non-constant
    function, index D meaning absent.
    """
    M = X.shape[1]
    cols = []
    for picks in itertools.product(range(D + 1), repeat=J - 1):
        c = np.ones(M)
        for j, d in enumerate(picks):
            if d < D:
                c = c * np.vectorize(f[j + 1])(X[d]).astype(float)
        cols.append(c)
    s = np.linalg.svd(np.stack(cols), compute_uv=False)
    return int(np.sum(s / s[0] > tol)), len(cols)

def load_txt(path, hint='set recompute_data = True and rerun'):
    """Read a results file as a 2-D array, or exit with a message naming it.

    Every experiment caches its measurements in results/ and can replot
    without recomputing (recompute_data = False); this decides what a missing
    or empty cache does.
    """
    if not os.path.exists(path):
        raise SystemExit(f'{path} not found -- {hint}')
    rows = np.atleast_2d(np.loadtxt(path))
    if not rows.size:
        raise SystemExit(f'{path} is empty -- {hint}')
    return rows
