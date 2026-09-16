"""
Weak form (TT-WSINDy) vs. strong form (MANDy) coefficient accuracy on Fermi-Pasta-Tsingou-Ulam (FPUT).
"""
import os, sys
sys.path.insert(0, '.')
sys.path.insert(0, '../TT-WSINDy')
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.signal import correlate
from scipy.integrate import odeint
from feature_tensor import feature_tensor
from test_function import piecewise_polynomial
import exputils as xu

RTOL = ATOL = 1e-12     

def _fpu_force(x, beta):
    """Acceleration field with fixed walls."""
    xp = np.concatenate(([0.0], x, [0.0]))
    dr = xp[2:] - xp[1:-1]          # x_{i+1} - x_i
    dl = xp[1:-1] - xp[:-2]         # x_i - x_{i-1}
    return (dr - dl) + beta * (dr ** 3 - dl ** 3)


def _fpu_rhs(z, t, n, beta):
    """First-order form of the equations of motion, with state z = [x, v]."""
    return np.concatenate((z[n:], _fpu_force(z[:n], beta)))


def fpu_energy(x, v, beta):
    """Total Hamiltonian of one configuration."""
    xp = np.concatenate(([0.0], x, [0.0]))
    d = xp[1:] - xp[:-1]
    U = np.sum(0.5 * d ** 2 + (beta / 4.0) * d ** 4)
    return 0.5 * np.sum(v ** 2) + U


def fput_trajectory(n, m, dt=0.015, beta=0.7, amplitude=3.0, seed=0):
    """Generate one uniformly-sampled FPUT trajectory.

    Parameters
    ----------
    n : int
        Number of oscillators.
    m : int
        Number of recorded snapshots (spaced by dt).
    dt : float
        Spacing between recorded snapshots, not an integrator step: odeint
        chooses its own adaptive steps.
    beta : float
        Cubic coupling strength.
    amplitude : float
        Half-width of the uniform random initial displacement box. Larger
        amplitude raises the energy, exciting more monomials, so the candidate
        library becomes full rank. Released from rest.
    seed : int
        RNG seed for the random initial condition.

    Returns
    -------
    t : ndarray (m,)
        Sample times.
    positions : ndarray (n, m)
        x(t) along the trajectory.
    velocities : ndarray (n, m)
        v(t) along the trajectory.
    """
    rng = np.random.default_rng(seed)
    x0 = 2.0 * amplitude * rng.random(n) - amplitude
    z0 = np.concatenate((x0, np.zeros(n)))       # released from rest

    t = np.arange(m) * dt
    z = odeint(_fpu_rhs, z0, t, args=(n, beta), rtol=RTOL, atol=ATOL)
    return t, z[:, :n].T.copy(), z[:, n:].T.copy()


def fput_trajectories(n, m, n_traj, dt=0.015, beta=0.7, amplitude=3.0, seed=0):
    """Generate n_traj FPUT trajectories from independent initial conditions.

    Every trajectory shares the sampling (m snapshots spaced by dt) and differs
    only in its random initial displacement, drawn with seed + p.

    Returns
    -------
    t : ndarray (m,)
        Sample times, common to all trajectories.
    positions : ndarray (n_traj, n, m)
    velocities : ndarray (n_traj, n, m)
    """
    trajs = [fput_trajectory(n, m, dt=dt, beta=beta, amplitude=amplitude,
                             seed=seed + p) for p in range(n_traj)]
    t = trajs[0][0]
    return t, np.stack([x for _, x, _ in trajs]), np.stack([v for _, _, v in trajs])

def _poly_add(p, q):
    r = dict(p)
    for k, v in q.items():
        r[k] = r.get(k, 0.0) + v
    return r

def _poly_scale(p, c):
    return {k: c * v for k, v in p.items()}

def _poly_mul(p, q):
    r = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            powers = {}
            for var, pw in m1 + m2:
                powers[var] = powers.get(var, 0) + pw
            key = tuple(sorted(powers.items()))
            r[key] = r.get(key, 0.0) + c1 * c2
    return r

def true_coeffs(D, J, beta):
    """Exact coefficient tensor of the FPUT force, per output oscillator.

    Returns
    -------
    Ws : list of ndarray
        Ws[i] is the (J,)*D coefficient tensor of oscillator i's equation.
    """
    Ws = []
    for i in range(D):
        L, R = i - 1, i + 1
        # linear forms dr and dl (a wall drops a term)
        dr = {}
        if R <= D - 1:
            dr[((R, 1),)] = dr.get(((R, 1),), 0.0) + 1.0
        dr[((i, 1),)] = dr.get(((i, 1),), 0.0) - 1.0
        dl = {}
        dl[((i, 1),)] = dl.get(((i, 1),), 0.0) + 1.0
        if L >= 0:
            dl[((L, 1),)] = dl.get(((L, 1),), 0.0) - 1.0

        lin = _poly_add(dr, _poly_scale(dl, -1.0))
        dr3 = _poly_mul(_poly_mul(dr, dr), dr)
        dl3 = _poly_mul(_poly_mul(dl, dl), dl)
        cub = _poly_scale(_poly_add(dr3, _poly_scale(dl3, -1.0)), beta)
        force = _poly_add(lin, cub)

        W = np.zeros((J,) * D)
        for mono, coeff in force.items():
            idx = [0] * D
            for var, pw in mono:
                idx[var] = pw
            W[tuple(idx)] += coeff
        Ws.append(W)
    return Ws

def weak_coefficients(Xs, f, t0, tM, D, J,
                      r_frac=1.0 / 60.0, degree=16):
    """TT-WSINDy (weak form) coefficient tensors, one row per output dim.

    Xs is (P, D, M): P trajectories, each of M snapshots spanning [t0, tM].
    The test-function radius is a fraction r_frac of that per-trajectory span,
    and both the library and the LHS are convolved one trajectory at a time, so
    no weak-form row straddles a trajectory boundary. FPUT is second order, so
    the test function is taken at order 2.
    """
    M = Xs.shape[2]
    phi, dphi = piecewise_polynomial((tM - t0) * r_frac, degree, t0, tM, M,
                                     order=2)
    Theta = feature_tensor(xu.flatten_trajectories(Xs), f, phi=phi, low_rank=True,
                           n_traj=Xs.shape[0])
    Y = -1 * np.concatenate(
        [correlate(X, np.expand_dims(dphi, axis=0), mode='valid') for X in Xs],
        axis=1).transpose()                                     # (P*Mp, D)
    return np.stack([xu.tt_pi_coeffs(Theta, Y[:, d], (J,) * D)
                     for d in range(D)])


def strong_coefficients(Xs, f, dt, D, J):
    """MANDy (strong form) coefficient tensors, one row per output dim.

    The LHS x'' is a 3-point central finite difference of each trajectory; the
    library is sampled pointwise at the interior snapshots. Xs is (P, D, M), as
    in weak_coefficients.
    """
    Xddot = (Xs[:, :, 2:] - 2 * Xs[:, :, 1:-1]
             + Xs[:, :, :-2]) / dt ** 2                         # (P, D, M-2)
    Theta = feature_tensor(xu.flatten_trajectories(Xs[:, :, 1:-1]), f, phi=None,
                           low_rank=True, n_traj=Xs.shape[0])
    return np.stack([xu.tt_pi_coeffs(Theta, Xddot[:, d, :].ravel(), (J,) * D)
                     for d in range(D)])

# Experiment
if __name__ == "__main__":

    # ----- parameters -----
    D = 4               # oscillators
    n_traj = 6          # independent trajectories (distinct initial conditions)
    M = 10000           # snapshots per trajectory
    dt = 0.015          # Snapshot spacing
    beta = 0.7          # cubic coupling
    amplitude = 3.0     # initial-displacement half-width (drives to full rank)
    r_frac = 1.0 / 120.0 # test-fn radius as a fraction of the per-trajectory span
    degree = 16         # test-function polynomial degree

    noise_levels = np.array([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 4e-1])
    n_trials = 40       # noise realizations averaged per level

    f = [lambda x: 1, lambda x: x, lambda x: x ** 2, lambda x: x ** 3]
    J = len(f)
    LABELS = ['', 'x{}', 'x{}^2', 'x{}^3']   # one per library function
    xu.check_labels(f, LABELS)

    DATA = "results/weakvstrongformFPUT.txt"
    recompute_data = True   # False skips the sweep and just
                            # replots what DATA already holds

    if recompute_data:

        # ----- generate the trajectories (positions only are used in regression) -----
        t, Xs, Vs = fput_trajectories(D, M, n_traj, dt=dt, beta=beta,
                                      amplitude=amplitude, seed=3)
        t0, tM = t[0], t[-1]
        Xflat = xu.flatten_trajectories(Xs)

        e0 = np.array([fpu_energy(Xs[p, :, 0], Vs[p, :, 0], beta)
                       for p in range(n_traj)])
        eN = np.array([fpu_energy(Xs[p, :, -1], Vs[p, :, -1], beta)
                       for p in range(n_traj)])
        Flin = np.stack([_fpu_force(Xflat[:, k], 0.0) for k in range(Xflat.shape[1])])
        Ffull = np.stack([_fpu_force(Xflat[:, k], beta) for k in range(Xflat.shape[1])])
        nl_frac = np.linalg.norm(Ffull - Flin) / np.linalg.norm(Ffull)
        rank, jD = xu.library_rank(Xflat, f, J)

        print("=" * 66)
        print("FPUT trajectories")
        print(f"  D={D} oscillators, {n_traj} trajectories x M={M} snapshots"
              f" = {n_traj * M} total, dt={dt}, T={tM:.1f}")
        print(f"  beta={beta}, amplitude={amplitude}")
        print(f"  energy drift |dH|/H0    : {np.max(np.abs(eN - e0) / np.abs(e0)):.3e}"
              f"  (max over trajectories)")
        print(f"  displacement range      : [{Xflat.min():.3f}, {Xflat.max():.3f}]")
        print(f"  nonlinear force fraction: {nl_frac:.3f}")
        print(f"  library rank            : {rank}/{jD}"
              f"  {'(full rank)' if rank == jD else '(RANK DEFICIENT -- raise amplitude/M)'}")
        print("=" * 66)

        Wtrue = np.stack(true_coeffs(D, J, beta))

        # ----- clean-data sanity report -----
        Ww0 = weak_coefficients(Xs, f, t0, tM, D, J, r_frac=r_frac, degree=degree)
        Ws0 = strong_coefficients(Xs, f, dt, D, J)
        print("clean-data relative coefficient error (no noise):")
        print(f"  TT-WSINDy (weak)  : {xu.rel_err(Ww0, Wtrue):.3e}")
        print(f"  MANDy     (strong): {xu.rel_err(Ws0, Wtrue):.3e}")
        xu.rule('-')
        xu.print_coeff_table(Wtrue, Ww0, Ws0, LABELS)
        xu.rule()

        # ----- noise sweep -----
        weak_err = np.zeros((noise_levels.size, n_trials))
        strong_err = np.zeros((noise_levels.size, n_trials))
        xstd = Xs.std()
        xfrob_normalized = np.linalg.norm(Xflat, ord='fro') / np.sqrt(Xs.size)

        print(f"noise sweep (relative coefficient error, mean over {n_trials} trials):")
        print(f"  {'noise sigma':>12}  {'TT-WSINDy':>12}  {'MANDy':>12}  {'ratio S/W':>10}")
        sweep_st = time()
        for i, sigma in enumerate(noise_levels):
            for tr in range(n_trials):
                rng = np.random.default_rng(1000 * i + tr)
                Xn = Xs + sigma * xfrob_normalized * rng.standard_normal(Xs.shape)
                Ww = weak_coefficients(Xn, f, t0, tM, D, J, r_frac=r_frac,
                                       degree=degree)
                Ws = strong_coefficients(Xn, f, dt, D, J)
                weak_err[i, tr] = xu.rel_err(Ww, Wtrue)
                strong_err[i, tr] = xu.rel_err(Ws, Wtrue)
            wm, sm = weak_err[i].mean(), strong_err[i].mean()
            print(f"  {sigma:>12.0e}  {wm:>12.3e}  {sm:>12.3e}  {sm / wm:>10.1f}")
        print(f"(noise sweep walltime: {time() - sweep_st:.1f}s)")
        print("=" * 66)

        # ----- save results -----
        os.makedirs("results", exist_ok=True)
        out = np.column_stack([noise_levels,
                               weak_err.mean(1), weak_err.std(1),
                               strong_err.mean(1), strong_err.std(1)])
        np.savetxt(DATA, out,
                   header=f"D={D} n_traj={n_traj} M={M} dt={dt} beta={beta} "
                          f"amplitude={amplitude} r_frac={r_frac:.4f}\n"
                          "noise weak_mean weak_std strong_mean strong_std")

    # the figure is always drawn from DATA, so a rerun and a replot agree
    if not os.path.exists(DATA):
        raise SystemExit(f"{DATA} not found -- set recompute_data = True and rerun")
    noise_levels, wm, ws, sm, ss = np.atleast_2d(np.loadtxt(DATA)).T

    # ----- plot -----
    floor = max(noise_levels[1] / 10, 1e-6)   # x-position for the sigma=0 point
    x_axis = np.where(noise_levels > 0, noise_levels, floor)

    # mean +- one standard deviation over trials
    ax = plt.figure(figsize=(7, 5)).gca()
    hw = ax.errorbar(x_axis, wm, yerr=ws, marker='o', capsize=3,
                     color='C0', ls='-', label='TT-WSINDy (weak form)')
    hs = ax.errorbar(x_axis, sm, yerr=ss, marker='s', capsize=3,
                     color='C1', ls='-',
                     label='MANDy (strong form, finite diff.)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'relative noise level $\sigma$')
    plt.ylabel('relative coefficient error')
    plt.title('Weak vs. strong form TT regression on FPUT '
              f'(D={D}, {n_traj} traj. x M={M})')
    plt.legend(handles=[hw, hs])
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"results/weakvstrongformFPUT.png", dpi=150)
    plt.show()
