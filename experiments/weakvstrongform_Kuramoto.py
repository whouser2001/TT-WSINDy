"""
Weak form (TT-WSINDy) vs. strong form (MANDy) coefficient accuracy on the Kuramoto model.
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

def kuramoto(x, t, omega, K, h):
    """Kuramoto right-hand side for arbitrary d.

    Parameters
    ----------
    x : ndarray (d,)
        Phases. Unwrapped (not reduced mod 2 pi); see kuramoto_trajectory.
    t : float
        Time, unused (autonomous), present for odeint's signature.
    omega : ndarray (d,)
        Natural frequencies.
    K : float
        Coupling strength.
    h : float
        Pinning / forcing amplitude.

    Returns
    -------
    dx : ndarray (d,)
        Phase velocities.
    """
    S, C = np.sin(x), np.cos(x)
    # mean-field form of the coupling: exact for any d, and O(d) rather than
    # O(d^2). The j = i term contributes nothing, so it is left in.
    return omega + (K / x.size) * (C * S.sum() - S * C.sum()) + h * S


def gen_kuramoto(x0, t, omega, K, h):
    """Integrate the Kuramoto model to a d x M data matrix."""
    return odeint(kuramoto, x0, t, args=(omega, K, h),
                  rtol=RTOL, atol=ATOL).T


def kuramoto_trajectory(d, M, dt=0.01, K=0.5, h=0.5, omega_min=-5.0,
                        omega_max=5.0, seed=0, omega=None):
    """Generate one Kuramoto trajectory of d oscillators, uniformly sampled.

    Frequencies are equidistant on [omega_min, omega_max] unless an omega is
    supplied, and the initial phases are drawn uniformly from (-pi, pi].

    The spread must be wide enough, and K small enough, to keep the phases
    from locking, which would leave the {1, sin, cos} library rank deficient.
    The orbit also needs enough samples and a long enough span to reach full
    rank; check the rank the experiment prints when changing d. With an odd d
    the grid contains omega = 0 exactly, so keep h below the smallest
    |omega_i|.

    Parameters
    ----------
    d : int
        Number of oscillators (arbitrary; the right-hand side is O(d)).
    M : int
        Number of recorded snapshots, spaced by dt.
    dt : float
        Spacing between recorded snapshots, not an integrator step: odeint
        adapts its own.
    K : float
        Coupling strength.
    h : float
        Pinning / forcing amplitude.
    omega_min, omega_max : float
        Endpoints of the equidistant frequency grid.
    seed : int
        RNG seed for the initial phases (the frequencies are deterministic).
    omega : ndarray (d,), optional
        Frequencies to use instead of drawing them. Pooled trajectories must
        share one omega, or they are not samples of the same system.

    Returns
    -------
    t : ndarray (M,)
        Sample times.
    X : ndarray (d, M)
        Phases along the trajectory, unwrapped, so they grow without bound. Do
        not reduce them mod 2 pi: a jump would wreck both the finite difference
        and the weak-form convolution. Because |X| grows with the time span,
        the noise sweep below perturbs the phases by an absolute number of
        radians rather than scaling by rms(X).
    omega : ndarray (d,)
        The natural frequencies used, needed to state the true coefficients.
    """
    rng = np.random.default_rng(seed)
    if omega is None:
        omega = np.linspace(omega_min, omega_max, d)
    x0 = rng.uniform(-np.pi, np.pi, d)
    t = np.arange(M) * dt
    return t, gen_kuramoto(x0, t, omega, K, h), omega


def kuramoto_trajectories(d, M, n_traj, dt=0.01, K=0.5, h=0.5,
                          omega_min=-5.0, omega_max=5.0, seed=0):
    """Generate n_traj Kuramoto trajectories of one system, uniformly sampled.

    The equidistant frequencies are shared by every trajectory, which differ
    only in their random initial phases (drawn with seed + p).

    Returns
    -------
    t : ndarray (M,)
        Sample times, common to all trajectories.
    positions : ndarray (n_traj, d, M)
    omega : ndarray (d,)
    """
    omega = np.linspace(omega_min, omega_max, d)
    trajs = [kuramoto_trajectory(d, M, dt=dt, K=K, h=h, seed=seed + p,
                                 omega=omega) for p in range(n_traj)]
    return trajs[0][0], np.stack([X for _, X, _ in trajs]), omega

def true_coeffs(D, J, omega, K, h):
    """Exact coefficient tensor of the Kuramoto right-hand side, per output dim.

    Parameters
    ----------
    D : int
        Number of oscillators.
    J : int
        Number of candidate functions per dimension, assumed to be
        f = {1, sin, cos}; J >= 3 is required.
    omega : ndarray (D,)
        Natural frequencies.
    K, h : float
        Coupling strength and pinning amplitude.

    Returns
    -------
    Ws : list of ndarray
        Ws[i] is the (J,)*D coefficient tensor of oscillator i's equation.
        Each equation has 2D nonzeros.

    Raises
    ------
    ValueError
        If J < 3, since the model needs all of {1, sin, cos}.
    """
    if J < 3:
        raise ValueError(
            f"Kuramoto needs f = {{1, sin, cos}}, so J >= 3 (got J={J})"
        )

    Ws = []
    for i in range(D):
        W = np.zeros((J,) * D)

        idx = [0] * D
        W[tuple(idx)] += omega[i]                   # constant

        idx = [0] * D; idx[i] = 1
        W[tuple(idx)] += h                          # pinning

        for j in range(D):
            if j == i:
                continue                            # cancels analytically
            idx = [0] * D; idx[j] = 1; idx[i] = 2
            W[tuple(idx)] += K / D                  # coupling
            idx = [0] * D; idx[j] = 2; idx[i] = 1
            W[tuple(idx)] -= K / D

        Ws.append(W)
    return Ws

def weak_coefficients(Xs, f, t0, tM, D, J,
                      r_frac=1.0 / 60.0, degree=16):
    """TT-WSINDy (weak form) coefficient tensors, one row per output dim.

    Xs is (P, D, M): P trajectories, each of M snapshots spanning [t0, tM].
    The test-function radius is a fraction r_frac of that per-trajectory span,
    and both the library and the LHS are convolved one trajectory at a time, so
    no weak-form row straddles a trajectory boundary.
    """
    M = Xs.shape[2]
    phi, dphi = piecewise_polynomial((tM - t0) * r_frac, degree, t0, tM, M,
                                     order=1)
    Theta = feature_tensor(xu.flatten_trajectories(Xs), f, phi=phi, low_rank=True,
                           n_traj=Xs.shape[0])
    Y = -1 * np.concatenate(
        [correlate(X, np.expand_dims(dphi, axis=0), mode='valid') for X in Xs],
        axis=1).transpose()                                     # (P*Mp, D)
    return np.stack([xu.tt_pi_coeffs(Theta, Y[:, d], (J,) * D)
                     for d in range(D)])


def strong_coefficients(Xs, f, dt, D, J):
    """MANDy (strong form) coefficient tensors, one row per output dim.

    The LHS x' is a 3-point central finite difference of each trajectory; the
    library is sampled pointwise at the interior snapshots. Xs is (P, D, M), as
    in weak_coefficients.
    """
    Xdot = (Xs[:, :, 2:] - Xs[:, :, :-2]) / (2 * dt)            # (P, D, M-2)
    Theta = feature_tensor(xu.flatten_trajectories(Xs[:, :, 1:-1]), f, phi=None,
                           low_rank=True, n_traj=Xs.shape[0])
    return np.stack([xu.tt_pi_coeffs(Theta, Xdot[:, d, :].ravel(), (J,) * D)
                     for d in range(D)])


# Experiment
if __name__ == "__main__":

    # ----- parameters -----
    D = 4               # oscillators
    n_traj = 5          # independent trajectories (distinct initial phases)
    M = 10000           # snapshots per trajectory
    dt = 0.01           # snapshot spacing
    K = 2               # coupling strength
    h = 0.2             # pinning amplitude
    omega_min = -5.0    # frequencies equidistant on [omega_min, omega_max],
    omega_max = 5.0     # spread wide to keep the phases unlocked
    r_frac = 1.0 / 120.0 # test-fn radius as a fraction of the per-trajectory span
    degree = 16         # test-function polynomial degree

    noise_levels = np.array([1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 4e-1])
    n_trials = 40       # noise realizations averaged per level

    f = [lambda x: 1, lambda x: np.sin(x), lambda x: np.cos(x)]
    J = len(f)
    LABELS = ['', 'sin(x{})', 'cos(x{})']   # one per library function
    xu.check_labels(f, LABELS)

    DATA = "results/weakvstrongformKuramoto.txt"
    recompute_data = True   # False skips the sweep and just
                            # replots what DATA already holds

    if recompute_data:

        # ----- generate the trajectories -----
        t, Xs, omega = kuramoto_trajectories(D, M, n_traj, dt=dt, K=K, h=h,
                                             omega_min=omega_min,
                                             omega_max=omega_max, seed=3)
        t0, tM = t[0], t[-1]
        Xflat = xu.flatten_trajectories(Xs)

        order_param = np.mean([np.abs(np.exp(1j * Xs[p]).mean(axis=0)).mean()
                               for p in range(n_traj)])
        Ffull = np.stack([kuramoto(Xflat[:, k], 0.0, omega, K, h)
                          for k in range(Xflat.shape[1])])
        coupling_frac = np.linalg.norm(Ffull - omega) / np.linalg.norm(Ffull)
        rank, jD = xu.library_rank(Xflat, f, J)

        print("=" * 66)
        print("Kuramoto trajectories")
        print(f"  D={D} oscillators, {n_traj} trajectories x M={M} snapshots"
              f" = {n_traj * M} total, dt={dt}, T={tM:.1f}")
        print(f"  K={K}, h={h}, omega equidistant on "
              f"[{omega_min:g}, {omega_max:g}]: "
              + ", ".join(f"{w:g}" for w in omega))
        print(f"  order parameter r       : {order_param:.3f}"
              f"  {'(unlocked)' if order_param < 0.9 else '(LOCKED -- lower K)'}")
        print(f"  phase range             : [{Xflat.min():.3f}, {Xflat.max():.3f}]")
        print(f"  coupling force fraction : {coupling_frac:.3f}")
        print(f"  library rank            : {rank}/{jD}"
              f"  {'(full rank)' if rank == jD else '(RANK DEFICIENT -- raise M/T)'}")
        print("=" * 66)

        Wtrue = np.stack(true_coeffs(D, J, omega, K, h))

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
        # sigma is an absolute phase perturbation in radians, since the phases
        # are unwrapped and grow with the time span
        weak_err = np.zeros((noise_levels.size, n_trials))
        strong_err = np.zeros((noise_levels.size, n_trials))

        print(f"noise sweep (relative coefficient error, mean over {n_trials} trials):")
        print(f"  {'noise sigma':>12}  {'TT-WSINDy':>12}  {'MANDy':>12}  {'ratio S/W':>10}")
        sweep_st = time()
        for i, sigma in enumerate(noise_levels):
            for tr in range(n_trials):
                rng = np.random.default_rng(1000 * i + tr)
                Xn = Xs + sigma * rng.standard_normal(Xs.shape)
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
                   header=f"D={D} n_traj={n_traj} M={M} dt={dt} K={K} h={h} "
                          f"omega_min={omega_min} omega_max={omega_max} "
                          f"r_frac={r_frac:.4f}\n"
                          "noise weak_mean weak_std strong_mean strong_std")

    # the figure is always drawn from DATA, so a rerun and a replot agree
    if not os.path.exists(DATA):
        raise SystemExit(f"{DATA} not found -- set recompute_data = True and rerun")
    noise_levels, wm, ws, sm, ss = np.atleast_2d(np.loadtxt(DATA)).T

    # ----- plot -----

    # mean +- one standard deviation over trials
    ax = plt.figure(figsize=(7, 5)).gca()
    hw = ax.errorbar(noise_levels, wm, yerr=ws, marker='o', capsize=3,
                     color='C0', ls='-', label='TT-WSINDy (weak form)')
    hs = ax.errorbar(noise_levels, sm, yerr=ss, marker='s', capsize=3,
                     color='C1', ls='-',
                     label='MANDy (strong form, finite diff.)')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-6, 1e0)
    plt.xlabel(r'phase noise $\sigma$ (radians)')
    plt.ylabel('relative coefficient error')
    plt.title('Weak vs. strong form TT regression on Kuramoto '
              f'(D={D}, {n_traj} traj. x M={M})')
    plt.legend(handles=[hw, hs])
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"results/weakvstrongformKuramoto.png", dpi=150)
    plt.show()
