"""Shared error-curve drawing for the weakvstrongform experiments.

Two styles show the same underlying thing -- how the relative coefficient error
is distributed over noise realizations at each sigma -- but they read very
differently:

  'errorbar'  mean +- one standard deviation, joined into a line. Needs only
              the summary statistics in results/weakvstrongform<System>.txt.
              Symmetric by construction, which is a poor fit for an error that
              is bounded below by zero and spans decades, but it is compact and
              two curves stay legible on one panel.

  'box'       the full per-trial distribution at each sigma: median, quartile
              box, 1.5 IQR whiskers, outliers as points. Needs the raw per-trial
              errors, which the experiments write alongside the summary as
              results/weakvstrongform<System>_trials.txt.

              A box summarizes n_trials numbers, so it says nothing an error bar
              does not when n_trials is 1 -- it degenerates to a single flat
              line. Check the n_trials each experiment ran at before reading
              anything into the boxes.

  'band'      the same per-trial distribution as 'box', drawn as a continuous
              median line inside ONE shaded band, spanning the interquartile
              range (25th-75th percentile) over trials -- see BAND_PCTL to
              widen it, or to switch it to the full min-max range. Nesting a
              second band inside the first only muddied the panels, since the
              two forms already overlap. Unlike a box plot this keeps the curve
              reading as a curve, so the two forms stay comparable at a glance
              across five decades of sigma, and unlike an error bar the band is
              free to be asymmetric -- which it is here, because these errors
              are bounded below and skewed. Also needs the per-trial file.

STYLE below is the toggle. Change it here to switch every figure at once, or
pass --box / --errorbar on the command line of any of the experiment scripts
(and of plot_weakvstrongform.py), which style_from_argv() picks up.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

# ---- the toggle -----------------------------------------------------------
STYLE = 'errorbar'          # 'errorbar' or 'box'
# ---------------------------------------------------------------------------

STYLES = ('errorbar', 'box', 'band')

# Box geometry on a LOG x axis. matplotlib draws a box in linear data units, so
# a constant width would look wildly uneven across decades; scaling both the
# width and the two forms' offsets by the position keeps them visually uniform.
BOX_FRAC = 0.34             # box width as a fraction of its x position
DODGE = 1.16                # multiplicative offset separating the two forms

# The single band drawn by the 'band' style: the percentile pair it spans, and
# how opaque it is. It has to stay readable where the two forms overlap, so it
# is kept light enough to see one through the other. Set BAND_PCTL to None for
# the full min-max range over trials instead.
BAND_PCTL = (25, 75)        # interquartile range
BAND_ALPHA = 0.25


def style_from_argv(argv=None, default=None):
    """Resolve the plot style, letting --box / --errorbar override the default.

    Parameters
    ----------
    argv : list of str, optional
        Defaults to sys.argv[1:].
    default : str, optional
        Style to use when no flag is given. Defaults to the module's STYLE.

    Returns
    -------
    style : {'errorbar', 'box'}
    """
    argv = sys.argv[1:] if argv is None else argv
    style = STYLE if default is None else default
    for a in argv:
        if a in ('--box', '--boxplot'):
            style = 'box'
        elif a in ('--band', '--bands'):
            style = 'band'
        elif a in ('--errorbar', '--errorbars'):
            style = 'errorbar'
    if style not in STYLES:
        raise ValueError(f"unknown plot style {style!r}, expected one of {STYLES}")
    return style


def _lighten(color, amount=0.74):
    """Blend a color toward white, for box fills that keep the edge readable."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def draw_series(ax, sigma, mean, std, trials=None, *, style='errorbar',
                color='C0', marker='o', ls='-', label=None, dodge=1.0):
    """Draw one form's error curve on ax, in the requested style.

    Parameters
    ----------
    ax : matplotlib Axes
    sigma : array (n_sigma,)
        Noise levels, the x positions. Must be positive (the axis is log).
    mean, std : array (n_sigma,)
        Per-sigma mean and standard deviation over trials. Used by 'errorbar'.
    trials : array (n_sigma, n_trials), optional
        Raw per-trial errors. Required by 'box', ignored by 'errorbar'.
    style : {'errorbar', 'box'}
    color : matplotlib color
        Line/edge color; boxes are filled with a lightened version of it.
    marker, ls : str
        Marker and line style, 'errorbar' only.
    label : str
        Legend label.
    dodge : float
        Multiplicative x offset, 'box' only, so two forms sit side by side
        rather than on top of each other. Ignored by 'errorbar'.

    Returns
    -------
    handle : artist suitable for ax.legend(handles=[...])
        An ErrorbarContainer for 'errorbar', a proxy Patch for 'box' (box plots
        produce no single artist that carries a label).
    """
    sigma = np.asarray(sigma, dtype=float)
    if style == 'errorbar':
        return ax.errorbar(sigma, mean, yerr=std, marker=marker, capsize=3,
                           color=color, ls=ls, label=label)

    if trials is None:
        raise ValueError(f"style={style!r} needs the per-trial errors "
                         "(results/weakvstrongform<System>_trials.txt)")
    trials = np.atleast_2d(np.asarray(trials, dtype=float))
    if trials.shape[0] != sigma.size:
        raise ValueError(f"trials has {trials.shape[0]} rows for "
                         f"{sigma.size} sigmas")

    if style == 'band':
        # median curve inside ONE band; the bounds are taken per sigma across
        # trials, so the band is free to be asymmetric about the median
        med = np.nanpercentile(trials, 50, axis=1)
        if BAND_PCTL is None:
            lo, hi = np.nanmin(trials, axis=1), np.nanmax(trials, axis=1)
        else:
            lo, hi = (np.nanpercentile(trials, q, axis=1) for q in BAND_PCTL)
        ax.fill_between(sigma, lo, hi, color=color, alpha=BAND_ALPHA, lw=0)
        line, = ax.plot(sigma, med, color=color, marker=marker, ls=ls,
                        lw=1.6, ms=4.5, label=label)
        return line

    pos = sigma * dodge
    face = _lighten(color)
    ax.boxplot([row[np.isfinite(row)] for row in trials],
               positions=pos, widths=BOX_FRAC * pos,
               patch_artist=True, manage_ticks=False,   # keep the log ticks
               boxprops=dict(facecolor=face, edgecolor=color, lw=1.0),
               medianprops=dict(color=color, lw=1.6),
               whiskerprops=dict(color=color, lw=1.0),
               capprops=dict(color=color, lw=1.0),
               flierprops=dict(marker='.', ms=3.5, mfc=color, mec='none',
                               alpha=0.65))
    return Patch(facecolor=face, edgecolor=color, label=label)


def save_trials(path, sigma, weak_err, strong_err, config=''):
    """Write the raw per-trial errors that the 'box' style needs.

    Columns are noise, then n_trials weak columns, then n_trials strong columns.
    """
    n_trials = np.shape(weak_err)[1]
    cols = (['noise']
            + [f'weak_t{k}' for k in range(n_trials)]
            + [f'strong_t{k}' for k in range(n_trials)])
    header = (f'{config}\nn_trials={n_trials}\n' if config
              else f'n_trials={n_trials}\n') + ' '.join(cols)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savetxt(path, np.column_stack([sigma, weak_err, strong_err]),
               header=header)


def read_trials(path):
    """Read a _trials.txt back as (weak, strong), each (n_sigma, n_trials).

    Returns None if the file does not exist, so a caller can fall back to the
    'errorbar' style rather than fail.
    """
    import os
    if not os.path.exists(path):
        return None
    data = np.atleast_2d(np.loadtxt(path))
    n_trials = (data.shape[1] - 1) // 2
    if n_trials < 1 or data.shape[1] != 1 + 2 * n_trials:
        raise ValueError(f'{path}: {data.shape[1]} columns is not 1 + 2*n_trials')
    return data[:, 1:1 + n_trials], data[:, 1 + n_trials:]


def fresh_from_argv(argv=None):
    """True if --fresh was passed, i.e. discard the trials already on disk.

    The experiments extend results/weakvstrongform<System>_trials.txt by
    default (see resume_trials), which is only the right thing to do when the
    trials on disk came from the same system and parameters. --fresh is the
    escape hatch after changing M, dt, or the model itself.
    """
    argv = sys.argv[1:] if argv is None else argv
    return any(a in ('--fresh', '--no-resume') for a in argv)


def resume_trials(path, sigma, n_trials, fresh=False):
    """Reload the noise realizations already computed, so a sweep can extend it.

    A noise sweep seeds trial `tr` at sigma index `i` with 1000 * i + tr, so
    trial indices name reproducible realizations: raising n_trials adds new
    realizations without disturbing the earlier ones, and there is no reason to
    recompute what a previous run already wrote to `path`.

    The count is per noise level rather than one number for the file, because
    the experiments checkpoint after every level: an interrupted sweep leaves
    the levels it finished full and the rest NaN, and resumes exactly there.

    Only the sigma grid is checked, since that is all the per-trial file
    records -- trials on disk are ASSUMED to come from the same system and
    parameters. Pass fresh=True (or --fresh on the command line, via
    fresh_from_argv) after changing anything else.

    Parameters
    ----------
    path : str
        A _trials.txt written by save_trials, possibly absent.
    sigma : array (n_sigma,)
        The noise levels this run will sweep.
    n_trials : int
        Trials this run wants at each level, reused ones included.
    fresh : bool
        Ignore `path` and start from scratch.

    Returns
    -------
    weak, strong : array (n_sigma, n_trials)
        Columns [0, done[i]) of row i reloaded from `path`, the rest NaN.
    done : int array (n_sigma,)
        Per level, the first trial index this run has to compute. All zeros
        when `path` is missing, was swept at other sigmas, or fresh=True.
    """
    sigma = np.asarray(sigma, dtype=float)
    weak = np.full((sigma.size, n_trials), np.nan)
    strong = np.full((sigma.size, n_trials), np.nan)
    done = np.zeros(sigma.size, dtype=int)
    if fresh:
        return weak, strong, done

    prev = read_trials(path)
    if prev is None:
        return weak, strong, done

    prev_sigma = np.atleast_2d(np.loadtxt(path))[:, 0]
    if prev_sigma.size != sigma.size or not np.allclose(prev_sigma, sigma):
        print(f"[resume] {path} was swept at different noise levels, "
              "recomputing every trial")
        return weak, strong, done

    prev_w, prev_s = prev
    n_prev = min(prev_w.shape[1], n_trials)
    # a level counts as done up to its first gap, so a half-written level is
    # recomputed from the gap on rather than trusted past it
    ok = np.isfinite(prev_w[:, :n_prev]) & np.isfinite(prev_s[:, :n_prev])
    for i, row in enumerate(ok):
        done[i] = n_prev if row.all() else int(np.argmin(row))
        weak[i, :done[i]] = prev_w[i, :done[i]]
        strong[i, :done[i]] = prev_s[i, :done[i]]

    if done.max():
        lo, hi = done.min(), done.max()
        span = f"{lo}" if lo == hi else f"{lo}-{hi}"
        print(f"[resume] reusing {span} trial(s) per level from {path}"
              + (f", computing up to {n_trials - lo} more" if lo < n_trials
                 else ""))
    return weak, strong, done
