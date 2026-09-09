"""
Read the four weakvstrongform result files and plot them in one figure.

Panels are drawn as error bars (mean +- std), as per-sigma box plots, or as a
median curve inside one percentile band -- errorplot.STYLE is the toggle, or pass
--errorbar / --box / --band on this script's command line. The box and band
styles additionally need results/weakvstrongform<System>_trials.txt, which the
experiments write next to the summary; a system missing that file falls back to
error bars with a note. Each style writes its own PNG, so they coexist.

Each experiment writes results/weakvstrongform<System>.txt with a config line,
a column header, and rows of
    noise  weak_mean  weak_std  strong_mean  strong_std
so this script only reads and plots -- it runs no regressions of its own. Run
the four experiments first; any missing file leaves its panel empty rather than
failing.

The y axis is the same dimensionless quantity in every panel. L96, Kuramoto and
Chua are pinned to a common 1e-6..1e0 window; FPUT is scaled to its own data
instead, because its strong-form error runs to ~6e2 and folding that into a
shared window flattens every other panel. The x axes are NOT shared: Kuramoto's
sigma is an absolute phase perturbation in radians, while the others scale sigma
by rms(X).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import errorplot

RESULTS = 'results'
# one output per style, so the versions coexist rather than overwrite
OUT_BY_STYLE = {
    'errorbar': os.path.join(RESULTS, 'weakvstrongform_all.png'),
    'box': os.path.join(RESULTS, 'weakvstrongform_all_box.png'),
    'band': os.path.join(RESULTS, 'weakvstrongform_all_band.png'),
}

# (file stem, panel title, x-axis label, y limits or None to fit that panel)
PINNED = (1e-6, 1e0)
SYSTEMS = [
    ('FPUT', 'Fermi-Pasta-Ulam-Tsingou',
     r'relative noise level $\sigma$', None),
    ('L96', 'Lorenz 96',
     r'relative noise level $\sigma$', PINNED),
    ('Kuramoto', 'Kuramoto Model',
     r'phase noise $\sigma$ (radians)', PINNED),
    ('Chua', "Chua's circuit",
     r'relative noise level $\sigma$', PINNED),
]

WEAK_COLOR, STRONG_COLOR = 'C0', 'C3'


def read_results(stem):
    """Read one results file.

    Returns
    -------
    (config, sigma, weak_mean, weak_std, strong_mean, strong_std) or None if
    the file does not exist. `config` is the first comment line, i.e. the
    parameters the experiment was run at.
    """
    path = os.path.join(RESULTS, f'weakvstrongform{stem}.txt')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        config = fh.readline().lstrip('#').strip()
    data = np.atleast_2d(np.loadtxt(path))
    sigma, wm, ws, sm, ss = (data[:, i] for i in range(5))
    return config, sigma, wm, ws, sm, ss


def wrap(text, width=52):
    """Greedy wrap, so a long parameter line fits under a panel title."""
    lines, cur = [], ''
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f'{cur} {word}'.strip()
    if cur:
        lines.append(cur)
    return '\n'.join(lines)


if __name__ == '__main__':

    style = errorplot.style_from_argv()
    loaded = {s[0]: read_results(s[0]) for s in SYSTEMS}
    trials = {s[0]: errorplot.read_trials(
                  os.path.join(RESULTS, f'weakvstrongform{s[0]}_trials.txt'))
              for s in SYSTEMS}
    if style in ('box', 'band'):
        no_trials = [s for s, t in trials.items()
                     if t is None and loaded[s] is not None]
        if no_trials:
            print(f"no per-trial file for: {', '.join(no_trials)} -- those "
                  f"panels fall back to error bars (rerun the experiment)")
    missing = [s for s, r in loaded.items() if r is None]
    if missing:
        print(f"no results for: {', '.join(missing)} "
              f"(run the corresponding weakvstrongform_*.py first)")

    # ----- console summary of what was read -----
    print(f"  {'system':>10}  {'sigmas':>7}  {'weak @ min sigma':>17}"
          f"  {'best S/W':>9}  {'at sigma':>9}")
    for stem, *_ in SYSTEMS:
        r = loaded[stem]
        if r is None:
            print(f"  {stem:>10}  {'--':>7}")
            continue
        _, sigma, wm, _, sm, _ = r
        ratio = sm / wm
        k = int(np.argmax(ratio))
        print(f"  {stem:>10}  {len(sigma):>7}  {wm[0]:>17.3e}"
              f"  {ratio[k]:>9.1f}  {sigma[k]:>9.0e}")

    def data_ylim(r):
        """Decade-rounded limits enclosing one panel's own weak+strong curves."""
        vals = np.concatenate([r[2], r[4]])
        return (10 ** np.floor(np.log10(vals.min()) - 0.2),
                10 ** np.ceil(np.log10(vals.max()) + 0.2))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
    handles = None

    for ax, (stem, title, xlabel, fixed) in zip(axes.ravel(), SYSTEMS):
        r = loaded[stem]
        if r is None:
            ax.set_title(f'{title}\n(no results file)', fontsize=11)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(*(fixed or PINNED))
            ax.grid(True, which='both', ls=':', alpha=0.5)
            continue
        config, sigma, wm, ws, sm, ss = r

        tw, ts = trials[stem] if trials[stem] is not None else (None, None)
        # a panel with no per-trial file still draws, as error bars
        panel_style = style if tw is not None else 'errorbar'
        hw = errorplot.draw_series(ax, sigma, wm, ws, tw, style=panel_style,
                                   color=WEAK_COLOR, marker='o',
                                   label='Weak form TT regression',
                                   dodge=1.0 / errorplot.DODGE)
        hs = errorplot.draw_series(ax, sigma, sm, ss, ts, style=panel_style,
                                   color=STRONG_COLOR, marker='s',
                                   label='MANDy (strong form TT regression)',
                                   dodge=errorplot.DODGE)
        # relative error 1 means the estimate carries no more signal than zero
        #hf = ax.axhline(1.0, color='0.4', ls='--', lw=1.0,
        #                label='rel. err = 1 (no recovery)')
        if handles is None:
            handles = [hw, hs]

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(*(fixed or data_ylim(r)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel('relative coefficient error')
        #ax.set_title(f'{title}\n{wrap(config)}', fontsize=9, loc='left')
        ax.set_title(f'{title}', fontsize=10, loc='left')
        ax.grid(True, which='both', ls=':', alpha=0.5)

    if handles is not None:
        fig.legend(handles=handles, loc='lower center', ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
    #fig.suptitle('Weak form (TT-WSINDy) vs. strong form (MANDy) '
    #             'coefficient accuracy', fontsize=13)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    os.makedirs(RESULTS, exist_ok=True)
    out_path = OUT_BY_STYLE[style]
    fig.savefig(out_path, dpi=150)
    print(f"\nwrote {out_path}  (style: {style})")
    plt.show()
