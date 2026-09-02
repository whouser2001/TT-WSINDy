"""Read the four weakvstrongform result files and plot them in one figure.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS = 'results'
OUT = os.path.join(RESULTS, 'weakvstrongform_all.png')

# (file stem, panel title, x-axis label, y limits or None to fit the data)
PINNED = (1e-6, 1e1)
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

WEAK_KW = dict(marker='o', capsize=2, lw=1.3, ms=3.5)
STRONG_KW = dict(marker='s', capsize=2, lw=1.3, ms=3.5, ls='--')


def dim_colors(D):
    lo, hi = (0.5, 0.9) if D > 1 else (0.75, 0.75)
    return (plt.cm.Blues(np.linspace(lo, hi, D)),
            plt.cm.Reds(np.linspace(lo, hi, D)))

def read_results(stem):
    path = os.path.join(RESULTS, f'weakvstrongform{stem}.txt')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        config = fh.readline().lstrip('#').strip()
    data = np.atleast_2d(np.loadtxt(path))
    D, rem = divmod(data.shape[1] - 1, 4)
    if rem or D < 1:
        raise ValueError(f'{path}: {data.shape[1]} columns is not 1 + 4D -- '
                         f'rerun weakvstrongform_{stem}.py to get the '
                         f'per-dimension format')
    sigma = data[:, 0]
    blocks = [data[:, 1 + k * D:1 + (k + 1) * D] for k in range(4)]
    return (config, sigma, *blocks)

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

    loaded = {s[0]: read_results(s[0]) for s in SYSTEMS}
    missing = [s for s, r in loaded.items() if r is None]
    if missing:
        print(f"no results for: {', '.join(missing)} "
              f"(run the corresponding weakvstrongform_*.py first)")

    # ----- console summary of what was read -----
    # the weak column is the RANGE over output dimensions at the smallest
    # sigma, and the ratio is taken on the dimension-averaged curves
    print(f"  {'system':>10}  {'D':>3}  {'sigmas':>6}"
          f"  {'weak @ min sigma (min..max over dims)':>38}"
          f"  {'best S/W':>9}  {'at sigma':>9}")
    for stem, *_ in SYSTEMS:
        r = loaded[stem]
        if r is None:
            print(f"  {stem:>10}  {'--':>3}")
            continue
        _, sigma, wm, _, sm, _ = r
        D = wm.shape[1]
        ratio = sm.mean(1) / wm.mean(1)
        k = int(np.argmax(ratio))
        span = f'{wm[0].min():.3e} .. {wm[0].max():.3e}'
        print(f"  {stem:>10}  {D:>3}  {len(sigma):>6}  {span:>38}"
              f"  {ratio[k]:>9.1f}  {sigma[k]:>9.0e}")

    def data_ylim(r):
        """Decade-rounded limits enclosing one panel's own weak+strong curves."""
        vals = np.concatenate([r[2].ravel(), r[4].ravel()])
        return (10 ** np.floor(np.log10(vals.min()) - 0.2),
                10 ** np.ceil(np.log10(vals.max()) + 0.2))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))

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
        D = wm.shape[1]
        wcol, scol = dim_colors(D)

        for d in range(D):
            ax.errorbar(sigma, wm[:, d], yerr=ws[:, d], color=wcol[d],
                        label=rf'weak $x_{{{d}}}$', **WEAK_KW)
        for d in range(D):
            ax.errorbar(sigma, sm[:, d], yerr=ss[:, d], color=scol[d],
                        label=rf'strong $x_{{{d}}}$', **STRONG_KW)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(*(fixed or data_ylim(r)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel('relative coefficient error')
        ax.set_title(f'{title}', fontsize=10, loc='left')
        ax.grid(True, which='both', ls=':', alpha=0.5)
        # the two columns line the dimensions of one form up under each other
        ax.legend(ncol=2, fontsize=7, loc='lower right', framealpha=0.85,
                  columnspacing=1.0, handlelength=1.8)

    fig.suptitle('Weak form (TT-WSINDy) vs. strong form (MANDy) '
                 'coefficient accuracy, per output dimension', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(RESULTS, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"\nwrote {OUT}")
    plt.show()
