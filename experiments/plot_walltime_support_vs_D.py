"""
Redraw the walltime / support figures from the saved data files.

walltime_support_vs_D.py recomputes every D on every run, so this script exists
to redraw its figures without paying for the sweep again. It reads
    results/walltime_support_vs_D<suffix>.txt   (main metrics, one row per D)
    results/support_sizes_vs_D<suffix>.txt      (support-size funnel)
where <suffix> is walltime_support_vs_D.FILE_SUFFIX (set by its NOISE_LEVEL),
so the redraw reads the same files a full run wrote. Pass a suffix as argv[1]
to redraw a different variant, e.g. '' for the clean-data files.
and hands them to that module's own make_figures(), so the figures are produced
by exactly the same code as a full run -- no plotting logic is duplicated here.

Which D appear is set by PLOT_EXCLUDE in walltime_support_vs_D.py.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import numpy as np
import walltime_support_vs_D as W

SUFFIX = sys.argv[1] if len(sys.argv) > 1 else W.FILE_SUFFIX
DATA = os.path.join(_HERE, 'results', f'walltime_support_vs_D{SUFFIX}.txt')
SIZES = os.path.join(_HERE, 'results', f'support_sizes_vs_D{SUFFIX}.txt')


def load(path):
    """Load a results file as a 2-D array of rows, or None if absent/empty."""
    if not os.path.exists(path):
        return None
    rows = np.atleast_2d(np.loadtxt(path))
    return rows if rows.size else None


if __name__ == '__main__':

    rows = load(DATA)
    if rows is None:
        raise SystemExit(f'{DATA} not found -- run walltime_support_vs_D.py first')
    sizes_rows = load(SIZES)
    if sizes_rows is None:
        raise SystemExit(f'{SIZES} not found -- run walltime_support_vs_D.py first')

    print(f'read {DATA}')
    have = [int(d) for d in rows[:, 0]]
    drawn = [d for d in have if d not in W.PLOT_EXCLUDE]
    print(f'main data D = {have}')
    print(f'PLOT_EXCLUDE = {sorted(W.PLOT_EXCLUDE)}  ->  plotting D = {drawn}')

    W.make_figures(rows, sizes_rows)
