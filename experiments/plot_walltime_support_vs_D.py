"""
Redraw the walltime / support figures from the saved data files.

walltime_support_vs_D.py recomputes every D on every run, so this script exists
to redraw its figures without paying for the sweep again. It reads
    results/walltime_support_vs_D.txt   (main metrics, one row per D)
    results/support_sizes_vs_D.txt      (support-size funnel)
and hands them to that module's own make_figures(), so the figures are produced
by exactly the same code as a full run -- no plotting logic is duplicated here.

Which D appear is set by PLOT_EXCLUDE in walltime_support_vs_D.py.
"""
import os
import numpy as np
import walltime_support_vs_D as W

DATA = 'results/walltime_support_vs_D.txt'
SIZES = 'results/support_sizes_vs_D.txt'


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

    have = [int(d) for d in rows[:, 0]]
    drawn = [d for d in have if d not in W.PLOT_EXCLUDE]
    print(f'main data D = {have}')
    print(f'PLOT_EXCLUDE = {sorted(W.PLOT_EXCLUDE)}  ->  plotting D = {drawn}')

    W.make_figures(rows, sizes_rows)
