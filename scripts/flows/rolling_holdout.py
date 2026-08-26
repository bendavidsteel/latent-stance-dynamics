"""Run the nested-split evaluation across rolling holdout windows and plot it.

One holdout window gives a single snapshot of out-of-time performance; running
several describes how it decays as the unseen period lengthens, which is the
question a landscape model of drifting opinion actually has to answer.

Each window is a separate training run: the split boundary moves, so both the
latent representation and the landscape model have to be refitted.

    python rolling_holdout.py -- sigma=0.3 n_dims=6      # extra hydra overrides
    python rolling_holdout.py --plot-only
"""

import argparse
import os
import subprocess
import sys

import polars as pl

import splits

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# train_in is the fit, not a test; the other five are what the design buys.
SCENARIOS = ['val_in', 'test_in', 'train_out', 'val_out', 'test_out']
LABELS = {
    'val_in': 'unseen trajectory, seen time',
    'test_in': 'unseen trajectory, seen time (test)',
    'train_out': 'seen trajectory, unseen time',
    'val_out': 'both unseen (val)',
    'test_out': 'both unseen (test)',
}


def run_window(holdout_days, overrides):
    cmd = [sys.executable, os.path.join(HERE, 'nn_potential.py'),
           f'split.holdout_days={holdout_days}'] + list(overrides)
    print(f'\n=== holdout {holdout_days}d ===\n{" ".join(cmd)}', flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def collect(out_root, prefix='step'):
    """Gather every scenario_metrics file written under out/."""
    name = f'scenario_metrics_{prefix}.parquet.zstd'
    paths = [os.path.join(r, name) for r, _, fs in os.walk(out_root) if name in fs]
    if not paths:
        raise SystemExit(f'no {name} under {out_root} -- run the windows first')
    return pl.concat([pl.read_parquet(p) for p in paths], how='diagonal_relaxed')


def plot(df, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for scenario in SCENARIOS:
        rows = df.filter(pl.col('scenario') == scenario).sort('holdout_days')
        if len(rows) == 0:
            continue
        ax.plot(rows['holdout_days'], rows['skill_score'], marker='o',
                label=f'{scenario} — {LABELS[scenario]}')
    ax.axhline(0, color='0.6', lw=0.8, ls='--')
    ax.set_xscale('log')
    ax.set_xticks(list(splits.HOLDOUT_DAYS))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('holdout window (days)')
    ax.set_ylabel('skill vs no-movement baseline')
    ax.set_title('Landscape-model skill by scenario and holdout window')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f'wrote {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, nargs='+', default=list(splits.HOLDOUT_DAYS))
    ap.add_argument('--prefix', default='step')
    ap.add_argument('--out-root', default=os.path.join(REPO, 'out'))
    ap.add_argument('--fig', default=os.path.join(REPO, 'figs', 'rolling_holdout_skill.png'))
    ap.add_argument('--plot-only', action='store_true')
    args, overrides = ap.parse_known_args()
    overrides = [o for o in overrides if o != '--']

    if not args.plot_only:
        for days in args.windows:
            run_window(days, overrides)

    df = collect(args.out_root, args.prefix).filter(
        pl.col('holdout_days').is_in(args.windows))
    print(df.select(['holdout_days', 'scenario', 'skill_score', 'model_mse',
                     'baseline_mse', 'n']).sort(['holdout_days', 'scenario']))
    plot(df, args.fig)


if __name__ == '__main__':
    main()
