import os

import hydra
import matplotlib.pyplot as plt
import polars as pl

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    dims = [0]
    max_filter_values = 3
    out_path = os.path.join('figs', 'rolling_mean_comparison.png')

    target_path = os.path.join(cfg.trend_path, 'pca_coords.parquet.zstd')
    target_df = pl.read_parquet(target_path, columns=['createtime', 'filter_value', 'coord_21d'])

    target_df = target_df \
        .filter(pl.col('filter_value') != '') \
        .select(['createtime', 'filter_value', 'coord_21d']) \
        .sort(['filter_value', 'createtime']) \
        .rename({'coord_21d': 'x0'})

    # Extract individual dimension columns
    raw_df = target_df.with_columns([
        pl.col('x0').arr.get(i).alias(f'x0_{i}') for i in dims
    ])

    # Apply rolling mean smoothing (same operation as nn_potential.py lines 61-66)
    smoothed_df = raw_df \
        .rolling('createtime', period=f'{cfg.rolling_mean_window}d', group_by='filter_value') \
        .agg([pl.col(f'x0_{i}').mean() for i in dims])

    # Pick a subset of filter_values to plot
    filter_values = smoothed_df['filter_value'].unique().sort().head(max_filter_values).to_list()

    fig, axes = plt.subplots(
        len(dims), 1,
        figsize=(12, 4 * len(dims)),
        sharex=True,
        squeeze=False,
    )

    for dim_idx, dim in enumerate(dims):
        ax = axes[dim_idx, 0]
        col = f'x0_{dim}'

        for fv in filter_values:
            raw_fv = raw_df.filter(pl.col('filter_value') == fv).sort('createtime')
            smooth_fv = smoothed_df.filter(pl.col('filter_value') == fv).sort('createtime')

            ax.plot(
                raw_fv['createtime'].to_list(),
                raw_fv[col].to_list(),
                alpha=0.3, linewidth=0.5,
                label=f'{fv} (raw)',
            )
            ax.plot(
                smooth_fv['createtime'].to_list(),
                smooth_fv[col].to_list(),
                linewidth=1.5,
                label=f'{fv} (smoothed {cfg.rolling_mean_window}d)',
            )

        ax.set_ylabel(f'PCA dim {dim}')
        ax.legend(fontsize='small')

    axes[-1, 0].set_xlabel('Date')
    fig.suptitle(f'Rolling mean smoothing ({cfg.rolling_mean_window}-day window)')
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == '__main__':
    main()
