import datetime
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    trend_path = cfg.trend_path

    target_path = os.path.join(trend_path, f'{cfg.dim_reduction_method}_coords.parquet.zstd')
    target_df = pl.read_parquet(target_path, columns=['createtime', 'filter_value', 'coord_21d'])

    target_df = target_df.filter(pl.col('createtime') >= datetime.datetime(2022, 1, 1))\
        .filter(pl.col('filter_value') != '')

    # get var(diff(coord)) for each dimension
    coord_diff_var = target_df.sort(['filter_value', 'createtime'])\
        .with_columns([
            pl.col('coord_21d').arr.get(i).diff().over('filter_value').alias(f'dim_{i}_diff') for i in range(21)
        ])\
        .select([pl.col(f'dim_{i}_diff').var() for i in range(21)])\
        .to_numpy()[0]

    num_dims = 10

    pca_metadata_df = pl.read_parquet(os.path.join(trend_path, 'pca_metadata.parquet.zstd'))
    explained_variance_ratios = pca_metadata_df.sort('n_dims')['explained_variance_ratio'][-1].to_numpy()
    explained_variance_ratios = explained_variance_ratios[:num_dims]
    n_dims = np.arange(1, num_dims + 1)
    coord_diff_var = coord_diff_var[:num_dims]

    cum_sum = np.cumsum(explained_variance_ratios)

    print(f"2 components explain {cum_sum[1]:.2%} of variance")
    print(f"10 components explain {cum_sum[9]:.2%} of variance")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax_twin = ax.twinx()
    ax.set_zorder(ax_twin.get_zorder() + 1)
    ax.patch.set_visible(False)
    ln_cumsum = ax.plot(n_dims, cum_sum, label='Cumulative Explained Variance Ratio')
    ln_exvar = ax.plot(n_dims, explained_variance_ratios, label='Explained Variance Ratio')
    ln_diff = ax_twin.plot(n_dims, coord_diff_var, label='Variance of Diff Coord', color='green')
    ax.set_xlabel('Number of PCA Dimensions')
    ax.set_ylabel('Explained Variance Ratio')
    ax_twin.set_ylabel('Variance of Derivative')
    # combine legends
    lns = ln_cumsum + ln_exvar + ln_diff
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc='upper left')
    
    fig.tight_layout()
    fig.savefig(os.path.join('figs', 'pca_explained_variance.png'))

if __name__ == '__main__':
    main()