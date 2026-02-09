import os

from adjustText import adjust_text
import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

def plot_fig(axes, pca_df):
    for user_target_df in pca_df.partition_by('filter_value'):
        axes[0].plot(user_target_df['coord'].arr.get(0), user_target_df['coord'].arr.get(1), alpha=0.1)
    axes[0].set_title('PCA Trajectories')

    # plot moving average pca trajectories
    for ax_idx, window_size in zip([1, 2, 3], [50, 100, 200]):
        for user_target_df in pca_df.partition_by('filter_value'):
            axes[ax_idx].plot(
                user_target_df['coord'].arr.get(0).rolling_mean(window_size),
                user_target_df['coord'].arr.get(1).rolling_mean(window_size),
                alpha=0.1
            )
        axes[ax_idx].set_title(f'PCA Trajectories (MA {window_size})')


    return axes

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    print("Loading data...")

    trend_name = os.path.basename(cfg.trend_path)
    # keywords = ['carbon', 'climate', 'energy', 'fossil', 'fuel', 'gas', 'oil', 'coal', 'solar', 'renewable', 'emissions', 'sustainability', 'environment', 'warming', 'greenhouse', 'net-zero', 'pipeline', 'nuclear']
    dir_name = f"{trend_name}/all"

    pca_path = os.path.join(cfg.trend_path, 'pca_coords.parquet.zstd')
    pca_df = pl.read_parquet(pca_path, columns=['createtime', 'filter_value', 'coord_21d'])\
        .rename({'coord_21d': 'coord'})

    # Create scatter plot of PCA coordinates colored by cluster
    fig, axes = plt.subplots(ncols=4, figsize=(20, 5))
    plot_fig(axes, pca_df)
    os.makedirs(f"./figs/{dir_name}", exist_ok=True)
    fig.savefig(f"./figs/{dir_name}/trajectories.png", dpi=300)
    
if __name__ == '__main__':
    main()