import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import skimage.feature
from adjustText import adjust_text

from pca_density import create_kde_background, do_pca, load_df, pivot_and_impute, plot_streamplot, format_pca_axis_label, get_top_component_features
from pca_clusters import get_clusters, load_text_df

def plot_fig(ax, target_df, components, stance_cols, cfg):
    coords = target_df['coord_0d'].to_numpy()
    
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    contours, (xx, yy), log_density = create_kde_background(coords, ax, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    plot_streamplot(ax, target_df, xx, yy)

    top_features = get_top_component_features(components, stance_cols, n_features=2)
    
    # Format axis labels with top contributing features
    x_label = format_pca_axis_label(1, top_features['PC1'])
    y_label = format_pca_axis_label(2, top_features['PC2'])
    
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)

    peak_coords, topic_names = get_clusters(log_density, xx, yy, coords, target_df, cfg)

    texts = []
    for peak_coord, topic_name in zip(peak_coords, topic_names):
        ax.plot(peak_coord[0], peak_coord[1], 'rX', markersize=12, markeredgewidth=2)

        txt = ax.text(peak_coord[0], peak_coord[1], topic_name, fontsize=8, color='darkblue',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='blue', alpha=0.8))
        texts.append(txt)

    adjust_text(
        texts, 
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5),
        force_text=(2.0, 2.0),
        force_static=(1.0, 1.0)
    )

    return ax

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    print("Loading data...")
    filter_val = 'SeedName'

    if len(cfg.trend_paths) > 1:
        trend_name = 'noun_phrase_claims_kernelreg'
    else:
        trend_name = os.path.basename(cfg.trend_path)
    keywords = ['carbon', 'climate', 'energy', 'fossil', 'fuel', 'gas', 'oil', 'coal', 'solar', 'renewable', 'emissions', 'sustainability', 'environment', 'warming', 'greenhouse', 'net-zero', 'pipeline', 'nuclear']
    dir_name = f"{trend_name}/climate"

    df = pl.DataFrame()
    for trend_path in cfg.trend_paths:
        all_trend_path = os.path.join(trend_path, 'loaded_trends.parquet.zstd')
        if os.path.exists(all_trend_path):
            file_df = pl.read_parquet(all_trend_path)
        else:
            file_df = load_df(trend_path, 'SeedName')
            file_df.write_parquet(all_trend_path, compression='zstd')

        df = pl.concat([df, file_df], how='diagonal_relaxed')

    df = df.filter(pl.col('target').str.to_lowercase().str.contains('|'.join([rf"\b{k}\b" for k in keywords])))

    text_df = load_text_df(cfg, columns=['seed'])
    seed_df = text_df.select([
        pl.col('seed').struct.field('SeedName'),
        pl.col('seed').struct.field('MainType'),
        pl.col('seed').struct.field('SubType')
    ]).unique('SeedName')
    seed_df = seed_df.filter(pl.col('MainType').is_in(['politician', 'influencer']))
    df = df.join(seed_df, left_on='filter_value', right_on='SeedName', how='inner').drop('MainType')

    target_df, feature_cols, stance_cols, volume_cols = pivot_and_impute(df, impute_fancy=True)
    pca, coords, components = do_pca(target_df, stance_cols)
    target_df = target_df.with_columns(pl.Series(name='coord_0d', values=coords))

    # Create scatter plot of PCA coordinates colored by cluster
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_fig(ax, target_df, components, stance_cols, cfg)
    os.makedirs(f"./figs/{dir_name}", exist_ok=True)
    fig.savefig(f"./figs/{dir_name}/pca_clusters.png", dpi=300)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (year, ax) in enumerate(zip(target_df['createtime'].dt.year().unique().sort().to_list(), axes)):
        year_df = target_df.filter(pl.col('createtime').dt.year() == year)
        plot_fig(ax, year_df, components, stance_cols, cfg)
        ax.set_title(f"Year: {year}", fontsize=10)
    fig.savefig(f"./figs/{dir_name}/pca_clusters_by_year.png", dpi=300)

    
if __name__ == '__main__':
    main()