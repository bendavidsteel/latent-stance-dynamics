import os
import numpy as np
import hydra
import matplotlib.pyplot as plt
import polars as pl
from adjustText import adjust_text
from pca_density import load_df, pivot_and_impute, do_pca, create_kde_background, get_top_component_features, format_pca_axis_label

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    russia_state_media_seed_names = [
        'Komsomolskaya Pravda',
        'Strategic Culture Foundation',
        'TASS',
        'RT en français',
        'RIA Novosti',
        'Sputnik News',
        'Life News',
        'RT'
    ]
    tenet_influencer_seed_names = ['Benny Johnson', 'Tim Pool', 'Matt Christiansen', 'Tayler Hansen', 'Dave Rubin', 'Lauren Southern']
    
    target_path = os.path.join(cfg.trend_path, 'pca_coords.parquet.zstd')
    if os.path.exists(target_path):
        target_df = pl.read_parquet(target_path)
        coords = target_df['coord_0d'].to_numpy()
        components = np.load(os.path.join(cfg.trend_path, 'pca_components.npy'))
        stance_cols = [col for col in target_df.columns if col.startswith('trend_mean_')]
        target_df = target_df.select(['createtime', 'filter_value', 'coord_0d'])
    else:
        all_trend_path = os.path.join(cfg.trend_path, 'loaded_trends.parquet.zstd')
        if os.path.exists(all_trend_path):
            df = pl.read_parquet(all_trend_path)
        else:
            df = load_df(cfg.trend_path, 'SeedName')
            df.write_parquet(all_trend_path, compression='zstd')
    
    df = df.filter(pl.col('filter_value').is_in(russia_state_media_seed_names + tenet_influencer_seed_names))
   
    targets = ['global humanitarian immigration', 'ukrainian military response to russia']
    fig, axes = plt.subplots(nrows=len(targets), figsize=(12, 10))
    for i, target in enumerate(targets):
        ax = axes[i]
        target_df = df.filter(pl.col('target') == target)
        seed_names = target_df['filter_value'].unique().to_list()
        for seed_name in seed_names:
            seed_name_df = target_df.filter(pl.col('filter_value') == seed_name).sort('createtime')
            seed_name_datetime = seed_name_df['createtime'].to_numpy()
            seed_name_coords = seed_name_df['trend_mean'].to_numpy()
            ax.plot(seed_name_datetime, seed_name_coords, label=seed_name)
        ax.set_title(f'Trend Movement for Target: {target}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Trend Mean')
        ax.legend()

    fig.tight_layout()
    fig.savefig('./figs/individual_target_trend_movement.png', dpi=300)

if __name__ == '__main__':
    main()