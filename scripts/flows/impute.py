
import os

import hydra
import polars as pl

from pca_density import load_df, pivot_and_impute
from pca_clusters import load_text_df

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    trend_path = cfg.trend_path
    target_path = os.path.join(trend_path, 'pivoted_and_imputed.parquet.zstd')
    all_trend_path = os.path.join(trend_path, 'loaded_trends.parquet.zstd')
    if os.path.exists(all_trend_path):
        df = pl.read_parquet(all_trend_path, columns=['createtime', 'volume', 'trend_mean', 'target', 'filter_type', 'filter_value'])
    else:
        df = load_df(trend_path, cfg.filter_column, min_filter_count=cfg.min_filter_count, group_by_every=cfg.group_by_every)
        df = df.select(['createtime', 'volume', 'trend_mean', 'target', 'filter_type', 'filter_value'])
        df.write_parquet(all_trend_path, compression='zstd')

    df = df.filter(pl.col('filter_value') != '')\
        .filter(pl.col('filter_type') == cfg.filter_column)\
        .drop('filter_type')

    top_target_df = df.group_by('target').agg(pl.col('volume').sum()).filter(pl.col('volume') >= cfg.min_target_volume)
    df = df.join(top_target_df.select('target'), on='target', how='inner').drop('volume')

    text_df = load_text_df(cfg, columns=['seed'])
    seed_df = text_df.select([
        pl.col('seed').struct.field('SeedName'),
        pl.col('seed').struct.field('PlatformHandleID'),
        pl.col('seed').struct.field('MainType'),
        pl.col('seed').struct.field('SubType')
    ]).unique(cfg.filter_column)
    seed_df = seed_df.filter(pl.col('MainType').is_in(['politician', 'influencer']) | ((pl.col('MainType') == 'foreign') & (~pl.col('SubType').is_in(['media', 'state']))))\
        .select(cfg.filter_column)
    df = df.join(seed_df, left_on='filter_value', right_on=cfg.filter_column, how='inner')
    
    target_df, stance_cols = pivot_and_impute(df, impute_fancy=True)
    
    target_df.write_parquet(target_path, compression='zstd')

if __name__ == '__main__':
    main()