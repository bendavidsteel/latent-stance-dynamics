import json
import os

import matplotlib.pyplot as plt
import polars as pl


def main():
    df = None
    trends_dir_path = './data/stance_targets/target_trends'
    target_name = 'trudeau'
    target_dir_path = os.path.join(trends_dir_path, target_name)
    general_df = pl.read_parquet(os.path.join(target_dir_path, 'all_all.parquet.zstd'))

    fig, ax = plt.subplots(figsize=(10, 4))
    for trend_file_name in os.listdir(target_dir_path):
        if trend_file_name.endswith('.parquet.zstd') and trend_file_name.startswith('platform_handle_id'):
            trend_file_path = os.path.join(target_dir_path, trend_file_name)
            file_df = pl.read_parquet(trend_file_path)
            ax.plot(file_df['createtime'], file_df['trend_mean'], color='blue', alpha=0.1)

    ax.plot(general_df['createtime'], general_df['trend_mean'], label='All', color='black', alpha=1, linewidth=2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Trend Mean')
    fig.savefig('./figs/trend_composition.png', dpi=300)

if __name__ == '__main__':
    main()
