import os
import re

import matplotlib.pyplot as plt
import polars as pl

from mining.estimate import spline_means

def main():
    dir_path = './data/stance_targets/'
    df = pl.DataFrame()
    for filename in os.listdir(dir_path):
        if re.search(r'\d{4}_\d{1,2}_doc_targets_with_stance.parquet.zstd', filename):
            file_df = pl.read_parquet(
                os.path.join(dir_path, filename),
                # Specify only the columns we need to improve loading time
                columns=['id', 'createtime', 'Document', 'Targets', 'Polarities', 'seed_id']
            )
            df = pl.concat([df, file_df], how='diagonal_relaxed')

    stance_df = df.filter(pl.col('Targets').list.len() == pl.col('Polarities').list.len()).explode(['Targets', 'Polarities']).rename({'Targets': 'Target', 'Polarities': 'Stance'})
    stance_df = stance_df.filter(pl.col('Target') == 'ukraine').sort('createtime')
    # Calculate trend with fixed random state for deterministic results
    trend_timestamps = stance_df['createtime'].dt.timestamp().to_numpy()
    trend_stances = stance_df['Stance'].to_numpy()
    
    fig, ax = plt.subplots()
    ax.scatter(trend_timestamps, trend_stances, label='Posts')

    window_sizes = [100, 1000, 3000]
    
    for window_size in window_sizes:
        ax.plot(trend_timestamps, stance_df['Stance'].rolling_mean(window_size).to_numpy(), label=f'Rolling mean {window_size}')
    fig.legend()
    fig.savefig('./figs/trend.png')

    trend_timestamps = trend_timestamps.reshape(1, -1)
    trend_stances = trend_stances.reshape(1, -1, 1)
    # trend_timestamps, means, confidence_intervals = spline_means(trend_timestamps, trend_stances, n_bootstraps=100)
    # pass

if __name__ == "__main__":
    main()