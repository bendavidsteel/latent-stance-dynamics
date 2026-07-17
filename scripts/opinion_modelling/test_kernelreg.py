import datetime
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.nonparametric.kernel_regression import KernelReg

from stancemining.estimate import _get_time_series_data

def boot_pred(stance, timestamps, test_x, bandwidth, n_samples):
    # Resample with replacement
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    boot_endog = stance[indices]
    boot_exog = timestamps[indices]

    # Fit kernel regression on bootstrap sample
    kr = KernelReg(boot_endog, boot_exog, var_type='c', bw=[bandwidth])
    boot_pred, _ = kr.fit(test_x)
    return boot_pred

def main():
    base_stance_path = './data/stance_targets/doc_stance'

    # Collect all file paths first, then read them in a single operation
    file_paths = [
        os.path.join(base_stance_path, file) 
        for file in os.listdir(base_stance_path)
        if re.search(r'\d{4}_\d{1,2}_doc_targets_with_stance.parquet.zstd', file)
    ]
    
    if not file_paths:
        raise ValueError("No stance data files found in the data directory")
    
    dfs = [
        pl.read_parquet(
            file_path,
            columns=['id', 'createtime', 'platform', 'Document', 'ParentDocument', 'Targets', 'Stances', 'seed']
        ) for file_path in file_paths
    ]
    
    # Concatenate the scanned DataFrames
    df = pl.concat(dfs, how='diagonal_relaxed')

    df = df.unique(['id', 'platform'])

    df = df.explode(['Targets', 'Stances'])\
        .rename({'Targets': 'Target', 'Stances': 'Stance'})
    df = df.drop_nulls('Target')
    
    df = df.with_columns(pl.col('seed').struct.unnest())

    # Get ordered list of all targets in a single pipeline
    target_count_df = df.group_by('Target').agg(pl.count().alias('count')).sort('count', descending=True)

    target = 'vaccine mandate'

    target_df = df.filter(pl.col('Target') == target)
    time_column = 'createtime'
    time_scale='1mo'

    seed_df = target_df.group_by(pl.col('seed').struct.field('SeedName')).agg([
        pl.count().alias('count'),
        pl.col('createtime').min().alias('start_date'),
        pl.col('createtime').max().alias('end_date')
    ]).filter((pl.col('count') > 10) & ((pl.col('end_date') - pl.col('start_date')).dt.total_days() > 1))

    num_exs = 5
    fig, ax = plt.subplots(nrows=num_exs, ncols=1, figsize=(10, 5 * num_exs))
    for i, filter_value in enumerate(seed_df['SeedName'].to_list()[:num_exs]):
        filtered_df = target_df.filter(pl.col('seed').struct.field('SeedName') == filter_value).sort('createtime')

        start_date = filtered_df[time_column].min().date()
        end_date = filtered_df[time_column].max().date()

        if end_date - start_date < datetime.timedelta(days=1):
            continue

        timestamps, stance, classifier_ids, test_x, trend_df = _get_time_series_data(filtered_df, time_column, time_scale)

        start_time = datetime.datetime.now()
        if True:
            n_bootstrap = 100
            bandwidth = 10.0
            n_samples = len(stance)
            all_preds = np.zeros((n_bootstrap, len(test_x)))
            
            for j in range(n_bootstrap):
                all_preds[j] = boot_pred(stance, timestamps, test_x, bandwidth, n_samples)
            
            # Compute statistics
            pred_mean = np.mean(all_preds, axis=0)
            pred_lower = np.percentile(all_preds, 5, axis=0)
            pred_upper = np.percentile(all_preds, 95, axis=0)

            pred_mean = np.clip(pred_mean, -1, 1)
            pred_lower = np.clip(pred_lower, -1, 1)
            pred_upper = np.clip(pred_upper, -1, 1)

        end_time = datetime.datetime.now()
        print(f"Kernel regression with bootstrapping took {end_time - start_time}")

        ax[i].scatter(timestamps, stance, label='Posts', alpha=0.3)
        ax[i].plot(test_x, pred_mean, label=f'LOWESS', color='orange')
        ax[i].fill_between(test_x, pred_lower, pred_upper, color='orange', alpha=0.2)

    fig.savefig('./figs/lowess_trend.png')

if __name__ == "__main__":
    main()
