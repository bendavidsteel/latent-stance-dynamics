import datetime
import os
import re

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from deduplicate_targets import process_data, remove_low_count_targets
from mining.estimate import setup_ordinal_gp_model, train_ordinal_likelihood_gp, get_model_prediction


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_timestamps(df, start_date):
    return df.select(((pl.col('createtime') - start_date).dt.total_hours() / (24 * 30)).alias('timestamps'))['timestamps'].to_numpy()


def plot_trends_for_filtered_df(filtered_df: pl.DataFrame, target_name, filter_type, filter_value, start_date, end_date):
    """Calculate trends for a filtered DataFrame with optimized operations"""
    # First sort by createtime - ensures consistent results

    target_slug = target_name.replace(' ', '_')
    trend_path = f"./data/stance_targets/target_trends/{target_slug}/{filter_type}_{filter_value}.parquet.zstd"
    if not os.path.exists(trend_path):
        return
    trend_df = pl.read_parquet(trend_path)

    fig, ax = plt.subplots()
    trend_createtime = trend_df['createtime'].to_numpy()
    trend_mean = trend_df['trend_mean'].to_numpy()
    trend_lower = trend_df['trend_lower'].to_numpy()
    trend_upper = trend_df['trend_upper'].to_numpy()
    ax.plot(trend_createtime, trend_mean, label='GP')
    ax.fill_between(trend_createtime, trend_lower, trend_upper, alpha=0.1)

    window_size = len(filtered_df) // 10
    
    sorted_df = filtered_df.sort('createtime')
    
    week_df = sorted_df.group_by_dynamic('createtime', every='1w').agg(pl.col('Stance').mean())
    week_time = week_df['createtime'].to_numpy()
    week_avg = week_df['Stance'].to_numpy()

    ax.plot(week_time, week_avg, label='MA', color='orange')

    ax.hlines(sorted_df['Stance'].mean(), xmin=sorted_df['createtime'].min(), xmax=sorted_df['createtime'].max(), label='Mean')

    post_createtime = sorted_df['createtime'].to_numpy()
    post_stance = sorted_df['Stance'].to_numpy()
    ax.scatter(post_createtime, post_stance, marker='x')

    ax.legend()

    ax.set_ylim([-1.2, 1.2])
    trend_plot_path = trend_path.replace('/data/', '/figs/').replace('.parquet.zstd', '.png')
    os.makedirs(os.path.dirname(trend_plot_path), exist_ok=True)
    fig.savefig(trend_plot_path)
    plt.close()

def precompute_trends_for_target(df, target_name, start_date, end_date):
    """Precompute trend data for a specific target with optimized vectorized operations"""
    # Get target data in one operation
    target_df = df.filter(pl.col('Target') == target_name)
    
    print(f"Processing target {target_name}: {target_df.shape[0]} points")
    
    # Process document text for tooltips in a single vectorized operation
    target_df = target_df.with_columns(
        pl.when(pl.col('Document').str.len_chars() > 300)
        .then(pl.col('Document').str.slice(0, 297) + pl.lit('...'))
        .otherwise(pl.col('Document'))
        .str.replace("<", "&lt;")
        .str.replace(">", "&gt;")
        .alias('document_text')
    )
    
    # Collect all trends in a list
    trend_data_list = []

    # Calculate all trends
    
    # Define all filter types to process
    filters = [
        {'column': 'PlatformHandleID', 'type_name': 'platform_handle_id'},
        {'column': 'platform', 'type_name': 'platform'},
        {'column': 'Party', 'type_name': 'party'},
        {'column': 'MainType', 'type_name': 'main_type'}
    ]
    min_filter_count = 5
    
    # For each filter type, calculate trends for each unique value
    for filter_def in filters:
        column = filter_def['column']
        filter_type = filter_def['type_name']
        
        # Get all unique values for this filter
        unique_values = target_df[column].unique()
        
        # Apply filtering and trend calculation for each value
        for filter_value in unique_values:
            filtered_df = target_df.filter(pl.col(column) == filter_value)
            if filtered_df.shape[0] < min_filter_count:
                continue
            plot_trends_for_filtered_df(filtered_df, target_name, filter_type, filter_value, start_date, end_date)
            print(f"  Processed {filter_type} {filter_value}: {len(filtered_df)} points")
    
    # First, the overall trend
    plot_trends_for_filtered_df(target_df, target_name, 'all', 'all', start_date, end_date)

def precompute_trends_for_all_targets(df, target_count_df):
    """Precompute trend data for all targets with optimized batch processing"""
    os.makedirs('./data/precomputed', exist_ok=True)

    start_date = datetime.date(2022, 1, 1)
    end_date = datetime.date(2025, 5, 1)
    
    # Initialize empty lists to collect all data
    all_trends_data = []
    all_raw_data = []
    
    for target in target_count_df.to_dicts():
        target_name = target['Target']
        print(f"Processing primary target: {target_name}")
            
        # Process the target with grouping
        precompute_trends_for_target(df, target_name, start_date, end_date)
        
    
def calculate_stance_statistics(df, valid_targets: pl.DataFrame):
    """Calculate average stance statistics for each target using efficient Polars operations"""
    target_stats = []
    
    for target_info in valid_targets.to_dicts():
        target_name = target_info['Target']
        
        # Filter data and calculate all statistics in one pipeline
        target_df = df.filter(pl.col('Target') == target_name)
        
        if len(target_df) > 0:
            # Calculate most metrics in a single aggregation
            stats = (target_df
                .select([
                    pl.col('Stance').mean().alias('avg_stance'),
                    pl.col('Stance').std().alias('stance_std'),
                    pl.col('Stance').abs().mean().alias('stance_abs'),
                    pl.when(pl.col('Stance') > 0.1).then(1).otherwise(0).sum().alias('n_positive'),
                    pl.when(pl.col('Stance') < -0.1).then(1).otherwise(0).sum().alias('n_negative'),
                    pl.count().alias('total_count')
                ])
                .with_columns(
                    (pl.col('total_count') - pl.col('n_positive') - pl.col('n_negative')).alias('n_neutral')
                )
                .to_dicts()[0]
            )
            
            # Get top values for categorical fields in one operation
            top_values = {}
            for field, name in [('platform', 'top_platform'), ('Party', 'top_party'), ('MainType', 'top_main_type')]:
                # Get the most common value
                top = target_df.group_by(field).count().sort('count', descending=True).head(1)
                if len(top) > 0:
                    top_values[name] = top[0, field]
                else:
                    top_values[name] = "Unknown"
            
            # Combine all metrics
            target_stat = {
                'Target': target_name,
                'count': target_info.get('count', 0),
                'avg_stance': float(stats['avg_stance']),
                'stance_std': float(stats['stance_std']),
                'stance_abs': float(stats['stance_abs']),
                'n_positive': int(stats['n_positive']),
                'n_negative': int(stats['n_negative']),
                'n_neutral': int(stats['n_neutral']),
                'top_platform': top_values['top_platform'],
                'top_party': top_values['top_party'],
                'top_main_type': top_values['top_main_type'],
            }
            
            target_stats.append(target_stat)
    
    # Create dataframe and save in a single operation
    target_stats_df = pl.DataFrame(target_stats)
    target_stats_df.write_parquet('./data/precomputed/target_statistics.parquet.zstd', compression="zstd")
    print(f"Saved stance statistics for {len(target_stats)} targets")
    
    return target_stats_df

def main():
    print("Starting precomputation process...")
    
    # Process the data
    df, target_count_df, unique_platforms, unique_parties, unique_main_types = process_data()
    print(f"Processed data for {len(target_count_df)} targets")

    # remove targets with low counts
    df, target_count_df = remove_low_count_targets(df, target_count_df, 50)

    # target_count_df = target_count_df.filter(pl.col('Target').str.contains_any(['israel', 'gaza', 'ukraine', 'russia', 'china', 'carbon tax', 'immigration']))

    # Precompute trend data for primary targets only (grouped)
    precompute_trends_for_all_targets(df, target_count_df)
    
    print("Precomputation complete!")

if __name__ == "__main__":
    dotenv.load_dotenv()
    main()