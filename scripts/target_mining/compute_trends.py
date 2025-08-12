import datetime
import json
import os
import re

import dotenv
import numpy as np
import polars as pl
import requests
from torch import multiprocessing
from tqdm import tqdm
import wandb
from stancemining.estimate import infer_stance_trends_for_target, _get_classifier_profiles

def get_session(base_url, username, password):
    res = requests.post(f"{base_url}/meologin", params={"username": username, "password": password}, verify=True)
    token = res.json()["access_token"]
    headers = {
        "Authorization": f"Bearer {token}"
    }
    return headers

def get_seedlist(base_url, headers):
    target_url = f"{base_url}/phh/seedlist/"
    res = requests.get(target_url, params={"query": '*'}, headers=headers)
    if res.status_code == 200:
        return res.json()
    else:
        return None

def process_data():
    """Load and process the main data from files using efficient Polars operations"""
    print("Loading and processing data files...")
    # Load data with fewer columns
    dir_path = './data/stance_targets/doc_stance'
    
    # Collect all file paths first, then read them in a single operation
    file_paths = [
        os.path.join(dir_path, file) 
        for file in os.listdir(dir_path)
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
    
    # Filter for consistent list lengths, then explode
    df = df.explode(['Targets', 'Stances'])\
        .rename({'Targets': 'Target', 'Stances': 'Stance'})
    df = df.drop_nulls('Target')
    
    df = df.with_columns(pl.col('seed').struct.unnest())

    # Get ordered list of all targets in a single pipeline
    target_count_df = df.group_by('Target').agg(pl.count().alias('count')).sort('count', descending=True)
    
    print(f"Processed {len(df)} records for {len(target_count_df)} targets")
    return df, target_count_df


def remove_low_count_targets(df, target_count_df, min_count):
    removed_target_df = target_count_df.filter(pl.col('count') < min_count)
    target_count_df = target_count_df.filter(pl.col('count') >= min_count)
    df = df.join(removed_target_df, on='Target', how='anti')
    return df, target_count_df




def precompute_trends_for_all_targets(df, target_count_df):
    """Precompute trend data for all targets with optimized batch processing"""

    classifier_profiles = _get_classifier_profiles('bendavidsteel/SmolLM2-135M-Instruct-stance-detection')
    
    filter_columns = ['platform', 'PlatformHandleID']

    dir_path = './data/stance_targets/target_trends'

    for target in target_count_df.to_dicts():
        target_name = target['Target']
        target_slug = target_name.lower().replace(' ', '_').replace('-', '_').replace("'", '').replace('"', '')
        trend_path = f'{dir_path}/{target_slug}_trends.parquet.zstd'
        gp_path = f'{dir_path}/{target_slug}_gp.parquet.zstd'
        # if os.path.exists(trend_path) and os.path.exists(gp_path):
        #     print(f"Skipping {target_name}, trends already computed.")
        #     continue

        print(f"Processing primary target: {target_name}")
            
        # Process the target with grouping
        target_trend_df, gp_params = infer_stance_trends_for_target(
            df,
            target_name,
            filter_columns=filter_columns,
            time_column='createtime',
            classifier_profiles=classifier_profiles,
            min_filter_count=20,
            verbose=True
        )
        if target_trend_df is None or gp_params is None:
            print(f"Skipping {target_name}, no trends computed.")
            continue
        target_trend_df.write_parquet(trend_path, compression='zstd')
        gp_df = pl.from_dicts(gp_params, schema_overrides={'loss': pl.Float64})
        gp_df.write_parquet(gp_path, compression='zstd')


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
    df, target_count_df = process_data()
    print(f"Processed data for {len(target_count_df)} targets")

    # remove targets with low counts
    df, target_count_df = remove_low_count_targets(df, target_count_df, 50)

    # target_count_df = target_count_df.filter(pl.col('Target').str.contains_any([
    #     'israel', 
    #     'gaza', 
    #     'ukraine', 
    #     'russia', 
    #     'china', 
    #     'carbon tax', 
    #     'immigration',
    #     'trudeau',
    #     'carney',
    #     'poilievre',
    #     'trump',
    #     'liberal',
    #     'conservative',
    #     'ndp',
    #     'taiwan'
    # ]))

    # Precompute trend data for primary targets only (grouped)
    precompute_trends_for_all_targets(df, target_count_df)
    
    print("Precomputation complete!")

if __name__ == "__main__":
    dotenv.load_dotenv()
    main()