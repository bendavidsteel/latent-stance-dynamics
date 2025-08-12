

import json
import os

import polars as pl
from tqdm import tqdm

def load_df():
    df = None
    trends_dir_path = './data/stance_targets/target_trends'
    pbar = tqdm()
    for target_file_name in os.listdir(trends_dir_path):
        target_path = os.path.join(trends_dir_path, target_file_name)
        if target_path.endswith('.parquet.zstd'):
            file_df = pl.read_parquet(target_path).filter(pl.col('filter_type') == 'PlatformHandleID')
            if 'trend_mean' not in file_df.columns:
                continue
            pbar.update(1)
            if pbar.n > 2000:
                break
            if df is not None:
                df = pl.concat([df, file_df])
            else:
                df = file_df
    pbar.close()
    return df




def main():
    print("Loading data...")
    df = load_df()

    dup_target_names = df.group_by(['createtime', 'target', 'filter_value']).agg(pl.col('trend_mean')).filter(pl.col('trend_mean').list.len() > 1).unique('target')['target'].to_list()

    filenames = os.listdir('./data/stance_targets/target_trends')
    for dup_target_name in dup_target_names:
        old_file_names = [f for f in filenames if dup_target_name in f]
        new_file_names = [f for f in filenames if dup_target_name.replace(' ', '_').replace('-', '_').replace("'", '').replace('"', '') in f]
        if len(old_file_names) == 2 and len(new_file_names) == 2 and set(old_file_names) != set(new_file_names):
            for old_file_name in old_file_names:
                os.remove(os.path.join('./data/stance_targets/target_trends', old_file_name))
    

if __name__ == '__main__':
    main()