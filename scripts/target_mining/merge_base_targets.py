import datetime
import os
import re

import polars as pl
from tqdm import tqdm

def main():
    
    targets_path = './data/stance_targets/base_claims'
    filenames = os.listdir(targets_path)
    df = pl.DataFrame({'filename': filenames})\
        .with_columns(pl.col('filename').str.split('.').list.get(0).str.split('_').list.slice(1, 3).alias('date_numbers'))\
        .with_columns([
            pl.col('date_numbers').list.get(0).cast(pl.Int32).alias('year'),
            pl.col('date_numbers').list.get(1).cast(pl.Int32).alias('month'),
            pl.col('date_numbers').list.get(2).cast(pl.Int32).alias('day')
        ])
    
    for (year, month), group_df in tqdm(df.partition_by(['year', 'month'], as_dict=True).items()):
        month_df = pl.DataFrame()
        for filename in group_df['filename'].to_list():
            day_df = pl.read_parquet(os.path.join(targets_path, filename))
            month_df = pl.concat([month_df, day_df], how='diagonal_relaxed')
        month_df.write_parquet(os.path.join(targets_path, f'targets_{year}_{month:02d}.parquet.zstd'), compression='zstd')

if __name__ == '__main__':
    main()
    