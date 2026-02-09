import os

import polars as pl
from tqdm import tqdm

def main():
    dir_path = '../sitrep/data/digital_trace/raw_platforms'

    data_files = os.listdir(dir_path)
    file_df = pl.DataFrame({'file': data_files})
    file_df = file_df.filter(pl.col('file').str.starts_with('twitter'))
    file_df = file_df.filter(pl.col('file').str.ends_with('parquet.zstd'))

    # parse out date from file name
    file_df = file_df.with_columns(pl.col('file').str.split('_').list.get(1).str.split('.').list.get(0).str.split('-').alias('date_numbers'))\
        .with_columns(pl.col('date_numbers').list.get(0).cast(pl.UInt16).alias('year'))\
        .with_columns(pl.col('date_numbers').list.get(1).cast(pl.UInt8).alias('month'))\
        .with_columns(pl.col('date_numbers').list.get(2).cast(pl.UInt8).alias('day'))

    file_df = file_df.filter(pl.col('year') == 2024)

    df = pl.DataFrame()
    for file_name in tqdm(file_df['file']):
        batch_df = pl.read_parquet(f'{dir_path}/{file_name}', columns=['lang', 'seed'])
        df = pl.concat([df, batch_df], how='diagonal_relaxed')

    df = df.filter(pl.col('seed').struct.field('MainType').is_in(['influencer', 'politician']))
    lang_df = df['lang'].value_counts().sort('count').with_columns(pl.col('count') / df.shape[0])

if __name__ == '__main__':
    main()