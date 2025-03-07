import os

import polars as pl

import stancemining
from tqdm import tqdm

def main():
    # load base targets from twitter, instagram and tiktok
    tiktok_target_df = pl.read_parquet('./data/tiktok/stance_targets/targets.parquet.zstd')
    tiktok_target_df = tiktok_target_df.with_columns(pl.col('text').alias('Document'))
    twitter_target_dir = './data/twitter/stance_targets'
    twitter_target_df = pl.DataFrame()
    for filename in os.listdir(twitter_target_dir):
        if filename.startswith('targets') and filename.endswith('.parquet.zstd'):
            twitter_target_file_df = pl.read_parquet(f'{twitter_target_dir}/{filename}')
            twitter_target_df = pl.concat([twitter_target_df, twitter_target_file_df], how='diagonal_relaxed')
    twitter_target_df = twitter_target_df.with_columns(pl.col('rawContent').alias('Document'))
    instagram_target_dir = './data/instagram/stance_targets'
    instagram_target_df = pl.DataFrame()
    for filename in os.listdir(instagram_target_dir):
        if filename.startswith('targets') and filename.endswith('.parquet.zstd'):
            instagram_target_file_df = pl.read_parquet(f'{instagram_target_dir}/{filename}')
            instagram_target_df = pl.concat([instagram_target_df, instagram_target_file_df], how='diagonal_relaxed')
    instagram_target_df = instagram_target_df.with_columns(pl.col('raw_caption').alias('Document'))

    dir_path = '../sitrep/data/digital_trace/raw_platforms'

    data_files = os.listdir(dir_path)
    file_df = pl.DataFrame({'file': data_files})
    file_df = file_df.filter(pl.col('file').str.ends_with('parquet.zstd'))

    # parse out date from file name
    file_df = file_df.with_columns(pl.col('file').str.split('_').list.get(1).str.split('.').list.get(0).str.split('-').alias('date_numbers'))\
        .with_columns(pl.col('date_numbers').list.get(0).cast(pl.UInt16).alias('year'))\
        .with_columns(pl.col('date_numbers').list.get(1).cast(pl.UInt8).alias('month'))\
        .with_columns(pl.col('date_numbers').list.get(2).cast(pl.UInt8).alias('day'))

    # group by month
    twitter_df = pl.DataFrame()
    instagram_df = pl.DataFrame()
    tiktok_df = pl.DataFrame()
    for file_d in tqdm(file_df.to_dicts()):
        file_name = file_d['file']
        platform = file_name.split('_')[0]
        if platform not in ['twitter', 'instagram', 'tiktok']:
            continue
        batch_df = pl.read_parquet(f'{dir_path}/{file_name}')
        if platform == 'twitter':
            batch_df = batch_df.select(['id', 'seed_id', 'date'])
            twitter_df = pl.concat([twitter_df, batch_df], how='diagonal_relaxed')
        elif platform == 'instagram':
            batch_df = batch_df.select(['id', 'seed_id', 'taken_at'])
            instagram_df = pl.concat([instagram_df, batch_df], how='diagonal_relaxed')
        elif platform == 'tiktok':
            batch_df = batch_df.select([pl.col('video_id').cast(pl.UInt64), 'seed_id', 'createtime'])
            tiktok_df = pl.concat([tiktok_df, batch_df], how='diagonal_relaxed')

    if 'createtime' in tiktok_target_df.columns:
        tiktok_target_df = tiktok_target_df.drop('createtime')
    if 'date' in twitter_target_df.columns:
        twitter_target_df = twitter_target_df.drop('date')
    if 'taken_at' in instagram_target_df.columns:
        instagram_target_df = instagram_target_df.drop('taken_at')

    tiktok_target_df = tiktok_target_df.join(tiktok_df.select(['video_id', 'seed_id', 'createtime']), on='video_id', how='left')
    twitter_target_df = twitter_target_df.join(twitter_df.select(['id', 'seed_id', 'date']), on='id', how='left')
    instagram_target_df = instagram_target_df.join(instagram_df.select(['id', 'seed_id', 'taken_at']), on='id', how='left')

    tiktok_target_df.write_parquet('./data/tiktok/stance_targets/targets.parquet.zstd', compression='zstd')
    twitter_target_df.write_parquet('./data/twitter/stance_targets/targets.parquet.zstd', compression='zstd')
    instagram_target_df.write_parquet('./data/instagram/stance_targets/targets.parquet.zstd', compression='zstd')


if __name__ == '__main__':
    main()