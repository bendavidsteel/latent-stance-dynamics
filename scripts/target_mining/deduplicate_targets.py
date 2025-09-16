import datetime
import logging
import multiprocessing as mp
import os
import re

import hydra
import numpy as np
import polars as pl
from tqdm import tqdm
import torch
import transformers

from stancemining.main import StanceMining
from stancemining.utils import deduplicate_all_similar_targets

def get_raw_file(filename, platform):
    raw_path = '~/repos/sitrep/data/digital_trace/raw_platforms'
    year, month, day = filename.split('.')[0].split('_')[1:]
    date_str = datetime.date(int(year), int(month), int(day)).strftime('%Y-%m-%d')
    raw_filename = f"{platform}_{date_str}.parquet.zstd"
    raw_day_df = pl.read_parquet(os.path.join(raw_path, raw_filename))
    return raw_day_df

def remove_bad_targets(target_df: pl.DataFrame):
    target_df = target_df.filter(~pl.col('Target').str.contains_any(['the user', 'url', 'the text', 'the speaker']))
    target_df = target_df.filter(~pl.col("Target").str.contains("^the assistant"))
    target_df = target_df.filter(pl.col('Target').str.len_chars() < 70)
    target_df = target_df.filter(pl.col('Target').str.len_chars() > 20)
    target_df = target_df.filter(~pl.col("Target").str.contains("^\w+ was mentioned.$"))
    nouns = ['place', 'person', 'link', 'claim', 'city', 'region', 'nation', 'user', 'holiday', 'greeting', 'holiday greeting']
    for noun in nouns:
        target_df = target_df.filter(~pl.col("Target").str.contains(f"^\w+ is a {noun}.$"))
        target_df = target_df.filter(~pl.col("Target").str.contains(f"^\w+ \w+ is a {noun}.$"))

    if 'index' not in target_df.columns:
        target_df = target_df.with_row_index()

    superset_target_df = target_df.join(
            target_df.select([
                'index',
                pl.col('Target').str.strip_chars('.').alias('shorter_target')
            ]),
            on='index',
            how='left'
        )\
        .filter(pl.col('Target').str.len_chars() > pl.col('shorter_target').str.len_chars() + 1)\
        .filter(pl.col('Target').str.contains(pl.col('shorter_target'), literal=True))\
        .select(['index', 'Target'])\
        .unique()
    target_df = target_df.join(superset_target_df, on=['index', 'Target'], how='anti')

    return target_df

def deduplicate_subset_targets(targets):
    targets = sorted(set(targets), key=len)  # Sort by length, shortest first
    result = []
    
    for target in targets:
        # Only add if no existing (shorter) target is a substring of this one
        if not any(existing.strip('.') in target for existing in result):
            result.append(target)
    
    return result

def remove_doc_bad_targets(df: pl.DataFrame):
    df = df.with_row_index()

    # remove cases where there are more than 3 targets
    df = df.with_columns(pl.col('Targets').list.slice(0, 3))
    target_df = df.select(['index', 'Targets']).explode('Targets').rename({'Targets': 'Target'})
    target_df = remove_bad_targets(target_df)
    target_df = target_df.group_by('index').agg(pl.col('Target')).rename({'Target': 'Targets'})
    df = df.drop('Targets').join(target_df, on='index', how='left').with_columns(pl.col('Targets').fill_null([]))
    df = df.drop('index')

    return df

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    logger = logging.getLogger('find_targets')

    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    mp.set_start_method('spawn')
    pl.set_random_seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    target_dir = config.base_target_path
    document_df = pl.DataFrame()
    for filename in tqdm(os.listdir(target_dir), desc='Loading files...'):
        if re.match('targets_\d{4}_\d{1,2}.parquet.zstd', filename):
            target_file_df = pl.read_parquet(f'{target_dir}/{filename}')
            document_df = pl.concat([document_df, target_file_df], how='diagonal_relaxed')
    
    logger.info(f'Loaded {document_df.shape[0]} documents from {len(os.listdir(target_dir))} files.')

    document_df = document_df.unique(['id', 'platform'])

    start_date_str = '2022-01-01'
    period = f'{start_date_str}-onwards'
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
    document_df = document_df.filter(pl.col('createtime') >= start_date)
    logger.info(f'Filtered documents to {document_df.shape[0]} after {start_date_str}.')

    # remove bad targets
    print(f"Before filtering bad targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets present.")
    document_df = remove_doc_bad_targets(document_df)
    print(f"After filtering bad targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets remain.")

    model = StanceMining(
        model_name='Qwen/Qwen3-4B',
        model_kwargs={'gpu_memory_utilization': 0.8},
        # embedding_model='minishlab/potion-base-4M',
        stance_target_type=config.stance_target_type,
        topic_model='bertopic',
        verbose=True,
    )

    target_path = './data/stance_targets/unique_targets.parquet.zstd'
    if not os.path.exists(target_path):
        unique_target_df = document_df.select('Targets').explode('Targets').unique('Targets').drop_nulls('Targets')
        unique_target_df.write_parquet(target_path, compression='zstd')

        embeddings = model._get_embeddings(unique_target_df['Targets'].to_list())
        # save embeddings
        unique_target_df = unique_target_df.with_columns(pl.Series(name='embeddings', values=embeddings))
        unique_target_df.write_parquet(target_path, compression='zstd')
    else:
        unique_target_df = pl.read_parquet(target_path)
    unique_target_df = unique_target_df.rename({'Targets': 'text', 'embeddings': 'embedding'})

    # reducing number of targets
    logger.info(f"Before de-duplicating similar targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets present.")
    
    if document_df.select('Targets').explode('Targets').unique('Targets').shape[0] < 4000000:
        document_df = deduplicate_all_similar_targets(document_df, model.embedding_model, batch_size=100000, minhash_threshold=0.5, max_embedding_distance=0.25)
    else:
        minhash_threshold = 0.7
        max_embedding_distance = 0.15
        while (document_df.select('Targets').explode('Targets').unique('Targets').shape[0] > 3000000) and (max_embedding_distance < 0.25):
            logger.info(f"De-duplicating similar targets with minhash_threshold={minhash_threshold} and max_embedding_distance={max_embedding_distance}.")
            if document_df.select('Targets').explode('Targets').unique('Targets').shape[0] > 4000000:
                logger.info("De-duplicating in batches")
                batch_size = 2000000
                new_document_df = pl.DataFrame()
                batch_idx = list(range(0, document_df.shape[0], batch_size))
                for i in batch_idx:
                    logger.info(f"Processing batch {i // batch_size + 1}/{len(batch_idx)}")
                    batch = document_df.slice(i, batch_size)
                    batch = deduplicate_all_similar_targets(batch, model.embedding_model, batch_size=batch_size, minhash_threshold=minhash_threshold, max_embedding_distance=max_embedding_distance)
                    new_document_df = pl.concat([new_document_df, batch], how='diagonal_relaxed')
                document_df = new_document_df
            else:
                document_df = deduplicate_all_similar_targets(document_df, model.embedding_model, batch_size=100000, minhash_threshold=minhash_threshold, max_embedding_distance=max_embedding_distance)
            logger.info(f"After de-duplicating similar targets with minhash_threshold={minhash_threshold} and max_embedding_distance={max_embedding_distance}, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets remain.")
            minhash_threshold -= 0.1
            minhash_threshold = max(minhash_threshold, 0.2)
            max_embedding_distance += 0.05
            document_df = document_df.sample(fraction=1.0, shuffle=True)

    logger.info(f"After de-duplicating similar targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets remain.")

    document_df.write_parquet(f'./data/stance_targets/{period}_doc_targets_deduplicated.parquet.zstd', compression='zstd')

if __name__ == '__main__':
    main()
