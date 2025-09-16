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

from deduplicate_targets import remove_bad_targets

def get_raw_file(filename, platform):
    raw_path = '~/repos/sitrep/data/digital_trace/raw_platforms'
    year, month, day = filename.split('.')[0].split('_')[1:]
    date_str = datetime.date(int(year), int(month), int(day)).strftime('%Y-%m-%d')
    raw_filename = f"{platform}_{date_str}.parquet.zstd"
    raw_day_df = pl.read_parquet(os.path.join(raw_path, raw_filename))
    return raw_day_df

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

    # remove cases where there are more than 5 targets
    df = df.with_columns(pl.col('Targets').list.slice(0, 5))
    target_df = df.select(['index', 'Targets']).explode('Targets').rename({'Targets': 'Target'})
    target_df = remove_bad_targets(target_df)
    target_df = target_df.group_by('index').agg(pl.col('Target')).rename({'Target': 'Targets'})
    df = df.drop('Targets').join(target_df, on='index', how='left').with_columns(pl.col('Targets').fill_null([]))
    df = df.drop('index')

    return df

def filter_to_common_targets(df: pl.DataFrame, num_targets: int):
    df = df.with_row_index()

    target_df = df.select(['index', 'Targets']).explode('Targets').rename({'Targets': 'Target'})
    unique_target_df = target_df.group_by('Target').len().sort('len', descending=True).head(num_targets).select('Target')
    target_df = target_df.join(unique_target_df, on='Target', how='inner')
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

    start_date_str = '2022-01-01'
    period = f'{start_date_str}-onwards'
    document_df = pl.read_parquet(f"./data/stance_targets/{period}_doc_targets_deduplicated.parquet.zstd")

    # remove bad targets
    print(f"Before filtering bad targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets present.")
    document_df = remove_doc_bad_targets(document_df)
    print(f"After filtering bad targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets remain.")

    document_df = filter_to_common_targets(document_df, num_targets=3000000)
    print(f"After filtering to common targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets remain.")

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

    if False:
        # reducing number of targets
        logger.info(f"Before de-duplicating similar targets, {document_df.select('Targets').explode('Targets').shape[0]} targets present.")
        document_df = deduplicate_all_similar_targets(document_df, model.embedding_model, batch_size=100000, max_distance=0.15)
        logger.info(f"After de-duplicating similar targets, {document_df.select('Targets').explode('Targets').shape[0]} targets remain.")

    toponymy_kwargs = {
        'clusterer': {
            'base_min_cluster_size': 20
        }
    }

    logger.info('Fitting model...')

    doc_target_df = model.fit_transform(
        document_df, 
        get_stance=False, 
        embedding_cache=unique_target_df, 
        # topic_model_kwargs=toponymy_kwargs,
        text_column='Document',
        parent_text_column='ParentDocument',
        max_layers=1
    )
    target_info_df = model.get_target_info()

    logger.info(f'Finished fitting model with {len(target_info_df)} targets.')

    doc_target_df.write_parquet(f'./data/stance_targets/{period}_doc_targets.parquet.zstd', compression='zstd')

    logger.info(f'Saving target info to {period}_target_info.parquet.zstd')
    target_info_df.write_parquet(f'./data/stance_targets/{period}_target_info.parquet.zstd', compression='zstd')
    logger.info("Successfully saved target info.")

    topic_model = model.get_topic_model()
    topic_model.save(f"./data/stance_targets/{period}_topic_model.pickle", serialization="pickle")

if __name__ == '__main__':
    main()
