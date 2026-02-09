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
from stancemining.utils import deduplicate_all_similar_targets, remove_doc_bad_targets

def get_raw_file(filename, platform):
    raw_path = '~/repos/sitrep/data/digital_trace/raw_platforms'
    year, month, day = filename.split('.')[0].split('_')[1:]
    date_str = datetime.date(int(year), int(month), int(day)).strftime('%Y-%m-%d')
    raw_filename = f"{platform}_{date_str}.parquet.zstd"
    raw_day_df = pl.read_parquet(os.path.join(raw_path, raw_filename))
    return raw_day_df

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
    deduplicate = True
    file_path = f"./data/stance_targets/{period}_{config.stance_target_type}_doc_targets_deduplicated.parquet.zstd"
    if os.path.exists(file_path):
        document_df = pl.read_parquet(file_path)
        deduplicate = False
        # document_df = document_df.head(10000)  # For testing
    else:
        target_dir = config.base_target_path
        document_df = pl.read_parquet([f'{target_dir}/{filename}' for filename in os.listdir(target_dir) if re.match('targets_\d{4}_\d{1,2}.parquet.zstd', filename)])
        
        logger.info(f'Loaded {document_df.shape[0]} documents from {len(os.listdir(target_dir))} files.')

        document_df = document_df.unique(['id', 'platform'])

        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
        document_df = document_df.filter(pl.col('createtime') >= start_date)

    # remove bad targets
    print(f"Before filtering bad targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets present.")
    document_df = remove_doc_bad_targets(document_df, config.stance_target_type)
    print(f"After filtering bad targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets remain.")

    document_df = filter_to_common_targets(document_df, num_targets=3000000)
    print(f"After filtering to common targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets remain.")

    model = StanceMining(
        model_name='Qwen/Qwen3-4B-Instruct-2507',
        model_kwargs={'gpu_memory_utilization': 0.8, 'max_model_len': 8192},
        # embedding_model='minishlab/potion-base-4M',
        stance_target_type=config.stance_target_type,
        topic_model='bertopic',
        verbose=True,
        use_embedding_cache=False
    )

    if deduplicate:
        # reducing number of targets
        logger.info(f"Before de-duplicating similar targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets present.")
        document_df = deduplicate_all_similar_targets(document_df, model.embedding_model, config.stance_target_type, batch_size=2000000)
        logger.info(f"After de-duplicating similar targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets remain.")

    toponymy_kwargs = {
        'clusterer': {
            'base_min_cluster_size': 20
        }
    }

    logger.info('Fitting model...')

    doc_target_df = model.fit_transform(
        document_df, 
        get_stance=False, 
        # topic_model_kwargs=toponymy_kwargs,
        text_column='Document',
        parent_text_column='ParentDocument',
        max_layers=1
    )
    target_info_df = model.get_target_info()

    logger.info(f'Finished fitting model with {len(target_info_df)} targets.')

    doc_target_df.write_parquet(f'./data/stance_targets/{period}_{config.stance_target_type}_doc_targets.parquet.zstd', compression='zstd')

    logger.info(f'Saving target info to {period}_{config.stance_target_type}_target_info.parquet.zstd')
    target_info_df.write_parquet(f'./data/stance_targets/{period}_{config.stance_target_type}_target_info.parquet.zstd', compression='zstd')
    logger.info("Successfully saved target info.")

if __name__ == '__main__':
    main()
