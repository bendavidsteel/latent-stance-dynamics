import datetime
import logging
import os
import re

import hydra
import numpy as np
import polars as pl
from tqdm import tqdm
import torch
import transformers

from stancemining.main import StanceMining
from stancemining.utils import remove_bad_targets

def get_raw_file(filename, platform):
    raw_path = '~/repos/sitrep/data/digital_trace/raw_platforms'
    year, month, day = filename.split('.')[0].split('_')[1:]
    date_str = datetime.date(int(year), int(month), int(day)).strftime('%Y-%m-%d')
    raw_filename = f"{platform}_{date_str}.parquet.zstd"
    raw_day_df = pl.read_parquet(os.path.join(raw_path, raw_filename))
    return raw_day_df


def remove_doc_bad_targets(df: pl.DataFrame):
    df = df.with_row_index()
    target_df = df.select(['index', 'Targets']).explode('Targets').rename({'Targets': 'Target'})
    target_df = remove_bad_targets(target_df)
    target_df = target_df.group_by('index').agg(pl.col('Target')).rename({'Target': 'Targets'})
    df = df.drop('Targets').join(target_df, on='index', how='left').with_columns(pl.col('Targets').fill_null([]))
    df = df.drop('index')
    return df

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    logger = logging.getLogger('find_targets')

    pl.set_random_seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    target_dir = config.base_target_path
    target_df = pl.DataFrame()
    for filename in tqdm(os.listdir(target_dir), desc='Loading files...'):
        if re.match('targets_\d{4}_\d{1,2}_\d{1,2}.parquet.zstd', filename):
            target_file_df = pl.read_parquet(f'{target_dir}/{filename}')
            target_df = pl.concat([target_df, target_file_df], how='diagonal_relaxed')
    
    logger.info(f'Loaded {target_df.shape[0]} targets from {len(os.listdir(target_dir))} files.')

    target_df = target_df.unique(['id', 'platform'])

    target_df = target_df.filter(pl.col('createtime') >= datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc))

    # remove bad targets
    target_df = remove_doc_bad_targets(target_df)

    logger.info(f'After removing bad targets, {target_df.shape[0]} targets remain.')

    model = StanceMining(
        model_name='Qwen/Qwen3-4B',
        stance_target_type=config.stance_target_type,
        topic_model='bertopic',
        verbose=True,
    )

    target_path = './data/stance_targets/unique_targets.parquet.zstd'
    if not os.path.exists(target_path):
        unique_target_df = target_df.select('Targets').explode('Targets').unique('Targets')
        unique_target_df.write_parquet(target_path, compression='zstd')

        embeddings = model._get_embeddings(unique_target_df['Targets'].to_list())
        # save embeddings
        unique_target_df = unique_target_df.with_columns(pl.Series(name='embeddings', values=embeddings))
        unique_target_df.write_parquet(target_path, compression='zstd')
    else:
        unique_target_df = pl.read_parquet(target_path)
    unique_target_df = unique_target_df.rename({'Targets': 'text', 'embeddings': 'embedding'})

    toponymy_kwargs = {
        'clusterer': {
            'base_min_cluster_size': 20
        }
    }

    logger.info('Fitting model...')

    doc_target_df = model.fit_transform(
        target_df, 
        get_stance=False, 
        dbscan_deduplicate=False,
        embedding_cache=unique_target_df, 
        # topic_model_kwargs=toponymy_kwargs,
        text_column='Document',
        parent_text_column='ParentDocument',
        max_layers=2
    )
    target_info_df = model.get_target_info()

    logger.info(f'Finished fitting model with {len(target_info_df)} targets.')

    period = '2025-01-onwards'
    doc_target_df.write_parquet(f'./data/stance_targets/{period}_doc_targets.parquet.zstd', compression='zstd')

    logger.info(f'Saving target info to {period}_target_info.parquet.zstd')
    target_info_df.write_parquet(f'./data/stance_targets/{period}_target_info.parquet.zstd', compression='zstd')
    logger.info("Successfully saved target info.")

    topic_model = model.get_topic_model()
    topic_model.save(f"./data/stance_targets/{period}_topic_model.pickle", serialization="pickle")

if __name__ == '__main__':
    main()
