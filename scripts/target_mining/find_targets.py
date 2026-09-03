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

from translate_targets import TRANSLATION_FILE, apply_translations

def get_raw_file(filename, platform):
    raw_path = '~/repos/sitrep/data/digital_trace/raw_platforms'
    year, month, day = filename.split('.')[0].split('_')[1:]
    date_str = datetime.date(int(year), int(month), int(day)).strftime('%Y-%m-%d')
    raw_filename = f"{platform}_{date_str}.parquet.zstd"
    raw_day_df = pl.read_parquet(os.path.join(raw_path, raw_filename))
    return raw_day_df

# Targets get characters inserted to dodge moderation ('t rump') or their spacing
# dropped ('nationaldebt'); on their letters alone these are the same target.
# \p{L} rather than a-z so non-Latin scripts do not collapse onto stray Latin characters.
SPELLING_KEY = r'[^0-9\p{L}]+'


def normalize_spelling_variants(df: pl.DataFrame) -> pl.DataFrame:
    """Merge targets that differ only in where their non-letter characters fall."""
    target_df = df.select('Targets').explode('Targets').drop_nulls().rename({'Targets': 'Target'})\
        .group_by('Target').agg(pl.len().alias('count'))\
        .with_columns(pl.col('Target').str.replace_all(SPELLING_KEY, '').alias('key'))\
        .filter(pl.col('key').str.len_chars() > 0)
    target_df = target_df.filter(pl.col('key').is_duplicated())

    # most used spelling wins, then the shortest, matching how clusters pick a primary target
    canonical = target_df.sort(['count', pl.col('Target').str.len_chars()], descending=[True, False])\
        .unique('key', keep='first').select(['key', pl.col('Target').alias('Canonical')])
    mapper = target_df.join(canonical, on='key')\
        .filter(pl.col('Target') != pl.col('Canonical'))\
        .select(['Target', 'Canonical'])

    remapped = df.select('Targets').with_row_index('row_id')\
        .explode('Targets').rename({'Targets': 'Target'})\
        .join(mapper, on='Target', how='left')\
        .select(['row_id', pl.coalesce(['Canonical', 'Target']).alias('Target')])\
        .drop_nulls('Target').unique(['row_id', 'Target'])\
        .group_by('row_id').agg(pl.col('Target').alias('Targets'))
    return df.with_row_index('row_id').drop('Targets')\
        .join(remapped, on='row_id', how='left')\
        .with_columns(pl.col('Targets').fill_null([]))\
        .drop('row_id')

def deduplicate_on_targets(document_df: pl.DataFrame, *args, **kwargs) -> pl.DataFrame:
    """Deduplicate similar targets on a frame holding only the target lists.

    deduplicate_all_similar_targets explodes what it is given, so passing whole
    documents carries every text column through ~52M rows and exhausts memory.
    """
    slim_df = document_df.select('Targets').with_row_index('row_id')
    slim_df = deduplicate_all_similar_targets(slim_df, *args, **kwargs)
    return document_df.with_row_index('row_id').drop('Targets')\
        .join(slim_df, on='row_id', how='left')\
        .drop('row_id')

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

        translation_path = os.path.join(os.path.dirname(target_dir.rstrip('/')), TRANSLATION_FILE)
        if os.path.exists(translation_path):
            translation_df = pl.read_parquet(translation_path)
            n_translated = translation_df.filter(pl.col('TargetEnglish').is_not_null()).height
            document_df = apply_translations(document_df, translation_df)
            logger.info(f'Applied {n_translated} target translations.')
        else:
            logger.warning(f'No target translations at {translation_path}; run translate_targets.py first.')

    # remove bad targets
    print(f"Before filtering bad targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets present.")
    document_df = remove_doc_bad_targets(document_df, config.stance_target_type)
    print(f"After filtering bad targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets remain.")

    document_df = filter_to_common_targets(document_df, num_targets=int(config.get('num_common_targets', 3000000)))
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
        document_df = deduplicate_on_targets(document_df, model.embedding_model, config.stance_target_type, batch_size=2000000)
        logger.info(f"After de-duplicating similar targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets remain.")

    # HDBSCAN's default 'eom' selection keeps the few largest stable clusters, which
    # collapses the targets into a handful of huge ones whose labels are then broadcast
    # over the corpus; 'leaf' takes the finest-grained clusters instead.
    from cuml.cluster import HDBSCAN
    topic_model_kwargs = {
        'hdbscan_model': HDBSCAN(
            min_samples=10,
            min_cluster_size=int(config.get('min_cluster_size', 200)),
            cluster_selection_method='leaf',
            verbose=True,
        )
    }

    logger.info('Fitting model...')

    doc_target_df = model.fit_transform(
        document_df, 
        get_stance=False, 
        topic_model_kwargs=topic_model_kwargs,
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
