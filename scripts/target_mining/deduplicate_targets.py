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
from stancemining.utils import remove_doc_bad_targets

from find_targets import deduplicate_on_targets, filter_to_common_targets, normalize_spelling_variants
from translate_targets import TRANSLATION_FILE, apply_translations

def get_raw_file(filename, platform):
    raw_path = '~/repos/sitrep/data/digital_trace/raw_platforms'
    year, month, day = filename.split('.')[0].split('_')[1:]
    date_str = datetime.date(int(year), int(month), int(day)).strftime('%Y-%m-%d')
    raw_filename = f"{platform}_{date_str}.parquet.zstd"
    raw_day_df = pl.read_parquet(os.path.join(raw_path, raw_filename))
    return raw_day_df

TRAILING_PUNCT = ',.:?!"\'“”‘’…'
LEADING_PUNCT = '@#"\'“”‘’'

EMOJI_REGEX = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]'

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    logger = logging.getLogger('find_targets')

    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    mp.set_start_method('spawn')
    pl.set_random_seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    target_dir = config.base_target_path
    document_df = pl.read_parquet([f'{target_dir}/{filename}' for filename in os.listdir(target_dir) if re.match('targets_\d{4}_\d{1,2}.parquet.zstd', filename)])
        
    logger.info(f'Loaded {document_df.shape[0]} documents from {len(os.listdir(target_dir))} files.')

    document_df = document_df.unique(['id', 'platform'])

    start_date_str = '2022-01-01'
    period = f'{start_date_str}-onwards'
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
    document_df = document_df.filter(pl.col('createtime') >= start_date)
    logger.info(f'Filtered documents to {document_df.shape[0]} after {start_date_str}.')

    translation_path = os.path.join(os.path.dirname(target_dir.rstrip('/')), TRANSLATION_FILE)
    if os.path.exists(translation_path):
        translation_df = pl.read_parquet(translation_path)
        n_translated = translation_df.filter(pl.col('TargetEnglish').is_not_null()).height
        document_df = apply_translations(document_df, translation_df)
        logger.info(f'Applied {n_translated} target translations.')
    else:
        logger.warning(f'No target translations at {translation_path}; run translate_targets.py first.')

    document_df = document_df.with_columns(pl.col('Targets').list.eval(pl.element().str.replace(r'\s*\([^)]*\)', '')\
                                                                       .str.replace_all(EMOJI_REGEX, '')\
                                                                       .str.strip_chars_end(TRAILING_PUNCT)\
                                                                       .str.strip_chars_start(LEADING_PUNCT)\
                                                                       .str.replace(',.*', '')\
                                                                       .str.replace('^rt @ ', '')\
                                                                       .str.replace('^rt@', '')\
                                                                       .str.strip_chars()))

    n_before = document_df.select('Targets').explode('Targets').n_unique()
    document_df = normalize_spelling_variants(document_df)
    n_after = document_df.select('Targets').explode('Targets').n_unique()
    logger.info(f'Spelling normalisation merged {n_before - n_after} targets, {n_after} remain.')

    # remove bad targets
    print(f"Before filtering bad targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets present.")
    document_df = remove_doc_bad_targets(document_df, config.stance_target_type)
    print(f"After filtering bad targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets remain.")

    document_df = filter_to_common_targets(document_df, num_targets=int(config.get('num_common_targets', 3000000)))
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

    # reducing number of targets
    logger.info(f"Before de-duplicating similar targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets present.")
    
    if document_df.select('Targets').explode('Targets').unique('Targets').shape[0] < 4000000:
        document_df = deduplicate_on_targets(document_df, model.embedding_model, config.stance_target_type, batch_size=3000000, minhash_threshold=0.5, max_embedding_distance=0.15)
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
                    batch = deduplicate_on_targets(batch, model.embedding_model, config.stance_target_type, batch_size=batch_size, minhash_threshold=minhash_threshold, max_embedding_distance=max_embedding_distance)
                    new_document_df = pl.concat([new_document_df, batch], how='diagonal_relaxed')
                document_df = new_document_df
            else:
                document_df = deduplicate_on_targets(document_df, model.embedding_model, config.stance_target_type, batch_size=100000, minhash_threshold=minhash_threshold, max_embedding_distance=max_embedding_distance)
            logger.info(f"After de-duplicating similar targets with minhash_threshold={minhash_threshold} and max_embedding_distance={max_embedding_distance}, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets remain.")
            minhash_threshold -= 0.1
            minhash_threshold = max(minhash_threshold, 0.2)
            max_embedding_distance += 0.05
            document_df = document_df.sample(fraction=1.0, shuffle=True)

    logger.info(f"After de-duplicating similar targets, {document_df.select('Targets').explode('Targets').unique('Targets').shape[0]} targets remain.")

    document_df.write_parquet(f'./data/stance_targets/{period}_{config.stance_target_type}_doc_targets_deduplicated.parquet.zstd', compression='zstd')

if __name__ == '__main__':
    main()
