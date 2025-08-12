import datetime
import logging
import os
import re

import hydra
import numpy as np
import polars as pl
from tqdm import tqdm
import torch

from stancemining.main import StanceMining


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    logger = logging.getLogger('find_targets')

    pl.set_random_seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    period = '2025-01-onwards'
    df_path = f'./data/stance_targets/{period}_doc_targets.parquet.zstd'

    doc_target_df = pl.read_parquet(df_path)

    model = StanceMining(
        model_name='Qwen/Qwen3-4B',
        stance_target_type=config.stance_target_type,
        topic_model='bertopic',
        verbose=True,
    )

    logger.info('Fitting model...')

    doc_target_df = model.fit_transform(
        doc_target_df, 
        get_stance=False, 
        generate_higher_level_targets=False,
        dbscan_deduplicate=True,
        text_column='Document',
        parent_text_column='ParentDocument',
        max_layers=2
    )
    
    doc_target_df.write_parquet(df_path, compression='zstd')


if __name__ == '__main__':
    main()
