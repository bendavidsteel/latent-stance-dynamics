import datetime
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    trend_path = cfg.trend_path

    target_path = os.path.join(trend_path, f'{cfg.dim_reduction_method}_coords.parquet.zstd')
    target_df = pl.read_parquet(target_path, columns=['createtime', 'filter_value', 'coord_21d'])

    target_df = target_df.filter(pl.col('createtime') >= datetime.datetime(2022, 1, 1))\
        .filter(pl.col('filter_value') != '')

    # get var(diff(coord)) for each dimension
    coord_diff_var = target_df.sort(['filter_value', 'createtime'])\
        .with_columns([
            pl.col('coord_21d').arr.get(i).diff().over('filter_value').alias(f'dim_{i}_diff') for i in range(21)
        ])
    
if __name__ == '__main__':
    main()