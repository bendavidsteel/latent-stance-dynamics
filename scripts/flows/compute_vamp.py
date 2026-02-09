
import os

import hydra
import numpy as np
import polars as pl
from tqdm import tqdm

from deeptime.decomposition import VAMP
from deeptime.util.data import TrajectoriesDataset

from pca_density import load_df, pivot_and_impute
from pca_clusters import load_text_df

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    trend_path = cfg.trend_path
    
    target_path = os.path.join(cfg.trend_path, 'vamp_coords.parquet.zstd')
    source_path = os.path.join(cfg.trend_path, 'pivoted_and_imputed.parquet.zstd')
    target_df = pl.read_parquet(source_path)
    stance_cols = [col for col in target_df.columns if col not in ['createtime', 'filter_value']]
   
    group_by_every = cfg.group_by_every
    assert group_by_every[-1] == 'd', "group_by_every should end with 'd' for days"
    days = int(group_by_every[:-1])

    target_df = target_df.sort(['filter_value', 'createtime'])\
        .with_columns(pl.col('createtime').shift(-1).over('filter_value').alias('createtime_next'))\
        .filter(pl.col('createtime_next').is_not_null())\
        .with_columns((pl.col('createtime_next') - pl.col('createtime')).alias('dt'))\
        .filter(pl.col('dt') <= pl.duration(days=days))

    filter_dfs = target_df.select(stance_cols + ['createtime', 'filter_value']).partition_by('filter_value')
    trajectories_data = [filter_df.sort('createtime')[stance_cols].to_numpy() for filter_df in filter_dfs]

    metadata_cols = ['instantaneous_coefficients', 'singular_values', 'timelagged_coefficients', 'var_cutoff', 'epsilon']

    chosen_lags = [1, 5, 10, 50, 100, 500]
    metadatas = []
    for chosen_lag in chosen_lags:
        print(f"Processing lagtime: {chosen_lag}")
        dataset = TrajectoriesDataset.from_numpy(chosen_lag, trajectories_data)
        
        # Step 1: Dimensionality reduction with VAMP
        vamp = VAMP(
            lagtime=chosen_lag,  # adjust based on your data
            var_cutoff=0.6,
        )
        model = vamp.fit(dataset).fetch_model()
        trajectories = [model.transform(traj) for traj in tqdm(trajectories_data, desc="Transforming trajectories")]

        filter_dfs = [filter_df.with_columns(pl.Series(name=f'coord_{traj.shape[1]}d', values=traj)) for filter_df, traj in zip(filter_dfs, trajectories)]
        target_df = pl.concat(filter_dfs)

        metadata = {k: v[np.newaxis,:] if isinstance(v, np.ndarray) else v for k, v in model.get_params().items() if k in metadata_cols}
        metadata['lagtime'] = chosen_lag
        metadatas.append(metadata)
    vamp_metadata_df = pl.from_dicts(metadatas)
    vamp_metadata_df.write_parquet(os.path.join(trend_path, 'vamp_metadata.parquet.zstd'), compression='zstd')
    target_df.write_parquet(target_path, compression='zstd')

if __name__ == '__main__':
    main()