
import os
import warnings

import hydra
import numpy as np
import polars as pl
from sklearn.utils.validation import check_array
from sklearn.decomposition import PCA, IncrementalPCA
from tqdm import tqdm

import sksfa


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    trend_path = cfg.trend_path
    
    target_path = os.path.join(cfg.trend_path, 'sfa_coords.parquet.zstd')
    source_path = os.path.join(cfg.trend_path, 'pivoted_and_imputed.parquet.zstd')
    target_df = pl.read_parquet(source_path)
    stance_cols = [col for col in target_df.columns if col.startswith('trend_mean_')]
   
    target_df = target_df.filter(pl.col('filter_value') != '')

    target_df = target_df.sort(['filter_value', 'createtime'])\
        .with_columns(pl.col('createtime').shift(-1).over('filter_value').alias('createtime_next'))\
        .filter(pl.col('createtime_next').is_not_null())\
        .with_columns((pl.col('createtime_next') - pl.col('createtime')).alias('dt'))\
        .filter(pl.col('dt') <= pl.duration(days=1))

    batch_indices = target_df.with_row_index()\
        .group_by('filter_value')\
        .agg([pl.col('index').min().alias('start_idx'), pl.col('index').max().alias('end_idx')])\
        .select(['start_idx', 'end_idx'])\
        .rows()

    X = target_df.select(stance_cols).to_numpy()

    metadatas = []
    n_components = 21
    print(f"Processing n_components: {n_components}")
    # Step 1: Dimensionality reduction with VAMP
    model = sksfa.SFA(n_components=n_components, fill_mode='fastest', robustness_cutoff=0.05)
    X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True,
                    copy=model.copy, ensure_min_features=1)
    # check_estimators test expects feature warnings before sample warnings
    X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True,
                    copy=model.copy,
                    ensure_min_samples=10)
    model.input_dim_ = X.shape[1]
    model.n_components_ = model.n_components
    model.pca_whiten_ = IncrementalPCA(whiten=True, batch_size=10000)
    # initialize internal pca methods
    model.pca_diff_ = IncrementalPCA()

    n_samples, input_dim = X.shape
    print("Fitting PCA whitening...")
    model.pca_whiten_.fit(X)
    print("Transforming data with PCA whitening...")
    X_whitened = model.pca_whiten_.transform(X)

    # Find non-trivial components
    input_evr = model.pca_whiten_.explained_variance_ratio_
    nontrivial_indices = np.argwhere(input_evr > model.robustness_cutoff)
    model.nontrivial_indices_ = nontrivial_indices.reshape((-1,))
    model.n_nontrivial_components_ = model.nontrivial_indices_.shape[0]
    model.n_trivial_ = input_dim - model.n_nontrivial_components_
    X_whitened = X_whitened[:, model.nontrivial_indices_]

    X_diff = X_whitened[1:] - X_whitened[:-1]
    model._diff_mean = X_diff.mean(axis=0)
    X_diff -= model._diff_mean
    for start_idx, end_idx in tqdm(batch_indices, desc="Fitting SFA differences"):
        current_batch = X_whitened[start_idx:end_idx + 1]
        batch_diff = current_batch[1:] - current_batch[:-1]
        model.pca_diff_.partial_fit(batch_diff - model._diff_mean)
    if model.n_nontrivial_components_ == 0:
        raise ValueError(f"While whitening, only trivial components were \
                found. This can be caused by passing 0-only input.")
    if model.n_nontrivial_components_ < model.n_components_:
        warning_string = f"During whitening, {model.n_trivial_} trivial components "\
                "with roughly zero explained variance have been found. This "\
                "probably means that the effective dimension of your input is "\
                "too low to find the desired {model.n_components_} slow features. "\
                "Ways to deal with this are:\n\tInjecting noise via the 'noise_std' "\
                "parameter.\n\tProviding more data.\n\tLowering the 'n_components' "\
                "parameter.\n\tLowering the threshold 'robustness_cutoff'."
        if model.fill_mode is not None:
            warning_string += f"\nSince 'fill_mode' is set to {model.fill_mode}, "\
                    "missing output features will be filled by "
            if model.fill_mode == "zero":
                warning_string += "a 0 signal."
            if model.fill_mode == "fastest":
                warning_string += "duplicates of the fastest signal."
            if model.fill_mode == "noise":
                warning_string += "independent white-noise."
            warnings.warn(warning_string, RuntimeWarning)
        else:
            warning_string += "\n\tSetting 'fill_mode' to replace trivial components\
                    with uninformative signals."
            raise ValueError(warning_string)
    model._compute_delta_values()
    model.is_fitted_ = True
    # trajectories = [model.transform(traj) for traj in tqdm(trajectories_data, desc="Transforming trajectories")]

    print("Transforming data...")
    y = X_whitened
    y = y[:, model.nontrivial_indices_]
    y = model.pca_diff_.transform(y)
    n_missing_components = max(model.n_components_ - y.shape[1], 0)
    if n_missing_components > 0:
        if model.fill_mode == "zero":
            y = np.pad(y, ((0, 0), (n_missing_components, 0)))
        if model.fill_mode == "fastest":
            y = np.pad(y, ((0, 0), (n_missing_components, 0)), "edge")
        if model.fill_mode == "noise":
            missing = np.random.normal(0, 1, (y.shape[0], n_missing_components))
            missing -= missing.mean(axis=0)
            missing /= missing.std(axis=0)
            y = np.hstack([missing, y])
        y = y[:, ::-1]
    else:
        y = y[:, -model.n_components_:][:, ::-1]

    coords = y
    W, b = model.affine_parameters()

    target_df = target_df.with_columns(pl.Series(name=f'coord_{n_components}d', values=coords))

    metadata = {}
    metadata['W'] = W
    metadata['b'] = b
    metadata['delta_values'] = model.delta_values_
    metadata['n_nontrivial_components'] = model.n_nontrivial_components_
    metadata['n_components'] = n_components
    metadatas.append(metadata)
    sfa_metadata_df = pl.from_dicts(metadatas)
    sfa_metadata_df.write_parquet(os.path.join(trend_path, 'sfa_metadata.parquet.zstd'), compression='zstd')
    target_df.write_parquet(target_path, compression='zstd')

if __name__ == '__main__':
    main()