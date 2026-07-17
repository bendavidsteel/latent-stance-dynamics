
import gc
import hashlib
import os

import hydra
import numpy as np
import omegaconf
import polars as pl
import wandb

import ppca_rs

from pca_density import load_df
from pca_clusters import load_text_df
from gridsearch_ppca_impute import pivot_no_impute, preprocess, compute_metrics, impute_ppca, get_seed_df


def get_cache_path(cfg):
    key_str = f"{cfg.trend_path}|{cfg.filter_column}|{cfg.min_filter_count}" \
              f"|{cfg.group_by_every}|{cfg.min_target_volume}"
    h = hashlib.md5(key_str.encode()).hexdigest()[:12]
    return os.path.join(cfg.trend_path, f'ppca_impute_cache_{h}.parquet.zstd')


def load_or_build_target_df(cfg):
    cache_path = get_cache_path(cfg)
    if os.path.exists(cache_path):
        print(f"Loading cached target_df from {cache_path}")
        return pl.read_parquet(cache_path)

    print("Building target_df from scratch...")
    df = load_df(cfg.trend_path, cfg.filter_column, min_filter_count=cfg.min_filter_count, group_by_every=cfg.group_by_every)
    df = df.select(['createtime', 'volume', 'trend_mean', 'target', 'filter_type', 'filter_value'])

    df = df.filter(pl.col('filter_value') != '')\
        .filter(pl.col('filter_type') == cfg.filter_column)\
        .drop('filter_type')

    top_target_df = df.group_by('target').agg(pl.col('volume').sum()).filter(pl.col('volume') >= cfg.min_target_volume)
    df = df.join(top_target_df.select('target'), on='target', how='inner').drop('volume')

    seed_df = get_seed_df(cfg)
    df = df.join(seed_df, left_on='filter_value', right_on=cfg.filter_column, how='inner')

    target_df, stance_cols = pivot_no_impute(df)
    del df
    target_df = preprocess(target_df, stance_cols)

    target_df.write_parquet(cache_path, compression='zstd')
    print(f"Cached target_df to {cache_path}")
    return target_df


def run_split(split_idx, X_preprocessed, stance_cols, valid_group_idx, valid_col_idx,
              row_indices_per_group, rng, holdout_fraction,
              n_components, mean_prior_variance, transform_precision,
              noise_prior_alpha, noise_prior_beta):
    n_total_series = len(valid_group_idx)
    n_holdout = int(n_total_series * holdout_fraction)
    holdout_sel = rng.choice(n_total_series, size=n_holdout, replace=False)
    holdout_groups = valid_group_idx[holdout_sel]
    holdout_col_indices = valid_col_idx[holdout_sel]

    holdout_row_lists = [row_indices_per_group[g] for g in holdout_groups]
    group_sizes = np.array([len(rows) for rows in holdout_row_lists])
    mask_rows = np.concatenate(holdout_row_lists)
    mask_cols = np.repeat(holdout_col_indices, group_sizes)

    all_vals = X_preprocessed[mask_rows, mask_cols]
    known_mask = ~np.isnan(all_vals)
    true_values = all_vals[known_mask]
    holdout_rows = mask_rows[known_mask]
    holdout_cols = mask_cols[known_mask]

    X_masked = X_preprocessed.copy()
    X_masked[mask_rows, mask_cols] = np.nan

    n_output = X_masked.shape[1]
    prior = ppca_rs.Prior()\
        .with_mean_prior(np.zeros((n_output, 1)), mean_prior_variance * np.eye(n_output))\
        .with_transformation_precision(transform_precision)\
        .with_isotropic_noise_prior(noise_prior_alpha, noise_prior_beta)

    ds = ppca_rs.Dataset(X_masked.astype(np.float64))
    trainer = ppca_rs.PPCATrainer(ds)
    model = trainer.train(state_size=n_components, n_iters=50, prior=prior)

    inferred = model.infer(ds)
    X_ppca = inferred.extrapolated(model, ds).numpy()
    ppca_imputed = X_ppca[holdout_rows, holdout_cols].astype(np.float32)

    result = compute_metrics(ppca_imputed, true_values, holdout_cols, stance_cols, f'split_{split_idx}')
    wandb.log({
        f'split_{split_idx}/mae': result['mae'],
        f'split_{split_idx}/rmse': result['rmse'],
        f'split_{split_idx}/r2': result['r2'],
    })

    return result['mae']


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    project_name = 'ppca_impute_test'
    wandb_config = omegaconf.OmegaConf.to_object(cfg)
    wandb.init(project=project_name, config=wandb_config)

    target_df = load_or_build_target_df(cfg)
    stance_cols = [c for c in target_df.columns if c not in ['createtime', 'filter_value']]

    n_components = cfg.ppca.n_components
    mean_prior_variance = cfg.ppca.mean_prior_variance
    transform_precision = cfg.ppca.transform_precision
    noise_prior_alpha = cfg.ppca.noise_prior_alpha
    noise_prior_beta = cfg.ppca.noise_prior_beta

    filter_values = target_df['filter_value'].cast(pl.String).to_numpy()
    X_preprocessed = target_df.select(stance_cols).to_numpy().astype(np.float32)
    del target_df

    unique_fvs, group_ids = np.unique(filter_values, return_inverse=True)
    n_groups = len(unique_fvs)
    not_nan = ~np.isnan(X_preprocessed)
    row_indices_per_group = [np.where(group_ids == g)[0] for g in range(n_groups)]
    known_counts = np.array([not_nan[rows].sum(axis=0) for rows in row_indices_per_group])

    valid_group_idx, valid_col_idx = np.where(known_counts > 0)

    rng = np.random.default_rng(42)
    holdout_fraction = 0.01
    n_splits = 2
    maes = []

    for i in range(n_splits):
        print(f"=== Split {i+1}/{n_splits} ===")
        mae = run_split(
            i, X_preprocessed, stance_cols, valid_group_idx, valid_col_idx,
            row_indices_per_group, rng, holdout_fraction,
            n_components, mean_prior_variance, transform_precision,
            noise_prior_alpha, noise_prior_beta,
        )
        maes.append(mae)
        gc.collect()

    mean_mae = np.mean(maes)
    print(f"\nMean MAE over {n_splits} splits: {mean_mae:.6f}")
    wandb.log({'mean_mae': mean_mae})

    wandb.finish()


if __name__ == '__main__':
    main()
