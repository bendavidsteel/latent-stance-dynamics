
import gc
import hashlib
import os
import sys

import hydra
import numpy as np
import omegaconf
import polars as pl
import wandb

import ppca_rs

from pca_density import load_df
from pca_clusters import load_text_df


def pivot_no_impute(df):
    df = df.with_columns([
        pl.col('trend_mean').cast(pl.Float32),
        pl.col('filter_value').cast(pl.Enum(df['filter_value'].unique()))
    ])
    target_df = df.pivot(on='target', index=['createtime', 'filter_value'], values=['trend_mean'])
    stance_cols = [c for c in target_df.columns if c not in ['createtime', 'filter_value']]
    return target_df, stance_cols


def preprocess(target_df, stance_cols):
    target_df = target_df.with_columns(
        [pl.col(c).backward_fill().over('filter_value') for c in stance_cols]
    ).with_columns(
        [pl.col(c).forward_fill().over('filter_value') for c in stance_cols]
    )
    return target_df


def compute_metrics(imputed_values, true_values, holdout_cols, stance_cols, label):
    errors = imputed_values - true_values
    abs_errors = np.abs(errors)
    mae = np.mean(abs_errors)
    mae_std = np.std(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    corr = np.corrcoef(true_values, imputed_values)[0, 1]
    avg_vals = (np.abs(true_values) + np.abs(imputed_values)) / 2
    nonzero = avg_vals > 0
    mard = np.mean(abs_errors[nonzero] / avg_vals[nonzero]) if nonzero.any() else float('nan')

    print(f"  [{label}] MAE:  {mae:.6f} (std {mae_std:.6f})")
    print(f"  [{label}] MARD: {mard:.4%}")
    print(f"  [{label}] RMSE: {rmse:.6f}")
    print(f"  [{label}] R²:   {r2:.6f}")
    print(f"  [{label}] Corr: {corr:.6f}")

    per_target_mae = {}
    for col_idx, col_name in enumerate(stance_cols):
        col_mask = holdout_cols == col_idx
        if col_mask.sum() > 0:
            per_target_mae[col_name] = np.mean(abs_errors[col_mask])

    return {
        'mae': mae,
        'mae_std': mae_std,
        'mard': mard,
        'rmse': rmse,
        'r2': r2,
        'corr': corr,
        'per_target_mae': per_target_mae,
    }


def get_seed_df(cfg):
    text_df = load_text_df(cfg, columns=['seed'])
    seed_df = text_df.select([
        pl.col('seed').struct.field('SeedName'),
        pl.col('seed').struct.field('PlatformHandleID'),
        pl.col('seed').struct.field('MainType'),
        pl.col('seed').struct.field('SubType')
    ]).unique(cfg.filter_column)
    seed_df = seed_df.filter(pl.col('MainType').is_in(['politician', 'influencer']) | ((pl.col('MainType') == 'foreign') & (~pl.col('SubType').is_in(['media', 'state']))))\
        .select(cfg.filter_column)
    return seed_df


def get_cache_path(cfg):
    key_str = f"{cfg.trend_path}|{cfg.filter_column}|{cfg.min_filter_count}" \
              f"|{cfg.group_by_every}|{cfg.min_target_volume}"
    h = hashlib.md5(key_str.encode()).hexdigest()[:12]
    return os.path.join(cfg.trend_path, f'impute_cache_{h}.parquet.zstd')


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


def impute_zero(X_masked, holdout_rows, holdout_cols):
    return np.where(np.isnan(X_masked), 0.0, X_masked)[holdout_rows, holdout_cols]


def impute_mean(X_masked, holdout_rows, holdout_cols):
    col_means = np.nanmean(X_masked, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X_filled = X_masked.copy()
    nan_mask = np.isnan(X_filled)
    X_filled[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X_filled[holdout_rows, holdout_cols]


def impute_svd(X_masked, holdout_rows, holdout_cols):
    rank = np.power(X_masked.shape[1], 1/3).astype(int)
    sys.stdout.reconfigure(line_buffering=True)
    from fancyimpute import IterativeSVD
    imp = IterativeSVD(rank=rank, svd_algorithm='arpack')
    X_imputed = imp.fit_transform(X_masked)
    return X_imputed[holdout_rows, holdout_cols]


def impute_ppca(X_masked, holdout_rows, holdout_cols, cfg):
    n_output = X_masked.shape[1]
    prior = ppca_rs.Prior()\
        .with_mean_prior(np.zeros((n_output, 1)), cfg.ppca.mean_prior_variance * np.eye(n_output))\
        .with_transformation_precision(cfg.ppca.transform_precision)\
        .with_isotropic_noise_prior(cfg.ppca.noise_prior_alpha, cfg.ppca.noise_prior_beta)

    ds = ppca_rs.Dataset(X_masked.astype(np.float64))
    trainer = ppca_rs.PPCATrainer(ds)
    model = trainer.train(state_size=cfg.ppca.n_components, n_iters=50, prior=prior)

    inferred = model.infer(ds)
    X_ppca = inferred.extrapolated(model, ds).numpy()
    return X_ppca[holdout_rows, holdout_cols].astype(np.float32)


def run_split(split_idx, X_preprocessed, stance_cols, valid_group_idx, valid_col_idx,
              row_indices_per_group, rng, holdout_fraction, impute_method, cfg):
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

    if impute_method == 'zero':
        imputed = impute_zero(X_masked, holdout_rows, holdout_cols)
    elif impute_method == 'mean':
        imputed = impute_mean(X_masked, holdout_rows, holdout_cols)
    elif impute_method == 'svd':
        imputed = impute_svd(X_masked, holdout_rows, holdout_cols)
    elif impute_method == 'ppca':
        imputed = impute_ppca(X_masked, holdout_rows, holdout_cols, cfg)
    else:
        raise ValueError(f"Unknown impute_method: {impute_method}")

    result = compute_metrics(imputed, true_values, holdout_cols, stance_cols, f'split_{split_idx}')
    wandb.log({
        f'split_{split_idx}/mae': result['mae'],
        f'split_{split_idx}/mae_std': result['mae_std'],
        f'split_{split_idx}/rmse': result['rmse'],
        f'split_{split_idx}/r2': result['r2'],
    })

    return result['mae'], result['mae_std']


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    impute_method = cfg.impute_method
    project_name = 'impute_test'
    wandb_config = omegaconf.OmegaConf.to_object(cfg)
    wandb.init(project=project_name, config=wandb_config)

    target_df = load_or_build_target_df(cfg)
    stance_cols = [c for c in target_df.columns if c not in ['createtime', 'filter_value']]

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
    mae_stds = []

    for i in range(n_splits):
        print(f"=== Split {i+1}/{n_splits} (method={impute_method}) ===")
        mae, mae_std = run_split(
            i, X_preprocessed, stance_cols, valid_group_idx, valid_col_idx,
            row_indices_per_group, rng, holdout_fraction, impute_method, cfg,
        )
        maes.append(mae)
        mae_stds.append(mae_std)
        gc.collect()

    mean_mae = float(np.mean(maes))
    mean_mae_std = float(np.mean(mae_stds))
    print(f"\nMean MAE over {n_splits} splits: {mean_mae:.6f} (mean std {mean_mae_std:.6f})")
    wandb.log({'mean_mae': mean_mae, 'mean_mae_std': mean_mae_std})

    wandb.finish()


if __name__ == '__main__':
    main()
