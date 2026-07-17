
import os

import hydra
import numpy as np
import polars as pl

import ppca_rs
import tqdm

from pca_density import load_df, pivot_trends
from pca_clusters import load_text_df


MODEL_FILENAME = 'ppca_model.bin'


def get_dataset(target_df, feature_cols):
    X = target_df.select(feature_cols).to_numpy().astype(np.float64)
    n_missing = np.isnan(X).sum()
    print(f"Missing values: {n_missing} ({n_missing / X.size:.2%})")
    return ppca_rs.Dataset(X)

def save_ppca_model(model, path):
    with open(path, 'wb') as f:
        f.write(model.dump())

def load_ppca_model(path):
    with open(path, 'rb') as f:
        return ppca_rs.PPCAModel.load(f.read())

def model_metadata(model):
    # transform is (output_size, state_size), transpose to (state_size, output_size)
    components = np.array(model.transform).T

    # ppca_rs.singular_values[i]**2 equals the i-th singular value of the transform C
    # (verified empirically; the library's docstring is misleading). So signal variance
    # along principal direction i is s_i**2 = sv**4, and the implied sample-covariance
    # eigenvalue is sv**4 + sigma**2. Total variance = trace(C C^T) + d * sigma**2.
    sv = np.asarray(model.singular_values)
    sigma2 = model.isotropic_noise ** 2
    signal_variance = sv ** 4
    total_variance = signal_variance.sum() + model.output_size * sigma2
    explained_variance_ratio = (signal_variance + sigma2) / total_variance
    return components, explained_variance_ratio

def extrapolate_dataset(model, ds):
    print("Getting extrapolated values...")
    n_chunks = max(1, len(ds) // 1000)
    extrapolated = np.empty((len(ds), model.output_size), dtype=np.float64)
    offset = 0
    for chunk in tqdm.tqdm(ds.chunks(n_chunks), total=n_chunks, desc="Extrapolating"):
        chunk_out = model.infer(chunk).extrapolated(model, chunk).numpy()
        extrapolated[offset:offset + chunk_out.shape[0]] = chunk_out
        offset += chunk_out.shape[0]
    return extrapolated

def apply_ppca(model, target_df, feature_cols):
    if len(feature_cols) != model.output_size:
        raise ValueError(
            f"Column count mismatch: model expects {model.output_size} features, got {len(feature_cols)}."
        )
    ds = get_dataset(target_df, feature_cols)
    coords = model.infer(ds).states()
    extrapolated = extrapolate_dataset(model, ds)
    return coords, extrapolated

def do_ppca(target_df, feature_cols, n_components, n_iters=50, \
             transform_precision=1.0, noise_prior_alpha=2.0, noise_prior_beta=1.0, mean_prior_variance=1.0):
    ds = get_dataset(target_df, feature_cols)

    print(f"Fitting PPCA with state_size={n_components}, n_iters={n_iters}...")

    n_output = target_df.select(feature_cols).shape[1]
    prior = ppca_rs.Prior()\
        .with_mean_prior(np.zeros((n_output, 1)), np.eye(n_output) * mean_prior_variance)\
        .with_transformation_precision(transform_precision)\
        .with_isotropic_noise_prior(noise_prior_alpha, noise_prior_beta)

    trainer = ppca_rs.PPCATrainer(ds)
    model = trainer.train(state_size=n_components, n_iters=n_iters, prior=prior)

    print("PPCA training complete.")

    coords = model.infer(ds).states()
    components, explained_variance_ratio = model_metadata(model)
    extrapolated = extrapolate_dataset(model, ds)

    return model, coords, components, explained_variance_ratio, extrapolated


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    trend_path = cfg.trend_path
    target_path = os.path.join(trend_path, 'ppca_coords.parquet.zstd')
    all_trend_path = os.path.join(trend_path, 'loaded_trends.parquet.zstd')

    source_dir = os.path.join('data', 'stance_targets', 'noun_phrase_bkrr_trends')
    source_pivoted_path = os.path.join(source_dir, 'pivoted_and_imputed.parquet.zstd')
    source_model_path = os.path.join(source_dir, MODEL_FILENAME)
    source_stance_cols = None
    if os.path.exists(source_pivoted_path) \
            and os.path.abspath(source_pivoted_path) != os.path.abspath(target_path):
        use_source_pca = True
        source_df = pl.read_parquet(source_pivoted_path)
        source_stance_cols = [col for col in source_df.columns if col not in ['createtime', 'filter_value']]
    else:
        use_source_pca = False

    if os.path.exists(all_trend_path):
        df = pl.read_parquet(all_trend_path, columns=['createtime', 'volume', 'trend_mean', 'target', 'filter_type', 'filter_value'])
    else:
        if use_source_pca:
            df = load_df(trend_path, cfg.filter_column, targets=source_stance_cols, group_by_every=cfg.group_by_every)
        else:
            df = load_df(trend_path, cfg.filter_column, min_filter_count=cfg.min_filter_count, group_by_every=cfg.group_by_every)
        df = df.select(['createtime', 'volume', 'trend_mean', 'target', 'filter_type', 'filter_value'])
        df.write_parquet(all_trend_path, compression='zstd')

    df = df.filter(pl.col('filter_value') != '')\
        .filter(pl.col('filter_type') == cfg.filter_column)\
        .drop('filter_type')

    if not use_source_pca:
        top_target_df = df.group_by('target').agg(pl.col('volume').sum()).filter(pl.col('volume') >= cfg.min_target_volume)
        df = df.join(top_target_df.select('target'), on='target', how='inner').drop('volume')

    if use_source_pca:
        df = df.filter(pl.col('target').is_in(source_stance_cols))
        missing_cols = list(set(source_stance_cols) - set(df['target'].unique()))
        if len(missing_cols) > 50:
            raise ValueError(f"Too many missing columns: {len(missing_cols)}.")

    text_df = load_text_df(cfg, columns=['seed'])
    seed_df = text_df.select([
        pl.col('seed').struct.field('SeedName'),
        pl.col('seed').struct.field('PlatformHandleID'),
        pl.col('seed').struct.field('MainType'),
        pl.col('seed').struct.field('SubType')
    ]).unique(cfg.filter_column)
    seed_df = seed_df.filter(pl.col('MainType').is_in(['politician', 'influencer']) | ((pl.col('MainType') == 'foreign') & (~pl.col('SubType').is_in(['media', 'state']))))\
        .select(cfg.filter_column)
    df = df.join(seed_df, left_on='filter_value', right_on=cfg.filter_column, how='inner')

    if use_source_pca:
        target_df, stance_cols = pivot_trends(df, missing_cols=missing_cols)
    else:
        target_df, stance_cols = pivot_trends(df)

    if source_stance_cols is not None:
        target_df = target_df.select(['createtime', 'filter_value'] + source_stance_cols)
        stance_cols = source_stance_cols

    if use_source_pca and os.path.exists(source_model_path):
        print(f"Loading saved PPCA model from {source_model_path}")
        model = load_ppca_model(source_model_path)
        coords, extrapolated = apply_ppca(model, target_df, stance_cols)
        components, explained_variance_ratio = model_metadata(model)
    else:
        model, coords, components, explained_variance_ratio, extrapolated = do_ppca(
            target_df,
            stance_cols,
            n_components=cfg.ppca.n_components,
            transform_precision=cfg.ppca.transform_precision,
            noise_prior_alpha=cfg.ppca.noise_prior_alpha,
            noise_prior_beta=cfg.ppca.noise_prior_beta,
            mean_prior_variance=cfg.ppca.mean_prior_variance
        )
    assert len(stance_cols) == components.shape[1]

    save_ppca_model(model, os.path.join(trend_path, MODEL_FILENAME))

    # Save imputed data
    imputed_df = target_df.select(['createtime', 'filter_value']).hstack(
        pl.DataFrame({col: extrapolated[:, i] for i, col in enumerate(stance_cols)})
    )
    imputed_path = os.path.join(trend_path, 'pivoted_and_imputed.parquet.zstd')
    imputed_df.write_parquet(imputed_path, compression='zstd')

    n_dims = model.state_size
    target_df = target_df.with_columns(pl.Series(name=f'coord_{n_dims}d', values=coords))
    target_df = target_df.select(['createtime', 'filter_value', f'coord_{n_dims}d'])

    pca_metadata_df = pl.from_dicts([{
        'n_dims': n_dims,
        'explained_variance_ratio': explained_variance_ratio.tolist(),
        'components': components.tolist(),
    }])
    pca_metadata_df.write_parquet(os.path.join(trend_path, 'ppca_metadata.parquet.zstd'), compression='zstd')
    target_df.write_parquet(target_path, compression='zstd')

if __name__ == '__main__':
    main()
