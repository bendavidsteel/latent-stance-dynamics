
import os

import hydra
import polars as pl

from pca_density import do_pca

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    # Load the original training data and fit PCA
    pca_source_path = os.path.join('data', 'stance_targets', 'noun_phrase_bkrr_trends', 'pivoted_and_imputed.parquet.zstd')
    pca_df = pl.read_parquet(pca_source_path)
    stance_cols = [col for col in pca_df.columns if col not in ['createtime', 'filter_value']]

    n_dims = 21
    pca, _, _, _ = do_pca(pca_df, stance_cols, n_components=n_dims)

    # Get mean from source data for filling missing columns in target
    source_means = {col: pca_df[col].mean() for col in stance_cols}

    # Load target data to transform
    trend_path = cfg.trend_path
    target_path = os.path.join(trend_path, 'pivoted_and_imputed.parquet.zstd')
    target_df = pl.read_parquet(target_path)

    # Align columns: add any missing columns filled with source mean
    missing_cols = [col for col in stance_cols if col not in target_df.columns]
    if missing_cols:
        target_df = target_df.with_columns([
            pl.lit(source_means[col]).alias(col) for col in missing_cols
        ])

    # Apply the fitted PCA
    X = target_df.select(stance_cols).to_numpy()
    coords = pca.apply(X)

    # Save results
    target_df = target_df.with_columns(pl.Series(name=f'coord_{n_dims}d', values=coords))
    result_df = target_df.select(['createtime', 'filter_value', f'coord_{n_dims}d'])

    output_path = os.path.join(trend_path, 'pca_coords.parquet.zstd')
    result_df.write_parquet(output_path, compression='zstd')

if __name__ == '__main__':
    main()
