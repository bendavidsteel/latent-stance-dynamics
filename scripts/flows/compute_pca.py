
import os

import hydra
import polars as pl

from pca_density import do_pca

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    trend_path = cfg.trend_path
    
    target_path = os.path.join(cfg.trend_path, 'pca_coords.parquet.zstd')
    source_path = os.path.join(cfg.trend_path, 'pivoted_and_imputed.parquet.zstd')
    target_df = pl.read_parquet(source_path)
    stance_cols = [col for col in target_df.columns if col not in ['createtime', 'filter_value']]

    pca_metadata = []
    n_dims = 21
    pca, coords, components, explained_variance_ratio = do_pca(target_df, stance_cols, n_components=n_dims)
    assert len(stance_cols) == components.shape[1]
    target_df = target_df.with_columns(pl.Series(name=f'coord_{n_dims}d', values=coords))
    target_df = target_df.select(['createtime', 'filter_value', f'coord_{n_dims}d'])
    pca_metadata.append({
        'n_dims': n_dims,
        'explained_variance_ratio': explained_variance_ratio,
        'components': components
    })
        # np.save(os.path.join(trend_path, f'pca_components_{n_dims}d.npy'), components)
    pca_metadata_df = pl.from_dicts([{'n_dims': p['n_dims'], 'explained_variance_ratio': p['explained_variance_ratio'].tolist(), 'components': p['components'].tolist()} for p in pca_metadata])
    pca_metadata_df.write_parquet(os.path.join(trend_path, 'pca_metadata.parquet.zstd'), compression='zstd')
    target_df.write_parquet(target_path, compression='zstd')

if __name__ == '__main__':
    main()