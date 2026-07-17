
import os

import hydra
import numpy as np
import polars as pl

import picard


PICA_ROTATION_FILENAME = 'pica_rotation.npz'


def load_ppca_outputs(trend_path):
    coords_path = os.path.join(trend_path, 'ppca_coords.parquet.zstd')
    metadata_path = os.path.join(trend_path, 'ppca_metadata.parquet.zstd')
    if not os.path.exists(coords_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"PPCA outputs missing in {trend_path}. Run compute_ppca.py first."
        )

    coords_df = pl.read_parquet(coords_path)
    coord_col = next(c for c in coords_df.columns if c.startswith('coord_'))
    coords = np.array(coords_df[coord_col].to_list(), dtype=np.float64)

    metadata_df = pl.read_parquet(metadata_path)
    md = metadata_df.row(0, named=True)
    components = np.asarray(md['components'], dtype=np.float64)
    explained_variance_ratio = np.asarray(md['explained_variance_ratio'], dtype=np.float64)
    n_dims = int(md['n_dims'])

    if components.shape[0] != n_dims or coords.shape[1] != n_dims:
        raise ValueError(
            f"Inconsistent PPCA outputs: coords {coords.shape}, components {components.shape}, n_dims {n_dims}."
        )

    return coords_df, coord_col, coords, components, explained_variance_ratio, n_dims


def fit_picard_rotation(coords, *, ortho=False, extended=True, max_iter=500, tol=1e-7, random_state=0):
    K_dim = coords.shape[1]
    print(f"Fitting Picard ICA on PPCA latent space (n={coords.shape[0]}, K={K_dim}, ortho={ortho}, extended={extended})...")

    K_white, W_unmix, sources = picard.picard(
        coords.T,
        ortho=ortho,
        extended=extended,
        whiten=True,
        centering=True,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        verbose=False,
    )
    # picard centered with the column mean of coords.T, i.e. the per-latent-dim mean.
    latent_mean = coords.mean(axis=0)
    # sources Y = W_unmix @ K_white @ (coords.T - latent_mean[:, None]).
    # Mixing matrix in latent space: A such that (coords - mean) = A @ sources.T.
    unmix_full = W_unmix @ K_white  # (K, K)
    mixing_full = np.linalg.inv(unmix_full)  # (K, K)
    return latent_mean, K_white, W_unmix, unmix_full, mixing_full, sources


def compose_ica_with_ppca(mixing_full, sources, components):
    # components: (K, D) PPCA loading rows (each row is one PC's pattern in feature space).
    # ic_components[i] in feature space = sum_j mixing_full[j, i] * components[j, :].
    # Equivalently: ic_components = mixing_full.T @ components, shape (K, D).
    ic_components = mixing_full.T @ components
    ic_coords = sources.T  # (N, K)
    return ic_components, ic_coords


def sort_and_orient(ic_components, ic_coords, unmix_full, mixing_full):
    # IC signal energy in feature space: var(coord_i) * ||loadings_i||^2.
    energies = ic_coords.var(axis=0) * np.linalg.norm(ic_components, axis=1) ** 2
    order = np.argsort(-energies)

    ic_components = ic_components[order]
    ic_coords = ic_coords[:, order]
    energies = energies[order]
    # Re-order rotation matrices to match the new IC order.
    unmix_full = unmix_full[order, :]
    mixing_full = mixing_full[:, order]

    # Sign convention: each IC's largest-magnitude loading should be positive.
    signs = np.sign(ic_components[np.arange(ic_components.shape[0]), np.argmax(np.abs(ic_components), axis=1)])
    signs[signs == 0] = 1.0
    ic_components = ic_components * signs[:, None]
    ic_coords = ic_coords * signs[None, :]
    unmix_full = unmix_full * signs[:, None]
    mixing_full = mixing_full * signs[None, :]

    return ic_components, ic_coords, energies, unmix_full, mixing_full


def save_pica_rotation(path, latent_mean, K_white, W_unmix, unmix_full, mixing_full, energies, order):
    np.savez(
        path,
        latent_mean=latent_mean,
        K_white=K_white,
        W_unmix=W_unmix,
        unmix_full=unmix_full,
        mixing_full=mixing_full,
        ic_signal_energy=energies,
        order=order,
    )


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    trend_path = cfg.trend_path

    coords_df, coord_col, coords, components, ppca_evr, n_dims = load_ppca_outputs(trend_path)

    pica_cfg = getattr(cfg, 'pica', {}) if hasattr(cfg, 'pica') else {}
    picard_cfg = getattr(pica_cfg, 'picard', {}) if hasattr(pica_cfg, 'picard') else {}
    ortho = bool(getattr(picard_cfg, 'ortho', False))
    extended = bool(getattr(picard_cfg, 'extended', True))
    max_iter = int(getattr(picard_cfg, 'max_iter', 500))
    tol = float(getattr(picard_cfg, 'tol', 1e-7))
    random_state = int(getattr(picard_cfg, 'random_state', 0))

    latent_mean, K_white, W_unmix, unmix_full, mixing_full, sources = fit_picard_rotation(
        coords,
        ortho=ortho,
        extended=extended,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )

    ic_components, ic_coords = compose_ica_with_ppca(mixing_full, sources, components)
    ic_components, ic_coords, energies, unmix_full, mixing_full = sort_and_orient(
        ic_components, ic_coords, unmix_full, mixing_full
    )

    # Re-order is now baked in; record identity order for the rotation file.
    order = np.arange(n_dims)
    save_pica_rotation(
        os.path.join(trend_path, PICA_ROTATION_FILENAME),
        latent_mean, K_white, W_unmix, unmix_full, mixing_full, energies, order,
    )

    out_coord_col = f'coord_{n_dims}d'
    pica_coords_df = coords_df.select(['createtime', 'filter_value'])\
        .with_columns(pl.Series(name=out_coord_col, values=ic_coords))
    pica_coords_df.write_parquet(os.path.join(trend_path, 'pica_coords.parquet.zstd'), compression='zstd')

    # PICA's per-IC "explained variance ratio" is the fraction of latent-signal energy each IC captures.
    # This is rotation-invariant in total but informative for ranking / interpretation.
    ic_explained_variance_ratio = (energies / energies.sum()).tolist()

    pica_metadata_df = pl.from_dicts([{
        'n_dims': n_dims,
        'explained_variance_ratio': ic_explained_variance_ratio,
        'components': ic_components.tolist(),
        'ic_signal_energy': energies.tolist(),
        'ppca_explained_variance_ratio': ppca_evr.tolist(),
    }])
    pica_metadata_df.write_parquet(os.path.join(trend_path, 'pica_metadata.parquet.zstd'), compression='zstd')

    print(f"Wrote pica_coords ({len(pica_coords_df)} rows, K={n_dims}) and pica_metadata to {trend_path}")
    print(f"IC signal energy ratios: {[f'{r:.3f}' for r in ic_explained_variance_ratio]}")


if __name__ == '__main__':
    main()
