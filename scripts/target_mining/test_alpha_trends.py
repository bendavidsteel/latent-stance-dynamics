
import os
import re

import dotenv
import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

from stancemining.estimate import (
    _get_time_series_data,
    bootstrap_bayesian_krr_gpu_batched,
    bootstrap_bayesian_krr_numba,
    GPU_AVAILABLE,
)


def loo_cv_alpha(K_train, y, alphas):
    """Compute LOO-CV MSE for all alphas using eigendecomposition.

    Decomposes K = Q Λ Qᵀ once, then for each alpha:
        (K + αI)⁻¹ = Q diag(1/(λ_i + α)) Qᵀ
    LOO residual for sample i: w_i / H_ii
    where w = (K + αI)⁻¹ y and H_ii = [(K + αI)⁻¹]_ii.
    """
    alphas = np.asarray(alphas)
    eigenvalues, Q = np.linalg.eigh(K_train)
    # Q_y[j] = Qᵀ @ y for component j
    Q_y = Q.T @ y  # (n,)
    # Q_sq[i, j] = Q[i, j]² — needed for diagonal of H
    Q_sq = Q ** 2  # (n, n)

    # For each alpha, compute inv_eig[j] = 1 / (λ_j + α)
    # Shape: (n_alphas, n)
    inv_eig = 1.0 / (eigenvalues[np.newaxis, :] + alphas[:, np.newaxis])

    # w_i = sum_j Q[i,j] * inv_eig[j] * Q_y[j]  →  w = Q @ diag(inv_eig) @ Qᵀ y
    # H_ii = sum_j Q[i,j]² * inv_eig[j]
    # Vectorized over alphas: (n_alphas, n)
    w = (inv_eig * Q_y[np.newaxis, :]) @ Q.T  # (n_alphas, n)
    H_diag = Q_sq @ inv_eig.T  # (n, n_alphas)

    loo_residuals = w / H_diag.T  # (n_alphas, n)
    scores = np.mean(loo_residuals ** 2, axis=1)  # (n_alphas,)

    best_idx = np.argmin(scores)
    return alphas[best_idx], scores.tolist()


def run_bkrr(stance, timestamps, test_x, lengthscale, alpha, n_bootstrap=100):
    """Run bootstrap BKRR and return mean, lower, upper."""
    if GPU_AVAILABLE:
        return bootstrap_bayesian_krr_gpu_batched(
            stance, timestamps, test_x,
            lengthscale=lengthscale, alpha=alpha, n_bootstrap=n_bootstrap
        )
    else:
        return bootstrap_bayesian_krr_numba(
            stance, timestamps, test_x,
            lengthscale=lengthscale, alpha=alpha, n_bootstrap=n_bootstrap
        )


def compute_loo_optimal_alpha(timestamps, stance, lengthscale, alpha_range):
    """Compute LOO-CV optimal alpha from the training data kernel matrix."""
    gamma = 1.0 / (2 * lengthscale ** 2)
    train_x = timestamps.reshape(-1, 1)
    diff = train_x - train_x.T
    K_train = np.exp(-gamma * diff ** 2)
    best_alpha, scores = loo_cv_alpha(K_train, stance, alpha_range)
    return best_alpha, scores


def plot_comparison(ax, test_dates, stance, timestamps, test_x, \
                    lengthscale, alpha, label, color):
    """Run BKRR for a given alpha and plot on the provided axis."""
    mean, lower, upper = run_bkrr(stance, timestamps, test_x, lengthscale, alpha)
    ax.plot(test_dates, mean, label=label, color=color)
    ax.fill_between(test_dates, lower, upper, alpha=0.1, color=color)
    return mean


def get_target_volumes(trend_dir):
    """Get total volume per target from pre-computed trend files."""
    trend_files = [
        os.path.join(trend_dir, f)
        for f in os.listdir(trend_dir)
        if f.endswith('_trends.parquet.zstd')
    ]
    dfs = []
    for f in tqdm(trend_files, desc="Loading trend files"):
        df = pl.read_parquet(f, columns=['target', 'volume'])
        dfs.append(df.group_by('target').agg(pl.col('volume').sum()))
    return pl.concat(dfs).group_by('target').agg(pl.col('volume').sum())\
        .sort('volume', descending=True)


def load_data_for_targets(base_stance_path, target_names):
    """Load raw stance data only for the selected targets, file by file to limit memory."""
    file_paths = sorted([
        os.path.join(base_stance_path, f)
        for f in os.listdir(base_stance_path)
        if re.search(r'\d{4}_\d{1,2}_doc_targets_with_stance.parquet.zstd', f)
    ])
    target_set = set(target_names)
    chunks = []
    for f in tqdm(file_paths, desc="Loading stance files"):
        chunk = pl.read_parquet(f, columns=['createtime', 'Targets', 'Stances', 'seed'])\
            .explode(['Targets', 'Stances'])\
            .rename({'Targets': 'Target', 'Stances': 'Stance'})\
            .filter(pl.col('Target').is_in(target_set))
        if len(chunk) > 0:
            chunk = chunk.with_columns(pl.col('seed').struct.unnest())\
                .select(['createtime', 'Target', 'Stance', 'SeedName', 'PlatformHandleID'])
            chunks.append(chunk)
    return pl.concat(chunks)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    print("Getting target volumes from trend files...")
    target_volumes = get_target_volumes(cfg.trend_path)
    target_volumes = target_volumes.filter(pl.col('volume') >= cfg.min_target_volume)

    # Pick targets at different volume levels
    n_targets = len(target_volumes)
    indices = [0, n_targets // 4, n_targets // 2, 3 * n_targets // 4, n_targets - 1]
    selected_targets = target_volumes[indices]

    print(f"Selected {len(selected_targets)} targets across volume range:")
    for row in selected_targets.to_dicts():
        print(f"  {row['target']}: {row['volume']} total volume")

    target_names = selected_targets['target'].to_list()
    print("Loading raw data for selected targets...")
    df = load_data_for_targets(cfg.base_stance_path, target_names)
    print(f"Loaded {len(df)} records")

    lengthscale_loc = 2.0
    lengthscale_scale = 0.1
    lengthscale = np.exp(lengthscale_loc - lengthscale_scale**2)

    fixed_alphas = [0.01, 0.1, 1.0, 10.0]
    alpha_range = np.logspace(-3, 2, 50)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    fig_dir = os.path.join('figs', 'trend_comparison')
    os.makedirs(fig_dir, exist_ok=True)

    for target_info in selected_targets.to_dicts():
        print(f"\nProcessing target: {target_info['target']} with volume {target_info['volume']}")
        target_name = target_info['target']
        target_slug = target_name.lower().replace(' ', '_').replace('-', '_')\
            .replace("'", '').replace('"', '').replace('/', '_')

        target_df = df.filter(pl.col('Target') == target_name)

        filter_counts = target_df.group_by(cfg.filter_column).len().sort('len', descending=True)
        if len(filter_counts) == 0:
            continue

        # Pick top, middle, and low-volume filter values
        fv_indices = [0]
        if len(filter_counts) > 2:
            fv_indices.append(len(filter_counts) // 2)
        if len(filter_counts) > 4:
            fv_indices.append(len(filter_counts) - 1)

        for fv_idx in fv_indices:
            print(f"  Filter value: {filter_counts[cfg.filter_column][fv_idx]} with {filter_counts['len'][fv_idx]} observations")
            filter_value = filter_counts[cfg.filter_column][fv_idx]
            n_obs = filter_counts['len'][fv_idx]

            if n_obs < cfg.min_filter_count:
                continue

            filtered_df = target_df.filter(pl.col(cfg.filter_column) == filter_value)

            timestamps, stance, classifier_ids, test_x, trend_df = \
                _get_time_series_data(filtered_df, cfg.stance_target_type, 'createtime', '1mo')

            if len(timestamps) < 3:
                continue

            test_dates = trend_df['createtime'].to_numpy()

            # Compute LOO-CV optimal alpha
            loo_alpha, loo_scores = compute_loo_optimal_alpha(
                timestamps, stance, lengthscale, alpha_range
            )

            # --- Figure 1: Trend comparison across alphas ---
            n_methods = len(fixed_alphas) + 1
            fig, axes = plt.subplots(n_methods, 1, figsize=(12, 3 * n_methods), sharex=True)

            raw_dates = filtered_df.sort('createtime')['createtime'].to_numpy()

            for i, alpha in enumerate(fixed_alphas):
                ax = axes[i]
                ax.scatter(raw_dates, stance, marker='x', alpha=0.3, color='gray', s=10)
                plot_comparison(
                    ax, test_dates, stance, timestamps, test_x,
                    lengthscale, alpha, f'α={alpha}', colors[i]
                )
                ax.set_ylabel('Stance')
                ax.set_ylim([-1.2, 1.2])
                ax.legend(loc='upper right')

            # LOO-CV optimal
            ax = axes[-1]
            ax.scatter(raw_dates, stance, marker='x', alpha=0.3, color='gray', s=10)
            plot_comparison(
                ax, test_dates, stance, timestamps, test_x,
                lengthscale, loo_alpha, f'α={loo_alpha:.4f} (LOO-CV)', colors[-1]
            )
            ax.set_ylabel('Stance')
            ax.set_ylim([-1.2, 1.2])
            ax.legend(loc='upper right')
            ax.set_xlabel('Date')

            fig.suptitle(
                f'{target_name} | {cfg.filter_column}={filter_value} | n={n_obs}',
                fontsize=12
            )
            fig.tight_layout()
            fig_path = os.path.join(fig_dir, f'{target_slug}_{filter_value}_trends.png')
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"Saved {fig_path}")

            # --- Figure 2: LOO-CV score vs alpha ---
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.semilogx(alpha_range, loo_scores, 'k-')
            ax2.axvline(loo_alpha, color='red', linestyle='--', label=f'optimal α={loo_alpha:.4f}')
            for alpha in fixed_alphas:
                ax2.axvline(alpha, color='gray', linestyle=':', alpha=0.5)
            ax2.set_xlabel('α')
            ax2.set_ylabel('LOO-CV MSE')
            ax2.set_title(
                f'LOO-CV: {target_name} | {cfg.filter_column}={filter_value} | n={n_obs}'
            )
            ax2.legend()
            fig2.tight_layout()
            fig2_path = os.path.join(fig_dir, f'{target_slug}_{filter_value}_loo_cv.png')
            fig2.savefig(fig2_path, dpi=150)
            plt.close(fig2)
            print(f"Saved {fig2_path}")


if __name__ == '__main__':
    dotenv.load_dotenv()
    main()
