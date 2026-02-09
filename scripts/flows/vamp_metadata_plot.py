
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    trend_path = cfg.trend_path
    vamp_metadata_df = pl.read_parquet(os.path.join(trend_path, 'vamp_metadata.parquet.zstd'))

    os.makedirs('figs', exist_ok=True)

    # Extract data for plotting
    lagtimes = vamp_metadata_df['lagtime'].to_numpy()
    # Singular values are stored as nested lists due to np.newaxis in compute_vamp.py
    singular_values_list = [np.array(sv).flatten() for sv in vamp_metadata_df['singular_values'].to_list()]

    # Plot 1: Singular value spectra for each lag time
    fig, ax = plt.subplots(figsize=(10, 6))
    for lagtime, sv in zip(lagtimes, singular_values_list):
        ax.plot(range(1, len(sv) + 1), sv, marker='o', label=f'lag={lagtime}')
    ax.set_xlabel('Component Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('VAMP Singular Value Spectra by Lag Time')
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join('figs', 'vamp_singular_value_spectra.png'), dpi=150)
    plt.close(fig)

    # Plot 2: Sum of squared singular values (VAMP-2 score proxy) vs lag time
    vamp2_scores = [np.sum(sv**2) for sv in singular_values_list]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(lagtimes, vamp2_scores, marker='o', linewidth=2)
    ax.set_xlabel('Lag Time')
    ax.set_ylabel('Sum of Squared Singular Values')
    ax.set_title('VAMP-2 Score vs Lag Time')
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join('figs', 'vamp2_score_vs_lagtime.png'), dpi=150)
    plt.close(fig)

    # Plot 3: Number of components retained (above var_cutoff threshold) vs lag time
    var_cutoff = vamp_metadata_df['var_cutoff'][0]
    n_components = []
    for sv in singular_values_list:
        cumvar = np.cumsum(sv**2) / np.sum(sv**2)
        n_comp = np.searchsorted(cumvar, var_cutoff) + 1
        n_components.append(min(n_comp, len(sv)))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(lagtimes, n_components, marker='o', linewidth=2)
    ax.set_xlabel('Lag Time')
    ax.set_ylabel(f'Number of Components (var_cutoff={var_cutoff})')
    ax.set_title('VAMP Dimensionality vs Lag Time')
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join('figs', 'vamp_n_components_vs_lagtime.png'), dpi=150)
    plt.close(fig)

    # Plot 4: Top-k singular values vs lag time
    k = 5
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(k):
        values = [np.array(sv)[i] if len(sv) > i else np.nan for sv in singular_values_list]
        ax.plot(lagtimes, values, marker='o', label=f'SV {i+1}')
    ax.set_xlabel('Lag Time')
    ax.set_ylabel('Singular Value')
    ax.set_title(f'Top {k} VAMP Singular Values vs Lag Time')
    ax.set_xscale('log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join('figs', 'vamp_top_singular_values_vs_lagtime.png'), dpi=150)
    plt.close(fig)

    # Plot 5: Implied timescales from singular values
    # For VAMP, implied timescales are: -lagtime / log(singular_value)
    fig, ax = plt.subplots(figsize=(10, 6))
    for lagtime, singular_values in zip(lagtimes, singular_values_list):
        sv = np.array(singular_values)
        # Filter out singular values >= 1 (would give negative or infinite timescales)
        valid_mask = (sv > 0) & (sv < 1)
        if np.any(valid_mask):
            timescales = -lagtime / np.log(sv[valid_mask])
            ax.scatter([lagtime] * len(timescales), timescales, alpha=0.7, label=f'lag={lagtime}')
    ax.set_xlabel('Lag Time')
    ax.set_ylabel('Implied Timescale')
    ax.set_title('VAMP Implied Timescales')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join('figs', 'vamp_implied_timescales.png'), dpi=150)
    plt.close(fig)

    print(f"Saved plots to figs/")

if __name__ == '__main__':
    main()
