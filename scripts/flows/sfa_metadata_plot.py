
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    trend_path = cfg.trend_path
    sfa_metadata_df = pl.read_parquet(os.path.join(trend_path, 'sfa_metadata.parquet.zstd'))

    os.makedirs('figs', exist_ok=True)

    # Extract data for plotting
    n_components_list = sfa_metadata_df['n_components'].to_numpy()
    delta_values_list = [np.array(dv).flatten() for dv in sfa_metadata_df['delta_values'].to_list()]

    # Plot 1: Delta values spectrum (slowness of each component)
    fig, ax = plt.subplots(figsize=(10, 6))
    for n_comp, delta_values in zip(n_components_list, delta_values_list):
        ax.plot(range(1, len(delta_values) + 1), delta_values, marker='o', label=f'n_comp={n_comp}')
    ax.set_xlabel('Component Index')
    ax.set_ylabel('Delta Value (Mean Squared Derivative)')
    ax.set_title('SFA Delta Values Spectrum (Lower = Slower)')
    if len(n_components_list) > 1:
        ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join('figs', 'sfa_delta_values_spectrum.png'), dpi=150)
    plt.close(fig)

    # Plot 2: Cumulative slowness (sum of inverse delta values)
    # Lower delta = slower feature, so 1/delta represents "slowness contribution"
    fig, ax = plt.subplots(figsize=(10, 6))
    for n_comp, delta_values in zip(n_components_list, delta_values_list):
        # Inverse delta gives slowness weight
        slowness = 1.0 / delta_values
        cumulative_slowness = np.cumsum(slowness) / np.sum(slowness)
        ax.plot(range(1, len(delta_values) + 1), cumulative_slowness, marker='o', label=f'n_comp={n_comp}')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Slowness (Normalized)')
    ax.set_title('SFA Cumulative Slowness by Component')
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, label='90% threshold')
    if len(n_components_list) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join('figs', 'sfa_cumulative_slowness.png'), dpi=150)
    plt.close(fig)

    # Plot 3: Implied timescales from delta values
    # For SFA, implied timescale ~ 1/sqrt(delta_value) (since delta = mean((dx/dt)^2))
    fig, ax = plt.subplots(figsize=(10, 6))
    for n_comp, delta_values in zip(n_components_list, delta_values_list):
        timescales = 1.0 / np.sqrt(delta_values)
        ax.plot(range(1, len(delta_values) + 1), timescales, marker='o', label=f'n_comp={n_comp}')
    ax.set_xlabel('Component Index')
    ax.set_ylabel('Implied Timescale (1/sqrt(delta))')
    ax.set_title('SFA Implied Timescales')
    ax.set_yscale('log')
    if len(n_components_list) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join('figs', 'sfa_implied_timescales.png'), dpi=150)
    plt.close(fig)

    # Plot 4: Top-k delta values comparison (if multiple configs)
    if len(n_components_list) > 1:
        k = min(5, min(len(dv) for dv in delta_values_list))
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(k):
            values = [dv[i] if len(dv) > i else np.nan for dv in delta_values_list]
            ax.plot(n_components_list, values, marker='o', label=f'Component {i+1}')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Delta Value')
        ax.set_title(f'Top {k} SFA Delta Values vs Number of Components')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join('figs', 'sfa_top_delta_values.png'), dpi=150)
        plt.close(fig)

    # Plot 5: Number of nontrivial components
    n_nontrivial = sfa_metadata_df['n_nontrivial_components'].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(n_components_list)), n_nontrivial, tick_label=[str(n) for n in n_components_list])
    ax.set_xlabel('Requested Components')
    ax.set_ylabel('Nontrivial Components')
    ax.set_title('SFA Nontrivial Components')
    fig.tight_layout()
    fig.savefig(os.path.join('figs', 'sfa_nontrivial_components.png'), dpi=150)
    plt.close(fig)

    print(f"Saved plots to figs/")

if __name__ == '__main__':
    main()
