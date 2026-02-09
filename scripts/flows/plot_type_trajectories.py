import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pca_clusters import load_text_df
from pca_density import get_top_component_features, format_pca_axis_label
from plot_nn_potential import load_dimension_labels, show_x_dim_labels, show_y_dim_labels

PLATFORMS = ['twitter', 'tiktok', 'instagram', 'bluesky']
PLATFORM_PRETTY = {
    'twitter': 'Twitter',
    'tiktok': 'TikTok',
    'instagram': 'Instagram',
    'bluesky': 'Bluesky'
}

# Canadian province codes to full names
CANADIAN_PROVINCES = {
    'AB': 'Alberta',
    'BC': 'British Columbia',
    'MB': 'Manitoba',
    'NB': 'New Brunswick',
    'NL': 'Newfoundland and Labrador',
    'NS': 'Nova Scotia',
    'NT': 'Northwest Territories',
    'NU': 'Nunavut',
    'ON': 'Ontario',
    'PE': 'Prince Edward Island',
    'QC': 'Quebec',
    'SK': 'Saskatchewan',
    'YT': 'Yukon',
    # Also handle full names that might already be present
    'Alberta': 'Alberta',
    'British Columbia': 'British Columbia',
    'Manitoba': 'Manitoba',
    'New Brunswick': 'New Brunswick',
    'Newfoundland and Labrador': 'Newfoundland and Labrador',
    'Nova Scotia': 'Nova Scotia',
    'Northwest Territories': 'Northwest Territories',
    'Nunavut': 'Nunavut',
    'Ontario': 'Ontario',
    'Prince Edward Island': 'Prince Edward Island',
    'Quebec': 'Quebec',
    'Saskatchewan': 'Saskatchewan',
    'Yukon': 'Yukon',
}

# US state codes
US_STATES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    'DC', 'PR', 'VI', 'GU', 'AS', 'MP',
    # Full state names
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
    'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
    'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
    'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
    'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
    'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
    'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
    'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
    'West Virginia', 'Wisconsin', 'Wyoming', 'District of Columbia',
    'Puerto Rico', 'US', 'USA', 'United States',
}


def map_province(province):
    """Map province values to standardized categories."""
    if province is None:
        return 'Other'
    if province in CANADIAN_PROVINCES:
        return CANADIAN_PROVINCES[province]
    if province in US_STATES:
        return 'US'
    return 'Other'


def plot_fig(
        ax, pca_df, color_col, components, feature_names, dimension_labels=None, window_size=100,
        show_x_axis_labels=True, show_y_axis_labels=True, show_x_tick_labels=True, show_y_tick_labels=True,
        show_legend=True, show_title=True, xrange=None, yrange=None
    ):
    color_values = pca_df[color_col].unique().sort().to_list()
    color_map = {val: plt.cm.tab10(i) for i, val in enumerate(color_values)}

    # Track which labels we've added to avoid duplicate legend entries
    added_labels = set()

    for user_target_df in pca_df.partition_by('filter_value'):
        color_val = user_target_df[color_col][0]
        label = color_val if color_val not in added_labels else None
        added_labels.add(color_val)

        ax.plot(
            user_target_df['coord'].arr.get(0).rolling_mean(window_size),
            user_target_df['coord'].arr.get(1).rolling_mean(window_size),
            alpha=0.3,
            color=color_map[color_val],
            label=label
        )

    # Set axis labels with PCA feature information
    top_features = get_top_component_features(components, feature_names, n_features=3)
    if show_x_axis_labels:
        ax.set_xlabel(format_pca_axis_label(1, top_features['PC1']))
    if show_y_axis_labels:
        ax.set_ylabel(format_pca_axis_label(2, top_features['PC2']))

    # Add dimension tick labels if available
    if dimension_labels is not None:
        if show_x_tick_labels:
            show_x_dim_labels(ax, dimension_labels, dim=0)
        else:
            ax.set_xticks([])
        if show_y_tick_labels:
            show_y_dim_labels(ax, dimension_labels, dim=1)
        else:
            ax.set_yticks([])

    if xrange is not None:
        ax.set_xlim(xrange)
    if yrange is not None:
        ax.set_ylim(yrange)

    ax.set_aspect('equal')

    if show_legend:
        legend = ax.legend(loc='best')
        for lh in legend.legend_handles:
            lh.set_alpha(1)

    if show_title:
        ax.set_title(f'{color_col}')

    return ax


def plot_platform_fig(
        ax, pca_df, components, feature_names, dimension_labels=None, window_size=100,
        show_x_axis_labels=True, show_y_axis_labels=True, show_x_tick_labels=True, show_y_tick_labels=True,
        show_legend=True, show_title=True, xrange=None, yrange=None
    ):
    color_map = {platform: plt.cm.tab10(i) for i, platform in enumerate(PLATFORMS)}

    # Track which labels we've added to avoid duplicate legend entries
    added_labels = set()

    for platform in PLATFORMS:
        platform_df = pca_df.filter(
            pl.col('filter_value').cast(pl.String) \
                .str.to_lowercase() \
                .str.contains(f'-{platform}-')
        )

        for user_target_df in platform_df.partition_by('filter_value'):
            label = PLATFORM_PRETTY[platform] if platform not in added_labels else None
            added_labels.add(platform)

            ax.plot(
                user_target_df['coord'].arr.get(0).rolling_mean(window_size),
                user_target_df['coord'].arr.get(1).rolling_mean(window_size),
                alpha=0.3,
                color=color_map[platform],
                label=label
            )

    # Set axis labels with PCA feature information
    top_features = get_top_component_features(components, feature_names, n_features=3)
    if show_x_axis_labels:
        ax.set_xlabel(format_pca_axis_label(1, top_features['PC1']))
    if show_y_axis_labels:
        ax.set_ylabel(format_pca_axis_label(2, top_features['PC2']))

    # Add dimension tick labels if available
    if dimension_labels is not None:
        if show_x_tick_labels:
            show_x_dim_labels(ax, dimension_labels, dim=0)
        else:
            ax.set_xticks([])
        if show_y_tick_labels:
            show_y_dim_labels(ax, dimension_labels, dim=1)
        else:
            ax.set_yticks([])

    if xrange is not None:
        ax.set_xlim(xrange)
    if yrange is not None:
        ax.set_ylim(yrange)

    ax.set_aspect('equal')

    if show_legend:
        legend = ax.legend(loc='best')
        for lh in legend.legend_handles:
            lh.set_alpha(1)

    if show_title:
        ax.set_title('Platform')

    return ax


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    print("Loading data...")

    seed_name_pca_df = pl.read_parquet(os.path.join('data', 'stance_targets', 'noun_phrase_bkrr_trends', 'pca_coords.parquet.zstd'))
    platform_handle_pca_df = pl.read_parquet(os.path.join('data', 'stance_targets', 'platform_handle_noun_phrase_bkrr_trends', 'pca_coords.parquet.zstd'))
    dir_name = "noun_phrase_bkrr_trends/all"

    pca_path = os.path.join(cfg.trend_path, 'pca_coords.parquet.zstd')
    pca_head_df = pl.read_parquet(pca_path, n_rows=1)

    # Load PCA components and feature names
    component_df = pl.read_parquet(os.path.join(cfg.trend_path, 'pca_metadata.parquet.zstd'))
    components = np.stack(component_df.filter(pl.col('n_dims') == 21)['components'][0].to_numpy())
    feature_names = [col for col in pca_head_df.columns if col not in ['createtime', 'filter_value', 'coord_21d']]

    # Load dimension labels if available
    dimension_labels_path = os.path.join('data', 'stance_targets', 'noun_phrase_bkrr_trends', 'pca_dimension_labels.json')
    dimension_labels = load_dimension_labels(dimension_labels_path)

    # Load platform-specific dimension labels
    platform_dimension_labels_path = os.path.join('data', 'stance_targets', 'platform_handle_noun_phrase_bkrr_trends', 'pca_dimension_labels.json')
    platform_dimension_labels = load_dimension_labels(platform_dimension_labels_path)

    # Load seed metadata
    text_df = load_text_df(cfg, columns=['seed'])
    seed_df = text_df.select([
        pl.col('seed').struct.field('SeedName'),
        pl.col('seed').struct.field('PlatformHandleID'),
        pl.col('seed').struct.field('MainType'),
        pl.col('seed').struct.field('Party'),
        pl.col('seed').struct.field('Province')
    ]).unique('SeedName')

    # Prepare seed_name_pca_df
    seed_name_pca_df = seed_name_pca_df \
        .select(['createtime', 'filter_value', 'coord_21d']) \
        .rename({'coord_21d': 'coord'}) \
        .with_columns(pl.col('filter_value').cast(pl.String)) \
        .join(seed_df, left_on='filter_value', right_on='SeedName', how='inner') \
        .with_columns(
            pl.col('Province').map_elements(map_province, return_dtype=pl.String).alias('Province')
        )

    # Prepare platform_handle_pca_df
    platform_handle_pca_df = platform_handle_pca_df \
        .select(['createtime', 'filter_value', 'coord_21d']) \
        .rename({'coord_21d': 'coord'})

    # Calculate shared axis ranges
    all_coords = np.vstack([
        seed_name_pca_df['coord'].to_numpy(),
        platform_handle_pca_df['coord'].to_numpy()
    ])
    xrange = (np.percentile(all_coords[:, 0], 0.5), np.percentile(all_coords[:, 0], 99.5))
    yrange = (np.percentile(all_coords[:, 1], 0.5), np.percentile(all_coords[:, 1], 99.5))

    os.makedirs(f"./figs/{dir_name}", exist_ok=True)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    # Plot order: Party (top-left), Platform (top-right), Province (bottom-left), MainType (bottom-right)
    plot_configs = [
        {'col': 'Party', 'idx': 0},
        {'col': 'Province', 'idx': 2},
        {'col': 'MainType', 'idx': 3},
    ]

    window_size = 100

    for config in plot_configs:
        idx = config['idx']
        col = config['col']
        plot_fig(
            axes[idx], seed_name_pca_df, color_col=col,
            components=components, feature_names=feature_names,
            dimension_labels=dimension_labels, window_size=window_size,
            show_x_axis_labels=(idx in [2, 3]),
            show_y_axis_labels=(idx in [0, 2]),
            show_x_tick_labels=(idx in [2, 3]),
            show_y_tick_labels=(idx in [0, 2]),
            show_legend=True,
            show_title=True,
            xrange=xrange,
            yrange=yrange
        )

    # Platform plot (top-right) - show its own dimension labels since they differ from other axes
    plot_platform_fig(
        axes[1], platform_handle_pca_df,
        components=components, feature_names=feature_names,
        dimension_labels=platform_dimension_labels, window_size=window_size,
        show_x_axis_labels=False,
        show_y_axis_labels=False,
        show_x_tick_labels=True,
        show_y_tick_labels=True,
        show_legend=True,
        show_title=True,
        xrange=xrange,
        yrange=yrange
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        top=0.95,
        bottom=0.12,
        wspace=0.4,
        hspace=0.15
    )

    fig.savefig(f"./figs/{dir_name}/trajectories_combined.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"Saved combined plot to ./figs/{dir_name}/trajectories_combined.png")


if __name__ == '__main__':
    main()
