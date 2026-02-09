import datetime
import json
import os

import hydra
import jax
import jax.numpy as jnp
import matplotlib.animation
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np
import polars as pl
from tqdm import tqdm

from plnn.models import DeepTimePhiPLNN
from plnn.pl.plot_plnn import compute_grad_phi

from nn_potential import INITIAL_DATE, UNIT_DAYS
from pca_density import create_kde_background, get_top_component_features, format_pca_axis_label

def t_to_datetime(t):
    return INITIAL_DATE + datetime.timedelta(days=t * UNIT_DAYS)

def show_x_dim_labels(ax, dimension_labels, dim):
    # Get dimension 0 (x-axis) labels
    dim0 = dimension_labels[str(dim)]
    x_tick_labels = []
    x_tick_positions = []

    if 'very_negative' in dim0:
        x_tick_labels.append(dim0['very_negative'])
        x_tick_positions.append(dim0['v_negative_threshold'])
    if 'negative' in dim0:
        x_tick_labels.append(dim0['negative'])
        x_tick_positions.append(dim0['negative_threshold'])
    if 'neutral' in dim0:
        x_tick_labels.append(dim0['neutral'])
        x_tick_positions.append(dim0['mean'])
    if 'positive' in dim0:
        x_tick_labels.append(dim0['positive'])
        x_tick_positions.append(dim0['positive_threshold'])
    if 'very_positive' in dim0:
        x_tick_labels.append(dim0['very_positive'])
        x_tick_positions.append(dim0['v_positive_threshold'])

    if x_tick_positions:
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, fontsize=8, rotation=15, ha='right')

def show_y_dim_labels(ax, dimension_labels, dim):
    # Get dimension 1 (y-axis) labels
    dim1 = dimension_labels[str(dim)]
    y_tick_labels = []
    y_tick_positions = []

    if 'very_negative' in dim1:
        y_tick_labels.append(dim1['very_negative'])
        y_tick_positions.append(dim1['v_negative_threshold'])
    if 'negative' in dim1:
        y_tick_labels.append(dim1['negative'])
        y_tick_positions.append(dim1['negative_threshold'])
    if 'neutral' in dim1:
        y_tick_labels.append(dim1['neutral'])
        y_tick_positions.append(dim1['mean'])
    if 'positive' in dim1:
        y_tick_labels.append(dim1['positive'])
        y_tick_positions.append(dim1['positive_threshold'])
    if 'very_positive' in dim1:
        y_tick_labels.append(dim1['very_positive'])
        y_tick_positions.append(dim1['v_positive_threshold'])

    if y_tick_positions:
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(y_tick_labels, fontsize=8)

def load_dimension_labels(dimension_labels_path):
    """Load dimension labels from JSON file created by pca_dimensions.py"""
    if os.path.exists(dimension_labels_path):
        with open(dimension_labels_path, 'r') as f:
            return json.load(f)
    return None


def setup_x_axis_labels(ax, components, feature_names, dim_1=0):
    """Setup axis labels with PCA feature information and optional dimension descriptions"""
    top_features = get_top_component_features(components, feature_names, n_features=3)
    x_label = format_pca_axis_label(1, top_features[f'PC{dim_1 + 1}'])
    ax.set_xlabel(x_label)

def setup_y_axis_labels(ax, components, feature_names, dim_2=1):
    """Setup axis labels with PCA feature information and optional dimension descriptions"""
    top_features = get_top_component_features(components, feature_names, n_features=3)
    y_label = format_pca_axis_label(2, top_features[f'PC{dim_2 + 1}'])
    ax.set_ylabel(y_label)
    

def add_legend(ax):
    """Add legend for streamplot and uncertainty hatching"""
    legend_elements = [
        matplotlib.lines.Line2D([0], [0], color='red', alpha=0.7, linewidth=2,
               label='Flow patterns'),
        Patch(facecolor='none', edgecolor='black', hatch='/', label='Low uncertainty'),
        Patch(facecolor='none', edgecolor='black', hatch='//', label='Medium uncertainty'),
        Patch(facecolor='none', edgecolor='black', hatch='xxx', label='High uncertainty')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

def plot_flow_field(ax, model, t, xs, ys, confinement_threshold, mc_dropout, key):
    """Compute flow field from model gradient"""
    z = np.array([xs.flatten(), ys.flatten()]).T
    z = jnp.array(z, dtype=jnp.float64)
    z_norms = np.linalg.norm(np.array([xs.flatten(), ys.flatten()]).T, ord=2, axis=1, keepdims=True)

    grad_phi, grad_phi_std = compute_grad_phi(model, t, z, mc_dropout=mc_dropout, key=key)
    f = -np.array(grad_phi)

    # zero out flow outside confinement threshold
    f = np.where(z_norms > confinement_threshold, 0.0, f)

    fu, fv = f.T
    fu = fu.reshape(xs.shape)
    fv = fv.reshape(ys.shape)

    # Calculate flow magnitude and linewidth
    flow_magnitude = np.sqrt(fu**2 + fv**2)
    lw = 5 * flow_magnitude / flow_magnitude.max()

    # Create streamplot
    ax.streamplot(xs, ys, fu, fv,
                 density=1.5, color='red',
                 arrowsize=1.5, linewidth=lw, arrowstyle='->')

    # Add hatching based on grad_phi_std variance
    grad_phi_std_np = np.array(grad_phi_std)
    grad_phi_std_mag = np.linalg.norm(grad_phi_std_np, axis=1).reshape(xs.shape)

    # zero out variance outside confinement threshold
    grad_phi_std_mag = np.where(z_norms.reshape(xs.shape) > confinement_threshold, 0.0, grad_phi_std_mag)   

    # Create contour levels based on variance thresholds
    if np.quantile(grad_phi_std_mag, 0.25) > 0:
        variance_levels = [0, np.quantile(grad_phi_std_mag, 0.25), np.quantile(grad_phi_std_mag, 0.5),
                          np.quantile(grad_phi_std_mag, 0.75), grad_phi_std_mag.max()]
        hatch_patterns = ['', '/', '//', 'xxx']
    elif np.quantile(grad_phi_std_mag, 0.5) > 0:
        variance_levels = [0, np.quantile(grad_phi_std_mag, 0.5),
                          np.quantile(grad_phi_std_mag, 0.75), grad_phi_std_mag.max()]
        hatch_patterns = ['', '//', 'xxx']
    elif np.quantile(grad_phi_std_mag, 0.75) > 0:
        variance_levels = [0, np.quantile(grad_phi_std_mag, 0.75), grad_phi_std_mag.max()]
        hatch_patterns = ['', 'xxx']
    else:
        return fu, fv, grad_phi_std  # No variance to plot

    # Overlay hatched regions
    ax.contourf(xs, ys, grad_phi_std_mag,
               levels=variance_levels,
               hatches=hatch_patterns,
               colors='none',  # transparent fill
               alpha=0.0)

    return fu, fv, grad_phi_std

def add_colorbar(contours, ax, fig=None):
    if fig is None:
        fig = plt.gcf()
    cbar = fig.colorbar(contours, ax=ax, shrink=0.3, format='%.1e', aspect=40)
    cbar.set_label('Log Stance State Density', rotation=270, labelpad=15)

def animate_density_streamplot(
        fig_path,
        model,
        target_df,
        components,
        coords,
        feature_names,
        t_range,
        spatial_res=100,
        temporal_res=100,
        temporal_stride=0.1,
        xrange=None,
        yrange=None,
        mc_dropout=None,
        key=None,
        t_to_datetime=None,
        confinement_threshold=None,
        dimension_labels=None
    ):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Create KDE background
    # filter into t_range
    ts = np.linspace(t_range[0], t_range[1] - temporal_stride, temporal_res)
    filtered_df = filter_df_to_time_range(target_df, [ts[0], ts[0] + temporal_stride])

    x_min, x_max = target_df['coord_21d'].to_numpy()[:,0].min(), target_df['coord_21d'].to_numpy()[:,0].max()
    y_min, y_max = target_df['coord_21d'].to_numpy()[:,1].min(), target_df['coord_21d'].to_numpy()[:,1].max()

    coords = filtered_df['coord_21d'].to_numpy()
    
    contours, (grid_x, grid_y), density = create_kde_background(coords, ax, alpha=0.7, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    
    # Add colorbar for KDE density
    if contours is not None:
        add_colorbar(contours, ax)
    
    # Calculate flow field for streamplot
    print("Calculating flow field...")

    # Get grid
    x = np.linspace(*xrange, spatial_res)
    y = np.linspace(*yrange, spatial_res)
    xs, ys = np.meshgrid(x, y)
    t = ts[0] + (temporal_stride/2.0)

    # plot flow field
    plot_flow_field(ax, model, t, xs, ys, confinement_threshold, mc_dropout, key)


    # Setup axis labels
    setup_x_axis_labels(ax, components, feature_names, dim_1=0)
    setup_y_axis_labels(ax, components, feature_names, dim_2=1)

    show_x_dim_labels(ax, dimension_labels, dim=0)
    show_y_dim_labels(ax, dimension_labels, dim=1)

    ax.set_title(f't={t_to_datetime(t)}', fontsize=14, pad=20)

    # Add legend
    add_legend(ax)

    def update(start_t):
        ax.clear()

        filtered_df = filter_df_to_time_range(target_df, [start_t, start_t + temporal_stride])
        coords = filtered_df['coord_21d'].to_numpy()

        # Recreate KDE background
        create_kde_background(coords, ax, alpha=0.7, levels=contours._levels, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        # Calculate flow field
        t = start_t + (temporal_stride/2.0)
        plot_flow_field(ax, model, t, xs, ys, confinement_threshold, mc_dropout, key)

        # Re-add axis labels and title
        setup_x_axis_labels(ax, components, feature_names, dim_1=0)
        setup_y_axis_labels(ax, components, feature_names, dim_2=1)

        show_x_dim_labels(ax, dimension_labels, dim=0)
        show_y_dim_labels(ax, dimension_labels, dim=1)

        ax.set_title(f't={t_to_datetime(t).date()}', fontsize=14, pad=20)

        # Re-add legend
        add_legend(ax)
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        ax.set_aspect('equal')

        return ax,
    
    fig.tight_layout()
    ani = matplotlib.animation.FuncAnimation(fig, update, frames=tqdm(ts), interval=100)
    ani.save(f'{fig_path}/nn_potential_density_streamplot_animation.mp4')


def figure_density_streamplot(fig_path, model, target_df, components, feature_names, t_range, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_density_streamplot(fig, ax, model, target_df, components, feature_names, t_range, **kwargs)
    fig.tight_layout()
    fig.savefig(f'{fig_path}/nn_potential_density_streamplot.png', dpi=150, bbox_inches='tight', pad_inches=0.2)
    return fig
    
def filter_df_to_time_range(target_df, t_range):
    start_date = INITIAL_DATE + datetime.timedelta(days=t_range[0] * UNIT_DAYS)
    end_date = INITIAL_DATE + datetime.timedelta(days=t_range[1] * UNIT_DAYS)
    target_df = target_df.filter(
        (pl.col('createtime') >= start_date) &
        (pl.col('createtime') <= end_date)
    )
    return target_df

def plot_density_streamplot(
        fig,
        ax,
        model,
        target_df,
        components,
        feature_names,
        t_range,
        dim_1=0,
        dim_2=1,
        spatial_res=100,
        levels=10,
        xrange=None,
        yrange=None,
        mc_dropout=None,
        key=None,
        t_to_datetime=None,
        confinement_threshold=None,
        show_colorbar=True,
        dimension_labels=None,
        show_x_axis_labels=True,
        show_y_axis_labels=True,
        show_x_tick_labels=True,
        show_y_tick_labels=True,
        show_legend=True,
        **kwargs
    ):
    # Create KDE background
    # filter into t_range
    target_df = filter_df_to_time_range(target_df, t_range)
    coords = target_df['coord_21d'].to_numpy()[:, [dim_1, dim_2]]
    contours, _, _ = create_kde_background(coords, ax, alpha=0.7, levels=levels, x_min=xrange[0], x_max=xrange[1], y_min=yrange[0], y_max=yrange[1])

    # Add colorbar for KDE density
    if show_colorbar and contours is not None:
        add_colorbar(contours, ax, fig=fig)

    # Calculate flow field for streamplot
    print("Calculating flow field...")

    # Get grid
    x = np.linspace(*xrange, spatial_res, dtype=np.float64)
    y = np.linspace(*yrange, spatial_res, dtype=np.float64)
    xs, ys = np.meshgrid(x, y)
    t = (t_range[0] + t_range[1]) / 2.0

    # Compute flow field
    plot_flow_field(ax, model, t, xs, ys, confinement_threshold, mc_dropout, key)

    # Setup axis labels (conditionally)
    if show_x_axis_labels:
        setup_x_axis_labels(ax, components, feature_names, dim_1=dim_1)
    if show_y_axis_labels:
        setup_y_axis_labels(ax, components, feature_names, dim_2=dim_2)

    if show_x_tick_labels:
        # Still set up tick labels without axis labels
        show_x_dim_labels(ax, dimension_labels, dim=dim_1)
    else:
        ax.set_xticks([])
    if show_y_tick_labels:
        show_y_dim_labels(ax, dimension_labels, dim=dim_2)
    else:
        ax.set_yticks([])

    # Set explicit axis limits to ensure streamplot/hatching fills the axes
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_aspect('equal')

    # Add legend (conditionally)
    if show_legend:
        add_legend(ax)

    return ax, contours

def get_most_recent_state(path):
    state_paths = [os.path.join(path, f) for f in os.listdir(path)]
    most_recent_path = sorted(state_paths, key=lambda p: os.path.getmtime(p))[-1]
    return most_recent_path

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    target_path = os.path.join(cfg.trend_path, 'pca_coords.parquet.zstd')
    target_head_df = pl.read_parquet(target_path, n_rows=1)
    target_df = pl.read_parquet(target_path, columns=['createtime', 'filter_value', 'coord_21d'])
    component_df = pl.read_parquet(os.path.join(cfg.trend_path, 'pca_metadata.parquet.zstd'))
    stance_cols = [col for col in target_head_df.columns if col not in ['createtime', 'filter_value', 'coord_21d']]
   
    components = np.stack(component_df.filter(pl.col('n_dims') == 21)['components'][0].to_numpy())

    assert len(stance_cols) == components.shape[1]
    target_df = target_df.select(['createtime', 'filter_value', 'coord_21d'])
    coords = target_df['coord_21d'].to_numpy()
    x_range = (np.percentile(coords[:,0], 0.5), np.percentile(coords[:,0], 99.5))
    y_range = (np.percentile(coords[:,1], 0.5), np.percentile(coords[:,1], 99.5))

    # Load dimension labels if available
    trend_path = cfg.trend_path
    trend_name = os.path.basename(trend_path.rstrip('/'))
    dimension_labels_path = os.path.join(trend_path, 'pca_dimension_labels.json')
    dimension_labels = load_dimension_labels(dimension_labels_path)

    # with jax.default_device(jax.devices("cpu")[0]):
    seed = 42
    rng = np.random.default_rng(seed=seed)
    key = jax.random.PRNGKey(int(rng.integers(2**32)))
    key, modelkey, initkey, trainkey = jax.random.split(key, 4)

    dtype = jnp.float32

    trend_name = os.path.basename(cfg.trend_path.rstrip('/'))

    fig_path = f'./figs/{trend_name}'
    os.makedirs(fig_path, exist_ok=True)

    states_path = os.path.join('./out/', trend_name, 'dims_0_1', 'states')
    state_path = get_most_recent_state(states_path)
    model, _ = DeepTimePhiPLNN.load(state_path, dtype=dtype)

    start_date = datetime.date(2022, 1, 1)
    end_date = target_df['createtime'].max()
    trange = ( (start_date - INITIAL_DATE.date()).days / UNIT_DAYS, (end_date - INITIAL_DATE.date()).days / UNIT_DAYS )

    plot_kwargs = {
        'mc_dropout': 100,
        'key': modelkey,
        't_to_datetime': t_to_datetime,
        'xrange': x_range,
        'yrange': y_range,
        't_range': trange,
        'confinement_threshold': model.confinement_threshold,
        'dimension_labels': dimension_labels,
        'spatial_res': 50,
        'temporal_res': 100,
    }

    PLOT_ANI = False
    PLOT_TIME = False
    PLOT_PLATFORM = True
    PLOT_DIMS = False
    PLOT_SINGLE = False

    PLATFORMS = ['twitter', 'tiktok', 'instagram', 'bluesky']

    if PLOT_ANI:
        animate_density_streamplot(fig_path, model, target_df, components, coords, stance_cols, **plot_kwargs)

    if PLOT_SINGLE:
        figure_density_streamplot(fig_path, model, target_df, components, stance_cols, **plot_kwargs)
    

    if PLOT_TIME:
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        axes = axes.flatten()
        years = range(2022, 2026, 1)
        for i, y in enumerate(years):
            t_range = (datetime.datetime(y, 1, 1) - INITIAL_DATE).days / UNIT_DAYS, (datetime.datetime(y, 12, 31) - INITIAL_DATE).days / UNIT_DAYS
            plot_kwargs['t_range'] = t_range
            plot_kwargs['show_colorbar'] = False
            # Only show axis labels on bottom-left (index 2), legend on top-right (index 1)
            plot_kwargs['show_axis_labels'] = (i == 2)
            plot_kwargs['show_x_tick_labels'] = i in [2,3]
            plot_kwargs['show_y_tick_labels'] = i in [0,2]
            plot_kwargs['show_legend'] = (i == 1)
            ax, contours = plot_density_streamplot(axes[i], model, target_df, components, stance_cols, **plot_kwargs)
            if 'levels' not in plot_kwargs:
                plot_kwargs['levels'] = contours._levels
            axes[i].set_title(f'{t_to_datetime(t_range[0]).strftime("%Y")}')

        fig.subplots_adjust(
            left=0.01,
            right=0.94,
            top=0.99,
            bottom=0.1,
            wspace=0.04,
            hspace=0.08
        )

        # Manually position colorbar axis: [left, bottom, width, height]
        cax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(contours, cax=cax, format='%.1e', aspect=40)
        cbar.set_label('Log Stance State Density', rotation=270, labelpad=15)

        fig.savefig('./figs/nn_potential_density_streamplot_time_snapshots.png', bbox_inches='tight', dpi=150, pad_inches=0)

    if PLOT_PLATFORM:
        platform_pretty = {
            'twitter': 'Twitter',
            'tiktok': 'TikTok',
            'instagram': 'Instagram',
            'bluesky': 'Bluesky'
        }

        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        axes = axes.flatten()

        contours = None
        for i, platform in enumerate(PLATFORMS):
            # Load platform-specific model
            platform_states_path = os.path.join('./out/', trend_name, f'dims_0_1_{platform}', 'states')
            platform_state_path = get_most_recent_state(platform_states_path)
            platform_model, _ = DeepTimePhiPLNN.load(platform_state_path, dtype=dtype)

            # Filter target_df by platform
            platform_df = target_df.filter(
                pl.col('filter_value').cast(pl.String) \
                    .str.to_lowercase() \
                    .str.contains(f'-{platform}-')
            )

            platform_plot_kwargs = plot_kwargs.copy()
            platform_plot_kwargs['t_range'] = trange
            platform_plot_kwargs['show_colorbar'] = False
            platform_plot_kwargs['confinement_threshold'] = platform_model.confinement_threshold
            platform_plot_kwargs['show_x_tick_labels'] = i in [2, 3]
            platform_plot_kwargs['show_y_tick_labels'] = i in [0, 2]
            platform_plot_kwargs['show_x_axis_labels'] = i in [2, 3]
            platform_plot_kwargs['show_y_axis_labels'] = i in [0, 2]
            platform_plot_kwargs['show_legend'] = (i == 1)

            if contours is not None:
                platform_plot_kwargs['levels'] = contours._levels

            _, contours = plot_density_streamplot(
                fig, axes[i], platform_model, platform_df, components, stance_cols,
                **platform_plot_kwargs
            )
            axes[i].set_title(platform_pretty[platform])

        fig.subplots_adjust(
            left=0.01,
            right=0.94,
            top=0.99,
            bottom=0.1,
            wspace=0.04,
            hspace=0.08
        )

        # Manually position colorbar axis
        cax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(contours, cax=cax, format='%.1e', aspect=40)
        cbar.set_label('Log Stance State Density', rotation=270, labelpad=15)

        fig.savefig(f'{fig_path}/nn_potential_density_streamplot_platform_snapshots.png', bbox_inches='tight', dpi=150, pad_inches=0)

    NUM_DIMS = 2
    if PLOT_DIMS:
        # Use GridSpec with a spacer row for different vertical spacing
        # Small gap between rows 1&2, large gap between rows 2&3 (for x-axis labels)
        fig = plt.figure(figsize=(10, 15 if NUM_DIMS == 3 else 6))
        gs = GridSpec(4 if NUM_DIMS == 3 else 1, 2, figure=fig,
                      height_ratios=[1, 1, 0.3, 1] if NUM_DIMS == 3 else [1],  # row 2 is a spacer
                      hspace=0.01,
                      wspace=0.02)

        axes = np.empty((3 if NUM_DIMS == 3 else 1, 2), dtype=object)
        axes[0, 0] = fig.add_subplot(gs[0, 0])
        axes[0, 1] = fig.add_subplot(gs[0, 1])
        if NUM_DIMS == 3:
            axes[1, 0] = fig.add_subplot(gs[1, 0])
            axes[1, 1] = fig.add_subplot(gs[1, 1])
            # Row 2 of GridSpec is the spacer (no axes)
            axes[2, 0] = fig.add_subplot(gs[3, 0])
            axes[2, 1] = fig.add_subplot(gs[3, 1])

        target_df = filter_df_to_time_range(target_df, trange)
        for user_target_df in target_df.partition_by('filter_value'):
            axes[0,0].plot(
                user_target_df['coord_21d'].arr.get(0).rolling_mean(cfg.rolling_mean_window),
                user_target_df['coord_21d'].arr.get(1).rolling_mean(cfg.rolling_mean_window),
                alpha=0.1
            )
            if NUM_DIMS == 3:
                axes[1,0].plot(
                    user_target_df['coord_21d'].arr.get(0).rolling_mean(cfg.rolling_mean_window),
                    user_target_df['coord_21d'].arr.get(2).rolling_mean(cfg.rolling_mean_window),
                    alpha=0.1
                )
                axes[2,0].plot(
                    user_target_df['coord_21d'].arr.get(1).rolling_mean(cfg.rolling_mean_window),
                    user_target_df['coord_21d'].arr.get(2).rolling_mean(cfg.rolling_mean_window),
                    alpha=0.1
                )

        ax_idxs = [0]
        dim_1s = [0]
        dim_2s = [1]
        if NUM_DIMS == 3:
            ax_idxs.extend([1,2])
            dim_1s.extend([0,1])
            dim_2s.extend([2,2])

        for ax_idx, dim_1, dim_2 in zip(ax_idxs, dim_1s, dim_2s):
            if NUM_DIMS == 3 and ax_idx == 0: # don't show x-axis labels on first row because it shares xaxis with 2nd row
                axes[ax_idx,0].set_xticks([])
            else:
                setup_x_axis_labels(axes[ax_idx,0], components, stance_cols, dim_1=dim_1)
                show_x_dim_labels(axes[ax_idx,0], dimension_labels, dim=dim_1)

            setup_y_axis_labels(axes[ax_idx,0], components, stance_cols, dim_2=dim_2)
            show_y_dim_labels(axes[ax_idx,0], dimension_labels, dim=dim_2)
            
            x_range = (np.percentile(coords[:,dim_1], 0.5), np.percentile(coords[:,dim_1], 99.5))
            y_range = (np.percentile(coords[:,dim_2], 0.5), np.percentile(coords[:,dim_2], 99.5))
            axes[ax_idx,0].set_xlim(x_range)
            axes[ax_idx,0].set_ylim(y_range)
            axes[ax_idx,0].set_aspect('equal')

        plot_kwargs['show_colorbar'] = False
        plot_kwargs['show_y_tick_labels'] = False
        plot_kwargs['show_y_axis_labels'] = False
        if NUM_DIMS == 3:
            plot_kwargs['show_x_tick_labels'] = False
            plot_kwargs['show_x_axis_labels'] = False

        plot_density_streamplot(fig, axes[0,1], model, target_df, components, stance_cols, **plot_kwargs)

        if NUM_DIMS == 3:
            plot_kwargs['show_x_tick_labels'] = True
            plot_kwargs['show_x_axis_labels'] = True
            x_range = (np.percentile(coords[:,0], 0.5), np.percentile(coords[:,0], 99.5))
            y_range = (np.percentile(coords[:,2], 0.5), np.percentile(coords[:,2], 99.5))
            plot_kwargs['xrange'] = x_range
            plot_kwargs['yrange'] = y_range

            dim_0_2_path = './out/states/dims_0_2/states/model_18.pth'
            model, _ = DeepTimePhiPLNN.load(dim_0_2_path, dtype=dtype)

            plot_density_streamplot(fig, axes[1,1], model, target_df, components, stance_cols, dim_1=0, dim_2=2, **plot_kwargs)

            x_range = (np.percentile(coords[:,1], 0.5), np.percentile(coords[:,1], 99.5))
            y_range = (np.percentile(coords[:,2], 0.5), np.percentile(coords[:,2], 99.5))
            plot_kwargs['xrange'] = x_range
            plot_kwargs['yrange'] = y_range

            dim_1_2_path = './out/states/dims_1_2/states/model_37.pth'
            model, _ = DeepTimePhiPLNN.load(dim_1_2_path, dtype=dtype)

            plot_density_streamplot(fig, axes[2,1], model, target_df, components, stance_cols, dim_1=1, dim_2=2, **plot_kwargs)

        fig.savefig(f'{fig_path}/nn_potential_2d_projections.png', dpi=150, bbox_inches='tight', pad_inches=0.2)
        
    
if __name__ == '__main__':
    main()  