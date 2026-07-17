import datetime
import logging
import os

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import omegaconf
import numpy as np
import polars as pl
import wandb
from tqdm import tqdm

from plnn.dataset import LandscapeSimulationDataset, NumpyLoader
from plnn.models import DeepTimePhiPLNN
from plnn.loss_functions import select_loss_function
from plnn.optimizers import get_optimizer_args, select_optimizer, get_dt_schedule
from plnn.model_training import train_model

logger = logging.getLogger(__name__)

INITIAL_DATE = datetime.datetime(2020, 1, 1)
UNIT_DAYS = 365.25

def df_to_data(df):
    return [[{'t0': d['t0'], 'x0': np.array(d['x0'])[np.newaxis,:], 't1': d['t1'], 'x1': np.array(d['x1'])[np.newaxis,:]} for d in p.to_dicts()] for p in df.partition_by('filter_value')]


def compute_rolling_means(cfg, target_df, dims):
    """Compute rolling means for each filter_value trajectory."""
    return target_df.with_columns([
            pl.col('x0').arr.get(i).alias(f'x0_{i}') for i in dims
        ])\
        .rolling('createtime', period=f'{cfg.rolling_mean_window}d', group_by='filter_value') \
        .agg([pl.col(f'x0_{i}').mean() for i in dims])\
        .with_columns(
            ((pl.col('createtime') - INITIAL_DATE).dt.total_days() / UNIT_DAYS).alias('t'),
        )\
        .sort(['filter_value', 'createtime'])


def build_horizon_pairs(rolling_df, horizon_days, dims, tolerance_frac=0.25):
    """Build (x0, x1) pairs where x1 is approximately horizon_days in the future.

    Estimates the median observation spacing, shifts by the appropriate number
    of rows, then filters to pairs within tolerance of the target horizon.
    """
    dim_cols = [f'x0_{i}' for i in dims]

    spacing_df = rolling_df.with_columns(
        (pl.col('createtime').diff().over('filter_value')).alias('dt')
    ).drop_nulls('dt')
    median_spacing_days = spacing_df['dt'].median().total_seconds() / 86400
    shift_n = max(1, round(horizon_days / median_spacing_days))

    tolerance_days = max(horizon_days * tolerance_frac, 3)

    paired = rolling_df\
        .with_columns(
            [pl.col('t').shift(-shift_n).over('filter_value').alias('t1'),
             pl.col('createtime').shift(-shift_n).over('filter_value').alias('future_createtime')] + \
            [pl.col(c).shift(-shift_n).over('filter_value').alias(f'x1_{i}') for c, i in zip(dim_cols, dims)]
        )\
        .drop_nulls(['t1'])\
        .with_columns(
            ((pl.col('future_createtime') - pl.col('createtime')).dt.total_days()).alias('actual_gap_days')
        )\
        .filter(
            (pl.col('actual_gap_days') >= horizon_days - tolerance_days) &
            (pl.col('actual_gap_days') <= horizon_days + tolerance_days)
        )\
        .with_columns([
            pl.concat_arr(dim_cols).alias('x0'),
            pl.concat_arr([f'x1_{i}' for i in dims]).alias('x1'),
        ])\
        .rename({'t': 't0'})\
        .select(['t0', 'x0', 't1', 'x1', 'filter_value', 'createtime'])

    return paired


def load_target_df(cfg):
    """Load coords and apply early filtering: platform + non-empty filter_value, sorted, renamed."""
    target_path = os.path.join(cfg.trend_path, f'{cfg.dim_reduction_method}_coords.parquet.zstd')
    target_df = pl.read_parquet(target_path)
    coord_col = [c for c in target_df.columns if c.startswith('coord_')][0]

    if cfg.platform != 'all':
        target_df = target_df.filter(
            pl.col('filter_value').cast(pl.String)\
                .str.to_lowercase()\
                .str.contains(f'-{cfg.platform}-')
        )

    target_df = target_df.filter(pl.col('filter_value') != '')
    target_df = target_df.select(['createtime', 'filter_value', coord_col])\
        .sort(['filter_value', 'createtime'])\
        .rename({coord_col: 'x0'})
    return target_df


def build_training_pairs(cfg, target_df):
    """Build training-time 1-step pairs (rolling mean + 1-step shift + timestep<10d filter).

    No shuffle here — apply_split shuffles for split_type='random'.
    """
    n_dims = cfg.n_dims
    paired = target_df.with_columns([
            pl.col('x0').arr.get(i).alias(f'x0_{i}') for i in range(n_dims)
        ])\
        .rolling('createtime', period=f'{cfg.rolling_mean_window}d', group_by='filter_value') \
        .agg([pl.col(f'x0_{i}').mean() for i in range(n_dims)])\
        .with_columns(((pl.col('createtime') - INITIAL_DATE).dt.total_days() / UNIT_DAYS).alias('t0'))\
        .sort(['filter_value', 't0'])\
        .with_columns(
            [pl.col('t0').shift(-1).over('filter_value').alias('t1'),
             pl.col('createtime').shift(-1).over('filter_value').alias('next_createtime')] + \
            [pl.col(f'x0_{i}').shift(-1).over('filter_value').alias(f'x1_{i}') for i in range(n_dims)]
        )\
        .drop_nulls([f'x0_{i}' for i in range(n_dims)] + [f'x1_{i}' for i in range(n_dims)])\
        .with_columns([
            pl.concat_arr([f'x0_{i}' for i in range(n_dims)]).alias('x0'),
            pl.concat_arr([f'x1_{i}' for i in range(n_dims)]).alias('x1'),
        ])\
        .select(['t0', 'x0', 't1', 'x1', 'filter_value', 'createtime', 'next_createtime'])\
        .with_columns(
            (pl.col('next_createtime') - pl.col('createtime')).alias('timestep'),
        )\
        .filter(pl.col('timestep') < pl.duration(days=10))\
        .drop(['timestep'])

    return paired


def compute_training_split(cfg, target_df=None):
    """Replicate training's train/val split metadata.

    Returns:
        (val_filter_values, cutoff_time):
          - val_filter_values: list[str] held out for val (split_type='filter_value'), else None
          - cutoff_time: datetime cutoff (split_type='time'), else None
        For split_type='random', both are None — apply_split handles random splits inline.

    If target_df is None, rebuilds it from cfg via load_target_df + build_training_pairs.
    Pass an existing target_df (e.g. from training) to avoid redundant work.
    """
    if cfg.split_type == 'random':
        return None, None

    if target_df is None:
        target_df = build_training_pairs(cfg, load_target_df(cfg))

    if cfg.split_type == 'filter_value':
        filter_values = target_df['filter_value'].unique().shuffle(seed=42).to_list()
        num_train = int(len(filter_values) * cfg.train_fraction)
        return filter_values[num_train:], None
    elif cfg.split_type == 'time':
        sorted_df = target_df.sort('createtime')
        cutoff_idx = int(len(sorted_df) * cfg.train_fraction)
        return None, sorted_df['createtime'].item(cutoff_idx)
    else:
        raise ValueError(f"Unknown split_type: {cfg.split_type}. Must be 'random', 'filter_value', or 'time'")


def apply_split(df, split_type, train_fraction, val_filter_values=None, cutoff_time=None):
    """Split df into train/val using metadata from compute_training_split.

    For 'random', shuffles df with seed=42 then takes head/tail.
    For 'filter_value' / 'time', filters df by val_filter_values / cutoff_time.
    """
    if split_type == 'random':
        df = df.sample(fraction=1.0, shuffle=True, seed=42)
        n_train = int(len(df) * train_fraction)
        return df.head(n_train), df.tail(len(df) - n_train)
    elif split_type == 'filter_value':
        if val_filter_values is None:
            raise ValueError("val_filter_values required for split_type='filter_value'")
        train_df = df.filter(~pl.col('filter_value').is_in(val_filter_values))
        val_df = df.filter(pl.col('filter_value').is_in(val_filter_values))
        return train_df, val_df
    elif split_type == 'time':
        if cutoff_time is None:
            raise ValueError("cutoff_time required for split_type='time'")
        train_df = df.filter(pl.col('createtime') < cutoff_time)
        val_df = df.filter(pl.col('createtime') >= cutoff_time)
        return train_df, val_df
    else:
        raise ValueError(f"Unknown split_type: {split_type}. Must be 'random', 'filter_value', or 'time'")


def load_seed_metadata(cfg):
    """Load seed metadata (MainType, Party) from the stance data."""
    dir_path = cfg.base_stance_path
    file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.parquet.zstd')]
    df = pl.read_parquet(file_paths, columns=['seed'])
    return df.select([
        pl.col('seed').struct.field('SeedName'),
        pl.col('seed').struct.field('MainType'),
        pl.col('seed').struct.field('SubType'),
        pl.col('seed').struct.field('Party'),
    ]).unique('SeedName')


def compute_metrics(model_losses, baseline_losses):
    """Compute standard evaluation metrics from per-sample losses."""
    model_mse = float(np.mean(model_losses))
    baseline_mse = float(np.mean(baseline_losses))
    skill = 1.0 - model_mse / baseline_mse if baseline_mse > 0 else 0.0
    frac_better = float(np.mean(model_losses < baseline_losses))
    return {
        'model_mse': model_mse,
        'baseline_mse': baseline_mse,
        'skill_score': skill,
        'frac_better': frac_better,
        'n': len(model_losses),
    }


def evaluate_horizon(model, paired_df, cfg, key, horizon_days, n_dims, seed_df,
                     split_type, train_fraction, val_filter_values=None, cutoff_time=None):
    """Evaluate model at a given horizon with breakdowns by dimension, MainType, Party."""
    prefix = f"horizon_{horizon_days}d"
    logger.info(f"Running {horizon_days}-day horizon evaluation...")

    if len(paired_df) == 0:
        logger.info(f"  {horizon_days}d: no pairs found, skipping")
        return

    _, val_paired = apply_split(
        paired_df, split_type, train_fraction,
        val_filter_values=val_filter_values, cutoff_time=cutoff_time,
    )

    # Join seed metadata for breakdowns
    val_paired = val_paired.with_columns(pl.col('filter_value').cast(pl.String)) \
        .join(seed_df, left_on='filter_value', right_on='SeedName', how='left')

    # Compute per-dimension x0 columns for mean splits
    dim_cols = [f'x0_{i}' for i in range(n_dims)]
    val_paired = val_paired.with_columns([
        pl.col('x0').arr.get(i).alias(f'x0_{i}') for i in range(n_dims)
    ])

    # Compute dimension means for above/below splits
    dim_means = {f'x0_{i}': val_paired[f'x0_{i}'].mean() for i in range(n_dims)}

    # Build all evaluation subsets: overall + breakdowns
    subsets = [('overall', val_paired)]

    # Above/below mean on each dimension
    for i in range(n_dims):
        col = f'x0_{i}'
        mean_val = dim_means[col]
        subsets.append((f'dim{i}_above_mean', val_paired.filter(pl.col(col) >= mean_val)))
        subsets.append((f'dim{i}_below_mean', val_paired.filter(pl.col(col) < mean_val)))

    # Categorical breakdowns
    for col in ['MainType', 'SubType', 'Party']:
        if col not in val_paired.columns:
            continue
        for val in val_paired.drop_nulls(col).filter(pl.col(col) != '')[col].unique().sort().to_list():
            subsets.append((f'{col}_{val}', val_paired.filter(pl.col(col) == val)))

    wandb_metrics = {}
    for subset_name, subset_df in subsets:
        if len(subset_df) == 0:
            logger.info(f"  {prefix}/{subset_name}: no samples, skipping")
            continue

        subset_data = df_to_data(subset_df)
        subset_dataset = LandscapeSimulationDataset(data=subset_data)
        subset_loader = NumpyLoader(
            subset_dataset,
            batch_size=min(cfg.eval_batch_size, len(subset_dataset)),
            shuffle=False,
        )

        key, subkey = jax.random.split(key)
        model_losses, baseline_losses = evaluate_dataloader(model, subset_loader, subkey)
        metrics = compute_metrics(model_losses, baseline_losses)

        logger.info(
            f"  {prefix}/{subset_name}: model_mse={metrics['model_mse']:.6f} "
            f"baseline_mse={metrics['baseline_mse']:.6f} skill={metrics['skill_score']:.4f} "
            f"frac_better={metrics['frac_better']:.3f} n={metrics['n']}"
        )
        if subset_name == 'overall':
            for k, v in metrics.items():
                wandb_metrics[f"{prefix}/{k}"] = v
        for k, v in metrics.items():
            wandb_metrics[f"{prefix}/{subset_name}/{k}"] = v

    for k, v in wandb_metrics.items():
        wandb.run.summary[k] = v


def compute_per_sample_mse(y_pred, y_true):
    """Compute per-sample MSE (squared L2 distance)."""
    return jnp.sum(jnp.square(y_pred - y_true), axis=(-2, -1))


@eqx.filter_jit
def _eval_batch(model, t0, t1, y0, y1, key):
    """JIT-compiled evaluation of a single batch."""
    y_pred = model(t0, t1, y0, key)
    model_mse = compute_per_sample_mse(y_pred, y1)
    baseline_mse = compute_per_sample_mse(y0, y1)
    return model_mse, baseline_mse


def evaluate_dataloader(model, dataloader, key):
    """Evaluate model and no-movement baseline on a dataloader.

    Returns arrays of per-sample MSE for the model and for the baseline.
    """
    inference_model = eqx.tree_inference(model, True)
    model_losses = []
    baseline_losses = []

    for data in tqdm(dataloader, desc="    Batches"):
        inputs, y1 = data
        t0, y0, t1 = inputs

        key, subkey = jax.random.split(key)
        model_mse, baseline_mse = _eval_batch(inference_model, t0, t1, y0, y1, subkey)

        model_losses.append(np.array(model_mse))
        baseline_losses.append(np.array(baseline_mse))

    return np.concatenate(model_losses), np.concatenate(baseline_losses)

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    project_name = 'potential_landscape_training'
    wandb_config = omegaconf.OmegaConf.to_object(cfg)
    wandb.init(project=project_name, config=wandb_config)

    logger.info("Loading data...")

    n_dims = cfg.n_dims
    trend_name = os.path.basename(cfg.trend_path.rstrip('/'))

    if cfg.platform != 'all':
        dir_path = f'./out/{trend_name}/dims_{"_".join([str(d) for d in range(n_dims)])}_{cfg.platform}'
    else:
        dir_path = f'./out/{trend_name}/dims_{"_".join([str(d) for d in range(n_dims)])}'

    if cfg.rolling_mean_window != 100:
        dir_path = f"{dir_path}_rm{cfg.rolling_mean_window}"

    target_df = load_target_df(cfg)

    # Keep rolling mean df for horizon evaluation after training
    rolling_df = compute_rolling_means(cfg, target_df, list(range(n_dims)))

    target_df = build_training_pairs(cfg, target_df)

    val_filter_values, cutoff_time = compute_training_split(cfg, target_df=target_df)
    train_df, val_df = apply_split(
        target_df, cfg.split_type, cfg.train_fraction,
        val_filter_values=val_filter_values, cutoff_time=cutoff_time,
    )

    train_data = df_to_data(train_df)
    val_data = df_to_data(val_df)
    train_dataset = LandscapeSimulationDataset(
        data=train_data
    )

    valid_dataset = LandscapeSimulationDataset(
        data=val_data
    )

    shuffle_train = True
    shuffle_valid = False

    batch_size = cfg.batch_size
    batch_size_train = batch_size
    batch_size_valid = batch_size

    train_dataloader = NumpyLoader(
        train_dataset, 
        batch_size=min(batch_size_train, len(train_dataset)), 
        shuffle=shuffle_train,
    )

    valid_dataloader = NumpyLoader(
        valid_dataset, 
        batch_size=min(batch_size_valid, len(valid_dataset)), 
        shuffle=shuffle_valid,
    )

    # Get dt schedule
    scheduler_kwargs = {
        'dt': cfg.dt,
        'dt_schedule_bounds': [0, 1],
        'dt_schedule_scales': [1.0, 0.1]
    }
    schedule_type = 'stepped'
    dt_schedule = get_dt_schedule(schedule_type, scheduler_kwargs)

    seed = 42
    rng = np.random.default_rng(seed=seed)
    key = jax.random.PRNGKey(int(rng.integers(2**32)))
    key, modelkey, initkey, trainkey = jax.random.split(key, 4)

    confinement_threshold = np.max(np.linalg.norm(target_df['x0'].to_numpy(), axis=1)) * 1.1

    model_type = 'deep_phi'
    dtype = jnp.float32
    cont_path = cfg.cont_path
    args_make = {
        'ndims' : n_dims, 
        'nparams' : n_dims, 
        'ncells' : len(train_dataset), 
        'sigma_init' : cfg.sigma,
        'confine' : cfg.confine,
        'confinement_factor' : cfg.confinement_factor,
        'confinement_threshold' : confinement_threshold,
        'dt0' : cfg.dt,
        'vbt_tol' : cfg.vbt_tol,
        'dt_min' : cfg.dt_min,
        'dt_max' : cfg.dt_max,
        'solver' : cfg.solver,
        'sample_cells' : cfg.model_do_sample,
    }
    args_init = {}

    # Add extra args based on Deep or GMM PLNN
    if model_type in ['deep_phi', 'ne_deep_phi', 'vae_plnn']:
        args_make.update({
            'include_phi_bias' : False,
            'phi_hidden_dims' : list(cfg.phi_hidden_dims),
            'phi_hidden_acts' : cfg.phi_hidden_acts,
            'phi_final_act' : cfg.phi_final_act,
            'phi_layer_normalize' : cfg.phi_layer_normalize,
            'phi_layer_dropout' : cfg.phi_layer_dropout,
        })
        args_init.update({
            'init_phi_weights_method' : cfg.init_phi_weights_method,
            'init_phi_weights_args' : cfg.init_phi_weights_args,
            'init_phi_bias_method' : cfg.init_phi_bias_method,
            'init_phi_bias_args' : cfg.init_phi_bias_args,
        })

    if cont_path:
        # Load previous model
        logger.info(f"Loading model from {cont_path}...")
        model, hyperparams = DeepTimePhiPLNN.load(cont_path, dtype=dtype)
        if cfg.dt > 0:
            logger.info(
                f"Overwriting loaded model's dt0. " \
                f"Was {model.dt0}. Now {cfg.dt}."
            )
            model = eqx.tree_at(lambda m: m.dt0, model, cfg.dt)
            hyperparams['dt0'] = cfg.dt
    else:
        # Construct and initialize the model
        logger.info("Constructing new model...")
        model, hyperparams = DeepTimePhiPLNN.make_model(
            key=modelkey, dtype=dtype,
            **args_make
        )
        model = model.initialize(
            initkey, dtype=dtype, **args_init
        )

    # Get the loss function
    loss_fn = select_loss_function(
        cfg.loss_fn_key, 
        kernel=cfg.loss_fn_kernel,
        bw_range=cfg.loss_fn_bw,
    )

    # Optimizer construction
    optimizer_args = get_optimizer_args(cfg, cfg.num_epochs)

    optimization_method = 'rms'
    optimizer = select_optimizer(
        optimization_method, optimizer_args,
        batch_size=batch_size, dataset_size=len(train_dataset),
    )

    # Plotting kwargs
    plotting_opts = {
        'equal_axes': True,
        'plot_radius': cfg.plot_radius,
        'plot_losses': True,
        'plot_sigma_hist': True,
        'sigma_true': None,  # TODO: include true value of sigma if given.
    }

    model = train_model(
        model,
        loss_fn,
        optimizer,
        train_dataloader,
        valid_dataloader,
        key=trainkey,
        num_epochs=cfg.num_epochs,
        batch_size=batch_size,
        min_epochs=cfg.min_epochs,
        patience=cfg.patience,
        dt_schedule=dt_schedule,
        hyperparams=hyperparams,
        plotting_opts=plotting_opts,
        reduce_dt_on_nan=True,
        reduce_cf_on_nan=True,
        logprint=logger.info,
        outdir=dir_path
    )

    # Load seed metadata for evaluation breakdowns
    seed_df = load_seed_metadata(cfg)

    # Common kwargs for evaluate_horizon
    eval_kwargs = dict(
        model=model, cfg=cfg, n_dims=n_dims, seed_df=seed_df,
        split_type=cfg.split_type, train_fraction=cfg.train_fraction,
        val_filter_values=val_filter_values if cfg.split_type == 'filter_value' else None,
        cutoff_time=cutoff_time if cfg.split_type == 'time' else None,
    )

    # 7-day and 30-day horizon evaluations
    for horizon_days in [7, 30]:
        key, evalkey = jax.random.split(key)
        paired_df = build_horizon_pairs(rolling_df, horizon_days, list(range(n_dims)))
        evaluate_horizon(paired_df=paired_df, key=evalkey, horizon_days=horizon_days, **eval_kwargs)

    wandb.finish()
    
if __name__ == '__main__':
    main()