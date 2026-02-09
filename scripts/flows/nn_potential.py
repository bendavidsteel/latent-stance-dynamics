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

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    project_name = 'potential_landscape_training'
    wandb_config = omegaconf.OmegaConf.to_object(cfg)
    wandb.init(project=project_name, config=wandb_config)

    logger.info("Loading data...")

    dims = cfg.dims
    trend_name = os.path.basename(cfg.trend_path.rstrip('/'))
    

    target_path = os.path.join(cfg.trend_path, 'pca_coords.parquet.zstd')
    target_df = pl.read_parquet(target_path, columns=['createtime', 'filter_value', 'coord_21d'])
    
    if cfg.platform != 'all':
        target_df = target_df.filter(
                pl.col('filter_value').cast(pl.String)\
                    .str.to_lowercase()\
                    .str.contains(f'-{cfg.platform}-')
            )
        dir_path = f'./out/{trend_name}/dims_{"_".join([str(d) for d in dims])}_{cfg.platform}'
    else:
        dir_path = f'./out/{trend_name}/dims_{"_".join([str(d) for d in dims])}'

    target_df = target_df.filter(pl.col('filter_value') != '')
    target_df = target_df.select(['createtime', 'filter_value', f'coord_21d'])\
        .sort(['filter_value', 'createtime'])\
        .with_columns(((pl.col('createtime') - INITIAL_DATE).dt.total_days() / UNIT_DAYS).alias('t0'))\
        .rename({'coord_21d': 'x0'})

    target_df = target_df.with_columns([
            pl.col('x0').arr.get(i).alias(f'x0_{i}') for i in dims
        ])\
        .with_columns([pl.col(f'x0_{i}').rolling_mean(cfg.rolling_mean_window).over('filter_value') for i in dims])\
        .with_columns(
            [pl.col('t0').shift(-1).over('filter_value').alias('t1'), pl.col('createtime').shift(-1).over('filter_value').alias('next_createtime')] + \
            [pl.col(f'x0_{i}').shift(-1).over('filter_value').alias(f'x1_{i}') for i in dims]
        )\
        .drop_nulls([f'x0_{i}' for i in dims] + [f'x1_{i}' for i in dims])\
        .with_columns([
            pl.concat_arr([f'x0_{i}' for i in dims]).alias('x0'),
            pl.concat_arr([f'x1_{i}' for i in dims]).alias('x1'),
        ])\
        .sample(fraction=1.0, shuffle=True, seed=42)\
        .select(['t0', 'x0', 't1', 'x1', 'filter_value', 'createtime', 'next_createtime'])
    
    target_df = target_df.with_columns([
            (pl.col('next_createtime') - pl.col('createtime')).alias('timestep'),
            (pl.col('t1') - pl.col('t0')).alias('dt'),
        ]).filter(pl.col('timestep') < pl.duration(days=10)).drop(['dt', 'timestep'])

    if cfg.split_type == 'random':
        train_df = target_df.head(int(len(target_df) * cfg.train_fraction))
        val_df = target_df.tail(len(target_df) - len(train_df))
    elif cfg.split_type == 'filter_value':
        filter_values = target_df['filter_value'].unique().shuffle(seed=42).to_list()
        num_train = int(len(filter_values) * cfg.train_fraction)
        train_filter_values = filter_values[:num_train]
        val_filter_values = filter_values[num_train:]
        train_df = target_df.filter(pl.col('filter_value').is_in(train_filter_values))
        val_df = target_df.filter(pl.col('filter_value').is_in(val_filter_values))
    elif cfg.split_type == 'time':
        sorted_df = target_df.sort('createtime')
        cutoff_idx = int(len(sorted_df) * cfg.train_fraction)
        cutoff_time = sorted_df['createtime'].item(cutoff_idx)
        train_df = target_df.filter(pl.col('createtime') < cutoff_time)
        val_df = target_df.filter(pl.col('createtime') >= cutoff_time)
    else:
        raise ValueError(f"Unknown split_type: {cfg.split_type}. Must be 'random', 'filter_value', or 'time'")

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
    cont_path = False
    args_make = {
        'ndims' : len(cfg.dims), 
        'nparams' : len(cfg.dims), 
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
        dt_schedule=dt_schedule,
        hyperparams=hyperparams,
        plotting_opts=plotting_opts,
        reduce_dt_on_nan=True,
        reduce_cf_on_nan=True,
        logprint=logger.info,
        outdir=dir_path
    )
    
    wandb.finish()
    
if __name__ == '__main__':
    main()