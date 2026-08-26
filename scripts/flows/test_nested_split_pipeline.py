"""Smoke test of the training script's data path on synthetic cells.

Exercises load_latent_df -> build_training_pairs -> label_pairs and asserts the
scenario cells are populated and leak-free, without needing plnn or a GPU.

Run as: python test_nested_split_pipeline.py
"""

import os
import sys
import tempfile
import types

import jax
import numpy as np
import polars as pl
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# plnn and wandb are only needed to train; stub them so the data path can run
for name in ('plnn', 'plnn.dataset', 'plnn.models', 'plnn.loss_functions',
             'plnn.optimizers', 'plnn.model_training', 'wandb', 'hydra'):
    sys.modules.setdefault(name, types.ModuleType(name))
for name, attrs in (
        ('plnn.dataset', ['LandscapeSimulationDataset', 'NumpyLoader']),
        ('plnn.models', ['DeepTimePhiPLNN']),
        ('plnn.loss_functions', ['select_loss_function']),
        ('plnn.optimizers', ['get_optimizer_args', 'select_optimizer', 'get_dt_schedule']),
        ('plnn.model_training', ['train_model'])):
    for a in attrs:
        setattr(sys.modules[name], a, object)
sys.modules['hydra'].main = lambda **kw: (lambda f: f)

import splits                                    # noqa: E402
import nn_potential as nnp                       # noqa: E402
from latent_gp.test_latents import synth, M      # noqa: E402


def make_cfg(cells_path, cache_dir):
    return OmegaConf.create({
        'n_dims': 3, 'platform': 'all', 'min_target_volume': 0,
        'rolling_mean_window': 292, 'trend_path': './data/trends',
        'split': {'holdout_days': 30, 'train_frac': 0.70, 'val_frac': 0.10, 'seed': 42},
        'latents': {
            'method': 'gpfa', 'cells_path': cells_path, 'cache_dir': cache_dir,
            'bin_factor': 1, 'interp_days': 1.0, 'n_fast': 1, 'fast_tau': 20.0, 'slow_kind': 'const',
            'slow_tau': 2560.0, 'rho': 0.0, 'iters': 8, 'infer_iters': 5,
            'causal_state': False, 'seed': 0,
        },
    })


def main():
    with tempfile.TemporaryDirectory() as td:
        cells = os.path.join(td, 'cells.parquet.zstd')
        synth(cells)
        cfg = make_cfg(cells, os.path.join(td, 'cache'))
        spec = nnp.split_spec(cfg)

        target_df = nnp.load_latent_df(cfg, spec)
        assert target_df['filter_value'].n_unique() == M
        assert target_df.columns == ['createtime', 'filter_value', 'x0']

        pairs = nnp.build_training_pairs(cfg, target_df, smooth=False)
        labelled = splits.label_pairs(pairs, spec, time_col='next_createtime')
        print(splits.summarise(labelled))

        train = splits.training_rows(labelled)
        seen = 0
        for traj in splits.TRAJ_SPLITS:
            for time in splits.TIME_SPLITS:
                name = splits.scenario_name(traj, time)
                cell = splits.select(labelled, traj, time)
                splits.check_leakage(train, cell, name)
                assert len(cell) > 0, f'{name} is empty'
                seen += len(cell)
        assert seen == len(labelled), (seen, len(labelled))
        print(f'all 6 scenario cells populated and leak-free ({seen} pairs)')

        # the cache must round-trip to the identical latent
        again = nnp.load_latent_df(cfg, spec)
        a = np.stack(target_df['x0'].to_numpy())
        b = np.stack(again['x0'].to_numpy())
        assert np.array_equal(a, b), 'cached latents differ from the fitted ones'
        print('latent cache round-trips exactly')

        # the landscape path must stay float32 even though latent_gp turns on x64
        assert jax.config.jax_enable_x64, 'latent_gp should have enabled x64'
        sample = nnp.df_to_data(pairs.head(4))[0][0]
        for k in ('t0', 'x0', 't1', 'x1'):
            assert np.asarray(sample[k]).dtype == np.float32, (k, sample[k])
        print('df_to_data emits float32 despite global x64')

        # apply_split's nested branch must agree with selecting the cell directly
        for scenario in ('val_out', 'test_out', 'val_in'):
            tr, cell = nnp.apply_split(pairs, 'nested', 0.8, spec=spec, scenario=scenario)
            want = splits.select(labelled, *scenario.rsplit('_', 1))
            assert len(cell) == len(want) and len(tr) == len(train), scenario
        print('apply_split nested branch matches direct scenario selection')

        # run_dir must separate configurations that change what the model sees
        d1 = nnp.run_dir(cfg)
        cfg.latents.n_fast = 2
        d2 = nnp.run_dir(cfg)
        cfg.split.holdout_days = 91
        d3 = nnp.run_dir(cfg)
        assert len({d1, d2, d3}) == 3, (d1, d2, d3)
        print('run_dir separates latent and split configurations')

    print('\nnested-split pipeline smoke test passed')


if __name__ == '__main__':
    main()
