"""End-to-end checks for the split-aware latent pipeline, on synthetic cells.

The leakage check is the point of the file: perturbing a held-out trajectory's
posts must leave every training trajectory's exported coordinate bit-identical.
That fails if the global parameters see held-out data, and it fails if per-seed
inference lets seeds mix.

Run as: python -m latent_gp.test_latents
"""

import datetime
import os
import sys
import tempfile

import numpy as np
import polars as pl
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import splits                                  # noqa: E402
from latent_gp import LatentConfig, build_latents, coord_cols   # noqa: E402

M, J, T, K_TRUE, C_TRUE = 40, 25, 60, 3, 1.2
BIN_DAYS = 2


def synth(path, flip_seeds=()):
    """Cells from a known factor model: 2 frozen dimensions plus one drifting.

    Each seed draws from its own stream, so flipping one seed cannot perturb
    another through the shared generator -- otherwise the leakage check below
    would be measuring the sampler, not the model.
    """
    rng = np.random.default_rng(0)
    W = rng.normal(size=(J, K_TRUE)) * 0.8
    b = rng.normal(size=J) * 0.3
    z = np.empty((M, T, K_TRUE))
    z[:, :, :2] = rng.normal(size=(M, 1, 2))                       # frozen
    z[:, :, 2] = np.cumsum(rng.normal(scale=0.15, size=(M, T)), 1)  # drifting

    rows = []
    t0 = datetime.datetime(2020, 1, 1)
    for m in range(M):
        rng = np.random.default_rng(1000 + m)
        for j in range(J):
            n = rng.poisson(6, size=T)
            f = z[m] @ W[j] + b[j]
            if m in flip_seeds:
                f = -f
            p_neg = norm.cdf(-C_TRUE - f)
            p_pos = norm.cdf(f - C_TRUE)
            for t in np.flatnonzero(n > 0):
                counts = rng.multinomial(
                    n[t], [p_neg[t], 1 - p_neg[t] - p_pos[t], p_pos[t]])
                rows.append((f'seed{m:03d}', f'target{j:02d}',
                             t0 + datetime.timedelta(days=int(t) * BIN_DAYS),
                             float(counts[2] - counts[0]),
                             float(counts[2] + counts[0]), int(n[t])))
    pl.DataFrame(rows, schema=['SeedName', 'target', 'bin', 's_sum', 's2_sum', 'n'],
                 orient='row').write_parquet(path, compression='zstd')
    return W, b, z


def run(path, spec, seed_split, lcfg):
    return build_latents(lcfg, spec, seed_split, log=lambda *a: None)


def main():
    rng = np.random.default_rng(0)
    spec = splits.SplitSpec(holdout_days=30)
    lcfg_kw = dict(n_dims=3, n_fast=1, fast_tau=20.0, bin_factor=1,
                   iters=8, infer_iters=5, min_target_volume=0)

    with tempfile.TemporaryDirectory() as td:
        clean = os.path.join(td, 'clean.parquet.zstd')
        W, b, z = synth(clean)

        seeds = [f'seed{m:03d}' for m in range(M)]
        traj = splits.assign_trajectory_split(seeds, spec)
        seed_split = dict(zip(traj['filter_value'], traj['traj_split']))
        n_by = {s: sum(v == s for v in seed_split.values()) for s in splits.TRAJ_SPLITS}
        print(f'trajectory split: {n_by}')
        assert all(n_by[s] > 0 for s in splits.TRAJ_SPLITS), n_by

        a = run(clean, spec, seed_split, LatentConfig(cells_path=clean, **lcfg_kw))

        # same data, but every held-out trajectory's stances reversed
        held = {s for s, v in seed_split.items() if v != 'train'}
        flipped = os.path.join(td, 'flipped.parquet.zstd')
        synth(flipped, flip_seeds={int(s[4:]) for s in held})
        bcfg = LatentConfig(cells_path=flipped, **lcfg_kw)
        bb = run(flipped, spec, seed_split, bcfg)

        c_coord, c_causal, c_sd = coord_cols(3)
        key = ['filter_value', 'createtime']
        j = a.select(key + [c_coord]).join(
            bb.select(key + [c_coord]).rename({c_coord: 'other'}), on=key, how='inner')
        j = j.join(traj, on='filter_value', how='left')

        def gap(rows):
            x = np.stack(rows[c_coord].to_numpy())
            y = np.stack(rows['other'].to_numpy())
            return float(np.abs(x - y).max())

        tr = gap(j.filter(pl.col('traj_split') == 'train'))
        ho = gap(j.filter(pl.col('traj_split') != 'train'))
        print(f'max |delta| on train trajectories: {tr:.3e}')
        print(f'max |delta| on held-out trajectories: {ho:.3e}')
        assert tr < 1e-9, f'held-out data reached the training representation ({tr:.2e})'
        assert ho > 1e-3, f'perturbation had no effect at all ({ho:.2e}) -- test is vacuous'

        # exported frame is well formed and spans only observed bins
        assert set(a.columns) >= {'createtime', 'filter_value', c_coord, c_causal,
                                  c_sd, 'n_posts', 'traj_split'}
        assert a['filter_value'].n_unique() == M
        assert a[c_coord].dtype == pl.Array(pl.Float64, 3), a[c_coord].dtype
        assert np.stack(a[c_sd].to_numpy()).min() >= 0

        # the drifting dimension must actually move and the frozen ones must not
        coords = np.stack(a.sort(['filter_value', 'createtime'])[c_coord].to_numpy())
        per_seed = coords.reshape(M, -1, 3)
        within = per_seed.std(1).mean(0)
        print('within-trajectory sd per latent dim:', np.round(within, 3))
        assert within[0] > 2 * within[1:].max(), \
            f'the fast dimension is not the one that moves: {within}'

        cau = np.stack(a.sort(['filter_value', 'createtime'])[c_causal].to_numpy())
        assert np.isfinite(cau).all()
        print('causal coords finite, shape', cau.shape)

        # interpolating onto a finer grid must reproduce the bin-grid values
        # exactly at the original bin centres, and add rows in between
        fine = run(clean, spec, seed_split,
                   LatentConfig(cells_path=clean, interp_days=0.5, **lcfg_kw))
        assert len(fine) > 1.9 * len(a), (len(fine), len(a))
        shared = a.select(key + [c_coord]).join(
            fine.select(key + [c_coord]).rename({c_coord: 'fine'}), on=key, how='inner')
        assert len(shared) == len(a), (len(shared), len(a))
        d = np.abs(np.stack(shared[c_coord].to_numpy())
                   - np.stack(shared['fine'].to_numpy())).max()
        print(f'{len(fine)} interpolated rows from {len(a)}; '
              f'max |delta| at original bin centres {d:.2e}')
        assert d < 1e-12, d

    print('\nall latent-pipeline checks passed')


if __name__ == '__main__':
    main()
