"""Split-aware latent trajectories for the landscape model.

Replaces impute + PPCA + rolling mean: the latent is smooth in time by
construction, so no post-hoc smoothing window is needed.

The split applies at two levels, and conflating them is what leaks:

  W, b, c   the global representation -- fit on training trajectories inside
            the training period only, because these are shared across seeds
            and so can carry held-out information into every prediction.

  z_m(t)    per-seed state -- inferred for every trajectory over the whole
            period with the global parameters frozen. This is measurement, not
            prediction: a held-out trajectory is allowed to be *observed*, it
            just must not shape the representation.

The smoothed state at t depends on observations after t, which inflates skill
at horizons short relative to the latent's own timescale. `causal_*` columns
hold the filtered state instead, which has no such dependence.
"""

import dataclasses
import datetime
import hashlib
import os

import numpy as np
import polars as pl

from . import cells, fit as fit_mod


@dataclasses.dataclass(frozen=True)
class LatentConfig:
    cells_path: str
    n_dims: int = 6
    n_fast: int = 2
    fast_tau: float = 80.0
    slow_kind: str = 'const'
    slow_tau: float = 2560.0
    bin_factor: int = 8
    rho: float = 0.0
    iters: int = 25
    infer_iters: int = 15
    min_target_volume: int = 400
    interp_days: float = 0.0        # 0 = keep the native bin grid
    seed: int = 0

    def __post_init__(self):
        # A sweep passes fast_tau=40 where the default is 80.0, and hydra keeps
        # it an int. The tag hashes the value's repr, so without coercion the
        # same configuration keys two different cache entries and every trial
        # refits from scratch.
        for f in dataclasses.fields(self):
            cast = {int: int, float: float, 'int': int, 'float': float}.get(f.type)
            if cast is not None:
                object.__setattr__(self, f.name, cast(getattr(self, f.name)))

    @property
    def tag(self):
        """Cache key: everything that changes the fitted latents."""
        body = '|'.join(f'{f.name}={getattr(self, f.name)}'
                        for f in dataclasses.fields(self) if f.name != 'cells_path')
        return hashlib.blake2b(body.encode(), digest_size=6).hexdigest()


def coord_cols(n_dims):
    return f'coord_{n_dims}d', f'causal_{n_dims}d', f'sd_{n_dims}d'


def _standardise(Ez, train_cells):
    """Centre and scale each dimension using training-region state only.

    W is identified only up to scale, so without this the latent's units drift
    between sweep configurations and any absolute-scale hyperparameter (sigma,
    the confinement threshold) has to be re-tuned for each one.
    """
    ref = Ez[train_cells[:, 0], train_cells[:, 1]]
    mu = ref.mean(0)
    sd = np.maximum(ref.std(0), 1e-6)
    return mu, sd


def build_latents(lcfg, spec, seed_split, cache_dir=None, log=print):
    """Fit the latent-GP factor model under `spec` and return per-bin states.

    `seed_split` maps trajectory id -> 'train' / 'val' / 'test'. Returns a frame
    of (createtime, filter_value, coord, causal coord, posterior sd, n_posts).
    """
    cache = None
    if cache_dir:
        cache = os.path.join(cache_dir, f'latents_{lcfg.tag}_{spec.tag}.parquet.zstd')
        if os.path.exists(cache):
            log(f'reusing cached latents {cache}')
            return pl.read_parquet(cache)

    df, meta = cells.load(lcfg.cells_path, lcfg.bin_factor,
                          min_target_volume=lcfg.min_target_volume)
    K = lcfg.n_dims
    t_cut, cutoff = cells.cutoff_bin(meta, spec.holdout_days)
    is_train = np.array([seed_split.get(s, 'test') == 'train' for s in meta['seeds']])
    log(f"M={meta['M']} J={meta['J']} T={meta['T']} K={K} "
        f"train seeds {int(is_train.sum())} cutoff {cutoff:%Y-%m-%d} (bin {t_cut})")

    comps = fit_mod.prior_components(K, lcfg.n_fast, lcfg.fast_tau,
                                     lcfg.slow_kind, lcfg.slow_tau)

    m_arr = df['m'].to_numpy()
    t_arr = df['t'].to_numpy()
    train_mask = is_train[m_arr] & (t_arr < t_cut)
    if not train_mask.any():
        raise ValueError('no training cells: check holdout_days against the data span')

    d_train = cells.pack(cells.deflate(df.filter(pl.Series(train_mask)), lcfg.rho), meta)
    log(f'fitting on {len(d_train["j"]):,} training cells')
    r = fit_mod.fit(d_train, comps, meta['dt'], K, lcfg.iters, seed=lcfg.seed)

    d_all = cells.pack(cells.deflate(df, lcfg.rho), meta)
    Ez, Ezz = fit_mod.infer(d_all, comps, meta['dt'], K,
                            r['W'], r['b'], r['c'], lcfg.infer_iters)
    Ez_c, _ = fit_mod.infer(d_all, comps, meta['dt'], K,
                            r['W'], r['b'], r['c'], lcfg.infer_iters, filtered=True)

    mu, sd = _standardise(Ez, np.stack([m_arr[train_mask], t_arr[train_mask]], 1))
    Ez = (Ez - mu) / sd
    Ez_c = (Ez_c - mu) / sd
    post_sd = np.sqrt(np.maximum(np.diagonal(Ezz, axis1=2, axis2=3), 0.0)) / sd

    out = _to_frame(df, meta, Ez, Ez_c, post_sd, K, lcfg.interp_days)
    out = out.join(pl.DataFrame({'filter_value': list(seed_split),
                                 'traj_split': list(seed_split.values())}),
                   on='filter_value', how='left')
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        out.write_parquet(cache, compression='zstd')
    return out


def _fine_grid(lo, hi, step):
    """Fractional bin positions per seed at `step` bins apart, plus their seed index."""
    n = np.floor((hi - lo) / step).astype(int) + 1
    offs = np.repeat(np.cumsum(np.r_[0, n[:-1]]), n)
    u = lo.repeat(n) + step * (np.arange(n.sum()) - offs)
    return np.repeat(np.arange(len(lo)), n), np.minimum(u, hi.repeat(n))


def _interpolate(Ez, m_i, u, hi_i):
    """Linear interpolation between bin centres.

    Exact for this prior family: the smoothed mean of a Wiener process between
    two grid points is the Brownian-bridge mean, which is linear in time, and a
    constant component is trivially linear. Nothing is being approximated.
    """
    t_lo = np.floor(u).astype(int)
    t_hi = np.minimum(t_lo + 1, hi_i)
    w = (u - t_lo)[:, None]
    return (1 - w) * Ez[m_i, t_lo] + w * Ez[m_i, t_hi]


def _to_frame(df, meta, Ez, Ez_c, post_sd, K, interp_days=0.0):
    """Rows spanning each seed's observed bins, optionally on a finer grid.

    Bins outside the span carry no data at all, so their state is pure prior
    and would be a fabricated observation for the landscape model.
    """
    span = df.group_by('m').agg(
        pl.col('t').min().alias('lo'), pl.col('t').max().alias('hi')).sort('m')
    lo = span['lo'].to_numpy().astype(float)
    hi = span['hi'].to_numpy().astype(float)
    seed_of = span['m'].to_numpy()

    step = (interp_days / meta['dt']) if interp_days > 0 else 1.0
    row, u = _fine_grid(lo, hi, step)
    m_i = seed_of[row]
    hi_i = hi[row].astype(int)
    t_i = np.floor(u).astype(int)

    posts = df.group_by(['m', 't']).agg(pl.col('n').sum().alias('n_posts'))
    origin = datetime.datetime(1970, 1, 1)
    offset = (meta['t0'] - origin).total_seconds() + 86400 * meta['dt'] / 2
    stamp = np.rint(1e6 * (offset + 86400 * meta['dt'] * u)).astype(np.int64)

    c_coord, c_causal, c_sd = coord_cols(K)
    out = pl.DataFrame({
        'm': m_i,
        't': t_i,
        'createtime': pl.Series(stamp).cast(pl.Datetime('us')),
        'filter_value': [meta['seeds'][m] for m in m_i],
        c_coord: _interpolate(Ez, m_i, u, hi_i),
        c_causal: _interpolate(Ez_c, m_i, u, hi_i),
        c_sd: post_sd[m_i, t_i],
    })
    return out.join(posts, on=['m', 't'], how='left') \
        .with_columns(pl.col('n_posts').fill_null(0.0)) \
        .drop(['m', 't']) \
        .sort(['filter_value', 'createtime'])
