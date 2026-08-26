"""Cell-level stance aggregate: loading, holdout masks and packing.

A cell is one (seed, target, time bin). Labels are {-1, 0, +1}, so the three
ordinal category counts are determined exactly by (n, s_sum, s2_sum) and the
aggregate never has to be rebuilt when the likelihood changes.
"""

import datetime

import numpy as np
import polars as pl
import jax.numpy as jnp


def load(path, bin_factor, seeds=None, min_target_volume=None):
    """Re-bin the 2-day aggregate onto the latent grid and index seeds/targets.

    `seeds` restricts the seed set before indices are assigned, so a subset run
    still produces contiguous indices.
    """
    df = pl.read_parquet(path)
    if seeds is not None:
        df = df.filter(pl.col('SeedName').is_in(list(seeds)))
    if min_target_volume:
        keep = df.group_by('target').agg(pl.col('n').sum().alias('v')) \
                 .filter(pl.col('v') >= min_target_volume).select('target')
        df = df.join(keep, on='target', how='inner')

    seed_names = df['SeedName'].unique().sort().to_list()
    targets = df['target'].unique().sort().to_list()
    t0 = df['bin'].min()
    df = df.with_columns([
        pl.col('SeedName').replace_strict({s: i for i, s in enumerate(seed_names)}).alias('m'),
        pl.col('target').replace_strict({s: i for i, s in enumerate(targets)}).alias('j'),
        ((pl.col('bin') - pl.lit(t0)).dt.total_days() // (2 * bin_factor))
            .cast(pl.Int64).alias('t'),
    ])
    # group_by does not preserve order, and row order sets the reduction order
    # of the scatters downstream -- sort so repeated runs agree bit for bit
    df = df.group_by(['m', 'j', 't']).agg(
        pl.col('n').sum(), pl.col('s_sum').sum(), pl.col('s2_sum').sum()
    ).sort(['m', 'j', 't'])
    df = df.with_columns([
        pl.col('n').cast(pl.Float64),
        ((pl.col('s2_sum') + pl.col('s_sum')) / 2).alias('n_pos'),
        ((pl.col('s2_sum') - pl.col('s_sum')) / 2).alias('n_neg'),
        (pl.col('n').cast(pl.Float64) - pl.col('s2_sum')).alias('n_neu'),
    ])
    meta = dict(seeds=seed_names, targets=targets, t0=t0,
                M=len(seed_names), J=len(targets), T=int(df['t'].max()) + 1,
                dt=2.0 * bin_factor)
    return df, meta


def bin_times(meta):
    """Datetime at the centre of each latent bin.

    The centre, not the left edge, so a bin's time label is unbiased with
    respect to the posts it aggregates.
    """
    half = datetime.timedelta(days=meta['dt'] / 2)
    return [meta['t0'] + datetime.timedelta(days=meta['dt'] * t) + half
            for t in range(meta['T'])]


def cutoff_bin(meta, holdout_days):
    """First bin index inside the holdout window.

    Bins are assigned by their centre, so a bin straddling the cutoff falls on
    whichever side holds most of its posts.
    """
    times = bin_times(meta)
    cutoff = times[-1] - datetime.timedelta(days=holdout_days)
    for t, ts in enumerate(times):
        if ts >= cutoff:
            return t, cutoff
    return meta['T'], cutoff


def holdout_masks(df, meta, frac=0.15, seed=0):
    """Two contiguous holdouts per seed: a forecast tail and an interior block.

    Contiguous blocks are what identify a timescale -- random per-cell holdout
    cannot, since neighbouring bins predict a held-out cell from the level
    alone. The tail tests extrapolation, the interior block interpolation.
    """
    rng = np.random.default_rng(seed)
    span = df.group_by('m').agg(pl.col('t').min().alias('lo'), pl.col('t').max().alias('hi'))
    lo = dict(zip(span['m'].to_list(), span['lo'].to_list()))
    hi = dict(zip(span['m'].to_list(), span['hi'].to_list()))

    fc_lo, in_lo, in_hi = {}, {}, {}
    for m in range(meta['M']):
        w = max(int(frac * (hi[m] - lo[m] + 1)), 2)
        fc_lo[m] = hi[m] - w + 1
        room = fc_lo[m] - lo[m] - 2 * w                 # keep a gap either side
        if room > 0:
            in_lo[m] = lo[m] + w + int(rng.integers(0, room))
            in_hi[m] = in_lo[m] + w
        else:
            in_lo[m] = in_hi[m] = -1

    m_arr = df['m'].to_numpy(); t_arr = df['t'].to_numpy()
    fc = np.array([t >= fc_lo[m] for m, t in zip(m_arr, t_arr)])
    inte = np.array([in_lo[m] <= t < in_hi[m] for m, t in zip(m_arr, t_arr)])
    return fc, inte


def deflate(df, rho):
    """Effective sample size per cell: posts in a burst are not independent.

    n_eff = n / (1 + (n-1) rho) is the classic design-effect correction; rho=0
    recovers the iid-per-post model.
    """
    if rho <= 0:
        return df.with_columns(pl.col('n').alias('n_eff'))
    return df.with_columns(
        (pl.col('n') / (1.0 + (pl.col('n') - 1.0) * rho)).alias('n_eff'))


def pack(df, meta):
    """Arrays for the E/M steps, with per-cell counts scaled to n_eff."""
    n = df['n'].to_numpy().astype(np.float64)
    ne = df['n_eff'].to_numpy().astype(np.float64)
    scale = ne / n
    return {
        'j': jnp.asarray(df['j'].to_numpy()),
        'flat': jnp.asarray(df['m'].to_numpy() * meta['T'] + df['t'].to_numpy()),
        'n': jnp.asarray(ne),
        's': jnp.asarray(df['s_sum'].to_numpy() * scale),
        's2': jnp.asarray(df['s2_sum'].to_numpy() * scale),
        'n_pos': jnp.asarray(df['n_pos'].to_numpy() * scale),
        'n_neg': jnp.asarray(df['n_neg'].to_numpy() * scale),
        'n_neu': jnp.asarray(df['n_neu'].to_numpy() * scale),
        'M': meta['M'], 'J': meta['J'], 'T': meta['T'], 'N': float(ne.sum()),
    }


def eval_set(df):
    """Held-out cells scored on their observed mean, weighted by real counts."""
    return dict(
        m=df['m'].to_numpy(), j=df['j'].to_numpy(), t=df['t'].to_numpy(),
        mean=(df['s_sum'] / df['n']).to_numpy(),
        n=df['n'].to_numpy().astype(np.float64),
        n_pos=df['n_pos'].to_numpy(), n_neg=df['n_neg'].to_numpy(),
        n_neu=df['n_neu'].to_numpy(),
    )


def seed_names(path):
    """Trajectory ids present in a cell file, without loading the counts."""
    return (pl.scan_parquet(path).select('SeedName').unique()
              .collect()['SeedName'].sort().to_list())
