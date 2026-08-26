"""Is the drift shared across targets within a seed, or idiosyncratic?

A factor model can only recover movement that is common to many targets: that
is what lets averaging beat the per-cell noise down. If each (seed, target)
drifts on its own, no latent trajectory exists to find and freezing z is right.

Split-half test: split each seed's targets into two halves, measure the
early->late change on each half independently, and correlate the halves across
seeds. Target main effects are removed first, since a shift common to all seeds
is the target's own drift, not the seed's.

The same test on the *level* is the positive control -- the level is known to
be shared, because that is exactly what a constant z fits.
"""

import argparse

import numpy as np
import polars as pl

from . import cells as prep


def split_half_r(vals, seed_idx, weights, M, n_rep=2000, min_per_half=4, seed=0):
    """Correlate the two halves' precision-weighted means across seeds.

    Each replicate redraws BOTH the target split and a bootstrap resample of
    seeds: a single arbitrary split is far too unstable to report on its own.

    Also returns a power check -- the spread of the half-means against their
    own sampling error. If that ratio is ~1 the half-means are pure noise, so
    r ~ 0 would mean "cannot measure" rather than "not shared".
    """
    rng = np.random.default_rng(seed)
    n = len(vals)
    rs, snrs = [], []
    for _ in range(n_rep):
        h = rng.integers(0, 2, n)
        key = seed_idx * 2 + h
        den = np.bincount(key, weights=weights, minlength=2 * M)
        num = np.bincount(key, weights=weights * vals, minlength=2 * M)
        cnt = np.bincount(key, minlength=2 * M)
        ok = (cnt[0::2] >= min_per_half) & (cnt[1::2] >= min_per_half)
        if ok.sum() < 8:
            continue
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = (num / den).reshape(M, 2)
            sv = (1.0 / den).reshape(M, 2)      # precision weights => 1/sum(w)
        x = mean[ok]
        snrs.append(np.var(x) / np.mean(sv[ok]))
        pick = rng.integers(0, len(x), len(x))
        xb = x[pick]
        if xb[:, 0].std() < 1e-12 or xb[:, 1].std() < 1e-12:
            continue
        rs.append(np.corrcoef(xb[:, 0], xb[:, 1])[0, 1])
    rs = np.array(rs)
    return (float(np.mean(rs)), float(np.percentile(rs, 2.5)),
            float(np.percentile(rs, 97.5)), int(ok.sum()), float(np.mean(snrs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--bin-factor', type=int, default=8)
    ap.add_argument('--min-half', type=int, default=5, help='min posts per half')
    args = ap.parse_args()

    df, meta = prep.load(args.data, args.bin_factor)
    mid = meta['T'] // 2
    print(f"split at bin {mid} of {meta['T']} ({meta['dt'] * mid:.0f} days in)")

    half = df.with_columns((pl.col('t') >= mid).cast(pl.Int8).alias('h'))
    agg = half.group_by(['m', 'j', 'h']).agg(
        pl.col('n').sum(), pl.col('s_sum').sum(), pl.col('s2_sum').sum(),
        pl.col('n_neu').sum())
    wide = agg.pivot(on='h', index=['m', 'j'],
                     values=['n', 's_sum', 's2_sum', 'n_neu'])
    cols = wide.columns
    ren = {c: c.replace('n_neu_', 'neu').replace('s2_sum_', 'q')
            .replace('s_sum_', 's').replace('n_', 'n') for c in cols}
    wide = wide.rename(ren).drop_nulls()
    wide = wide.filter((pl.col('n0') >= args.min_half) & (pl.col('n1') >= args.min_half))

    d = wide.with_columns([
        (pl.col('s1') / pl.col('n1') - pl.col('s0') / pl.col('n0')).alias('d_mean'),
        (pl.col('neu1') / pl.col('n1') - pl.col('neu0') / pl.col('n0')).alias('d_neu'),
        ((pl.col('s0') + pl.col('s1')) / (pl.col('n0') + pl.col('n1'))).alias('level'),
        # sampling variance of the early->late difference
        (((pl.col('q0') - pl.col('s0') ** 2 / pl.col('n0'))
          + (pl.col('q1') - pl.col('s1') ** 2 / pl.col('n1')))
         / (pl.col('n0') + pl.col('n1') - 2)).alias('v'),
    ])
    d = d.with_columns([
        (pl.col('v') * (1 / pl.col('n0') + 1 / pl.col('n1'))).alias('var_d'),
        (pl.col('n0') + pl.col('n1')).alias('ntot'),
    ]).filter(pl.col('var_d') > 0)
    print(f"(seed,target) pairs with >= {args.min_half} posts each side: {len(d)}")

    # remove target main effects: a shift common to all seeds is the target's
    # own drift, not any seed's
    for c in ('d_mean', 'd_neu', 'level'):
        d = d.with_columns(
            (pl.col(c) - (pl.col(c) * pl.col('ntot')).sum().over('j')
             / pl.col('ntot').sum().over('j')).alias(c))

    d = d.sort(['m', 'j'])          # group_by does not preserve order
    seeds = d['m'].to_numpy()
    w = 1.0 / d['var_d'].to_numpy()
    w_lvl = d['ntot'].to_numpy().astype(float)

    print(f"\n{'quantity':30} {'split-half r':>12} {'95% CI':>18} {'seeds':>6} "
          f"{'var/SE^2':>9} {'S-B':>6}")
    print('-' * 86)
    for name, vals, ww in (
        ('LEVEL (positive control)', d['level'].to_numpy(), w_lvl),
        ('drift: mean stance', d['d_mean'].to_numpy(), w),
        ('drift: neutral share', d['d_neu'].to_numpy(), w),
    ):
        r, lo, hi, nm, snr = split_half_r(vals, seeds, ww, meta['M'])
        sb = 2 * r / (1 + r) if r > -1 else np.nan
        snr_s = f"{snr:9.2f}" if name.startswith('drift') else f"{'-':>9}"
        print(f"{name:30} {r:12.3f}  [{lo:+.3f}, {hi:+.3f}]  {nm:6d} {snr_s} {sb:6.3f}")
    print("\nvar/SE^2 >> 1 means the per-half drift is measurable, so r ~ 0 is "
          "evidence of\nidiosyncratic drift rather than of insufficient power.")


if __name__ == '__main__':
    main()
