"""Subsample seeds from the cached cell-level stance aggregate.

The cache (built by the first pass over the stance files) holds every seed that
passes the PPCA-stage type filter, aggregated to (seed, target, 2-day bin) with
(n, s_sum, s2_sum). Those three moments determine the three ordinal category
counts exactly, so no re-extraction is needed for an ordinal likelihood.

Seeds are taken by systematic sampling on volume rank, which keeps the sample
representative of the whole volume range rather than of its head.

Run on prometheus.
"""

import argparse
import glob
import os

import polars as pl

STANCE_DIR = 'data/stance_targets/noun_phrase_stance'
CACHE = 'tmp/gpfa_all_cells.parquet.zstd'
BIN = '2d'
STANCE_MAP = {'AGAINST': -1.0, 'NEUTRAL': 0.0, 'FAVOR': 1.0}


def build_cells():
    if os.path.exists(CACHE):
        print(f'using cached aggregate {CACHE}', flush=True)
        return
    files = sorted(glob.glob(os.path.join(STANCE_DIR, '*doc_targets_with_stance*.parquet.zstd')))
    print(f'{len(files)} stance files', flush=True)
    parts = []
    for i, f in enumerate(files):
        df = pl.read_parquet(f, columns=['id', 'platform', 'createtime', 'Targets',
                                         'Stances', 'seed'])
        df = df.unique(['id', 'platform']).with_columns([
            pl.col('seed').struct.field('SeedName').alias('SeedName'),
            pl.col('seed').struct.field('MainType').alias('MainType'),
            pl.col('seed').struct.field('SubType').alias('SubType'),
        ]).drop('seed')
        df = df.filter(
            pl.col('MainType').is_in(['politician', 'influencer'])
            | ((pl.col('MainType') == 'foreign') & (~pl.col('SubType').is_in(['media', 'state'])))
        ).filter(pl.col('SeedName') != '')
        if not len(df):
            continue
        df = df.explode(['Targets', 'Stances']) \
            .rename({'Targets': 'target', 'Stances': 'stance'}) \
            .drop_nulls(['target', 'stance'])
        if not len(df):
            continue
        df = df.with_columns([
            pl.col('stance').replace_strict(STANCE_MAP, default=None).alias('s'),
            pl.col('createtime').dt.replace_time_zone(None).dt.truncate(BIN).alias('bin'),
        ]).drop_nulls('s')
        parts.append(df.group_by(['SeedName', 'target', 'bin']).agg(
            pl.col('s').sum().alias('s_sum'),
            (pl.col('s') ** 2).sum().alias('s2_sum'),
            pl.len().alias('n')))
        if (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(files)} files', flush=True)
    agg = pl.concat(parts).group_by(['SeedName', 'target', 'bin']).agg(
        pl.col('s_sum').sum(), pl.col('s2_sum').sum(), pl.col('n').sum())
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    agg.write_parquet(CACHE, compression='zstd')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fraction', type=float, default=0.25)
    ap.add_argument('--min-target-volume', type=int, default=400)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    build_cells()
    lf = pl.scan_parquet(CACHE)

    keep = (lf.group_by('target').agg(pl.col('n').sum().alias('v'))
              .filter(pl.col('v') >= args.min_target_volume).select('target').collect())
    print(f'targets with volume >= {args.min_target_volume}: {len(keep)}', flush=True)
    lf = lf.join(keep.lazy(), on='target', how='inner')

    seed_vol = lf.group_by('SeedName').agg(pl.col('n').sum().alias('v')) \
                 .sort('v', descending=True).collect()
    step = max(int(round(1.0 / args.fraction)), 1)
    chosen = seed_vol[::step]          # systematic on volume rank
    print(f'seeds: {len(seed_vol)} available -> {len(chosen)} chosen (every {step}th '
          f'by volume rank)', flush=True)
    print(chosen.select(pl.col('v').min().alias('vol_min'),
                        pl.col('v').median().alias('vol_med'),
                        pl.col('v').max().alias('vol_max')), flush=True)

    out = lf.join(chosen.select('SeedName').lazy(), on='SeedName', how='inner').collect()
    print(f'rows {len(out):,}  seeds {out["SeedName"].n_unique()}  '
          f'targets {out["target"].n_unique()}  posts {out["n"].sum():,}', flush=True)
    print(f'date range: {out["bin"].min()} .. {out["bin"].max()}', flush=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out.write_parquet(args.out, compression='zstd')
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
