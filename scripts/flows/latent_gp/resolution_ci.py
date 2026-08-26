"""Seed-clustered CIs on the reliability/resolution split for the best prior.

Reliability can be improved simply by widening the predictive distribution;
resolution cannot. So whether a drifting latent actually locates the seeds
better than a frozen one is a question about resolution, and it needs an
interval.
"""

import argparse

import numpy as np

from . import fit as fit_ordinal
from . import metrics
from . import cells as prep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--iters', type=int, default=25)
    ap.add_argument('--bin-factor', type=int, default=8)
    ap.add_argument('--dims', type=int, default=5)
    ap.add_argument('--tau', type=float, default=80.)
    ap.add_argument('--rho', type=float, default=0.0)
    ap.add_argument('--reps', type=int, default=300)
    ap.add_argument('--nfast', type=int, default=0,
                    help='per-dimension mix: this many fast dims, rest frozen/slow')
    ap.add_argument('--fast-tau', type=float, default=80.)
    ap.add_argument('--slow-tau', type=float, default=2560.)
    ap.add_argument('--rest', default='const', choices=['const', 'slow'])
    args = ap.parse_args()

    df, meta = prep.load(args.data, args.bin_factor)
    fc, inte = prep.holdout_masks(df, meta)
    d = prep.pack(prep.deflate(df.filter(~fc & ~inte), args.rho), meta)
    print(f"M={meta['M']} J={meta['J']} T={meta['T']} K={args.dims} "
          f"tau={args.tau:.0f} rho={args.rho}\n")

    frozen = fit_ordinal.fit(d, [dict(kind='const', var=1.0)], meta['dt'],
                             args.dims, args.iters)
    if args.nfast > 0:
        fast = dict(kind='wiener', tau=args.fast_tau)
        rest = (dict(kind='const', var=1.0) if args.rest == 'const'
                else dict(kind='wiener', tau=args.slow_tau))
        comps = [[fast]] * args.nfast + [[rest]] * (args.dims - args.nfast)
        print(f"drift prior: {args.nfast} fast (tau={args.fast_tau:.0f}) + "
              f"{args.dims - args.nfast} {args.rest}\n")
    else:
        comps = [dict(kind='wiener', tau=args.tau)]
    drift = fit_ordinal.fit(d, comps, meta['dt'], args.dims, args.iters)

    for tag, mask in (('forecast', fc), ('interior', inte)):
        ev = prep.eval_set(df.filter(mask))
        pa = fit_ordinal.predict(drift, ev)
        pb = fit_ordinal.predict(frozen, ev)
        sa = metrics.sub_problems(*pa, ev)
        sb = metrics.sub_problems(*pb, ev)
        print(f"--- {tag} holdout ---")
        for name in sa:
            p_a, n_succ, n_tot = sa[name]
            p_b = sb[name][0]
            (dr, drl, drh), (ds, dsl, dsh) = metrics.boot_decompose(
                p_a, p_b, n_succ, n_tot, ev['m'], meta['M'], reps=args.reps)
            vr = 'BETTER' if drh < 0 else ('worse' if drl > 0 else '  ns  ')
            vs = 'BETTER' if dsl > 0 else ('worse' if dsh < 0 else '  ns  ')
            print(f"  {name:12} REL {dr:+.5f} [{drl:+.5f},{drh:+.5f}] {vr}"
                  f"   RES {ds:+.5f} [{dsl:+.5f},{dsh:+.5f}] {vs}", flush=True)
        print()


if __name__ == '__main__':
    main()
