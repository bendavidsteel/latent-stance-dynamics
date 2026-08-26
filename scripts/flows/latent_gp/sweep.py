"""Prior comparison for the latent-GP factor model.

Non-conjugate, but the cell likelihood depends on z only through the scalar
f = W_j.z_m(t) + b_j, so each cell is replaced by a Gaussian site matching the
expected log-likelihood to second order. The per-bin assembly, the eigen-
factorized emission and the dynamax smoother are then unchanged from the
Gaussian model -- only the per-cell precision and pseudo-target differ.

Scored two ways: predictive log-likelihood per post (the ordinal model's own
metric) and MSE of the predicted cell mean (directly comparable to the
Gaussian fit).

Run as: python -m latent_gp.sweep --data ...
"""

import argparse

import numpy as np

from . import cells, core, metrics
from .fit import fit, cell_scores, score, boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--iters', type=int, default=25)
    ap.add_argument('--bin-factor', type=int, default=8)
    ap.add_argument('--dims', type=int, default=3)
    ap.add_argument('--rho', type=float, default=0.0)
    ap.add_argument('--additive', action='store_true',
                    help='sweep shared additive kernels (slow + fast components)')
    ap.add_argument('--mixed', action='store_true',
                    help='sweep per-dimension fast/slow timescale mixes')
    ap.add_argument('--fast-tau', type=float, default=80.)
    ap.add_argument('--slow-tau', type=float, default=2560.)
    ap.add_argument('--taus', default=None,
                    help='comma-separated Wiener timescales; skips the Matern configs')
    args = ap.parse_args()

    K = args.dims
    df, meta = cells.load(args.data, args.bin_factor)
    fc, inte = cells.holdout_masks(df, meta)
    tr = cells.deflate(df.filter(~fc & ~inte), args.rho)
    d = cells.pack(tr, meta)
    ev_fc, ev_in = cells.eval_set(df.filter(fc)), cells.eval_set(df.filter(inte))
    print(f"M={meta['M']} J={meta['J']} T={meta['T']} K={K} rho={args.rho}  "
          f"train cells {len(tr)}\n")

    # Only priors validated against the dense exact posterior are swept; Matern
    # beyond tau=320 and IWP-2 are excluded because the smoother loses accuracy
    # as Q -> 0 (see test_smoother.py).
    const = dict(kind='const', var=1.0)
    if args.additive:
        # const+Wiener and Wiener+Wiener are degenerate (both collapse to a
        # single Wiener), so a real two-timescale sum needs components of
        # different shape. Matern is capped at l=320, the conditioning limit.
        fast_w = dict(kind='wiener', tau=args.fast_tau)
        cfgs = [('frozen z (const)', [const]),
                (f'all-fast Wiener t={args.fast_tau:.0f}', [fast_w]),
                (f'MIX 2fast+{K - 2}const', [[fast_w]] * 2 + [[const]] * (K - 2))]
        # a low-variance component has a small Q, which pushes the smoother
        # back toward the ill-conditioned regime: l=320 with var=0.3 fails the
        # dense-posterior check, so it is not swept
        for ell, vfs in ((80., (0.3, 1.0)), (160., (0.3, 1.0)), (320., (1.0,))):
            for vf in vfs:
                cfgs.append((f'ADD const+Mat l={ell:.0f} v={vf}',
                             [const, dict(kind='matern32', tau=ell, var=vf)]))
        for ell in (80., 320.):
            cfgs.append((f'ADD Wien{args.slow_tau:.0f}+Mat l={ell:.0f}',
                         [dict(kind='wiener', tau=args.slow_tau),
                          dict(kind='matern32', tau=ell, var=1.0)]))
        cfgs_out = cfgs
    elif args.mixed:
        fast = dict(kind='wiener', tau=args.fast_tau)
        slow = dict(kind='wiener', tau=args.slow_tau)
        cfgs = [('frozen z (all const)', [const]),
                (f'all fast tau={args.fast_tau:.0f}', [fast]),
                (f'all slow tau={args.slow_tau:.0f}', [slow])]
        for nf in range(1, K):
            cfgs.append((f'mix {nf}fast+{K - nf}const',
                         [[fast]] * nf + [[const]] * (K - nf)))
            cfgs.append((f'mix {nf}fast+{K - nf}slow',
                         [[fast]] * nf + [[slow]] * (K - nf)))
        cfgs_out = cfgs
    else:
        cfgs_out = None
    cfgs = [('const only (frozen z)', [const])]
    if args.mixed or args.additive:
        taus = ()
    elif args.taus is None:
        for tau in (160., 320.):
            cfgs.append((f'Matern-3/2  tau={tau:.0f}', [dict(kind='matern32', tau=tau)]))
            cfgs.append((f'const+Matern tau={tau:.0f}',
                         [dict(kind='const', var=1.0), dict(kind='matern32', tau=tau)]))
        taus = (10., 20., 40., 80., 160., 320., 1280., 5120.)
    else:
        taus = tuple(float(x) for x in args.taus.split(','))
    for tau in taus:
        cfgs.append((f'Wiener      tau={tau:.0f}', [dict(kind='wiener', tau=tau)]))
    if cfgs_out is not None:
        cfgs = cfgs_out

    hdr = (f"{'config':22} {'fcRPS':>8} {'fcLL':>9} {'fcMSE':>8}   "
           f"{'neuREL':>8} {'neuRES':>8} {'polREL':>8} {'polRES':>8} {'thr':>6}")
    print(hdr); print('-' * len(hdr))
    print('RPS and MSE: lower better.  LL and RES: higher better.  REL: lower better.')
    print('REL is calibration, RES is discrimination -- a model that wins only on')
    print('REL is better calibrated, not better located.\n')

    base = {}
    for i, (name, comps) in enumerate(cfgs):
        r = fit(d, comps, meta['dt'], K, args.iters)
        f = score(r, ev_fc)
        print(f"{name:22} {f['rps']:8.5f} {f['ll']:9.5f} {f['mse']:8.5f}   "
              f"{f['neutrality']['rel']:8.5f} {f['neutrality']['res']:8.5f} "
              f"{f['polarity']['rel']:8.5f} {f['polarity']['res']:8.5f} {r['c']:6.3f}",
              flush=True)
        for tag, ev in (('fcast', ev_fc), ('interp', ev_in)):
            sc = cell_scores(r, ev)
            if i == 0:                      # cfgs[0] is the frozen-z baseline
                base[tag] = sc
                continue
            out = []
            for key, better_is_pos in (('rps', False), ('ll', True), ('se', False)):
                a = sc[key] * (ev['n'] if key == 'se' else 1.0)
                b = base[tag][key] * (ev['n'] if key == 'se' else 1.0)
                dv, lo, hi = boot(a, b, ev, meta['M'])
                good = lo > 0 if better_is_pos else hi < 0
                bad = hi < 0 if better_is_pos else lo > 0
                out.append(f"{key.upper()} {dv:+.5f} "
                           f"{'BETTER' if good else ('worse' if bad else '  ns  ')}")
            print(f"      vs frozen z, {tag:6} " + '  '.join(out))


if __name__ == '__main__':
    main()
