"""Proper scoring rules for the ordinal predictive distribution.

Cell-mean MSE only scores one functional of the forecast, and the log score
alone cannot say *why* one model beats another. Both are fixed here:

  rps   -- ranked probability score, the proper scoring rule for ordered
           categories. Unlike Brier or log score it respects the ordering, so
           predicting FAVOR when the truth is AGAINST costs more than
           predicting NEUTRAL.

  decompose -- Murphy decomposition of the Brier score into

               BS = reliability - resolution + uncertainty

           Reliability measures calibration (do forecasts of 0.3 happen 30% of
           the time), resolution measures discrimination (do forecasts vary
           informatively between cells), and uncertainty is the base rate, the
           same for every model. A model that wins only on reliability is
           better calibrated; one that wins on resolution actually locates the
           latent better.
"""

import numpy as np


def rps(p_neg, p_neu, p_pos, ev):
    """Per-cell RPS summed over that cell's posts. Categories are ordered."""
    P1 = p_neg
    P2 = p_neg + p_neu
    return (ev['n_neg'] * ((P1 - 1) ** 2 + (P2 - 1) ** 2)
            + ev['n_neu'] * (P1 ** 2 + (P2 - 1) ** 2)
            + ev['n_pos'] * (P1 ** 2 + P2 ** 2))


def brier(p, n_succ, n_tot):
    """Per-cell Brier score summed over posts, for a binary sub-problem."""
    return n_succ * (1 - p) ** 2 + (n_tot - n_succ) * p ** 2


def decompose(p, n_succ, n_tot, n_bins=50):
    """Murphy decomposition, binning forecasts into equal-mass groups.

    Returns (brier, reliability, resolution, uncertainty). The identity
    BS = REL - RES + UNC holds up to within-bin forecast spread, which is
    reported by the caller as a residual check.
    """
    keep = n_tot > 0
    p, n_succ, n_tot = p[keep], n_succ[keep], n_tot[keep]
    N = n_tot.sum()
    if N == 0:
        return (np.nan,) * 4
    obar = n_succ.sum() / N

    order = np.argsort(p, kind='stable')
    cum = np.cumsum(n_tot[order])
    cuts = np.searchsorted(cum, np.linspace(0, N, n_bins + 1)[1:-1])
    rel = res = 0.0
    for g in np.split(order, cuts):
        if not len(g):
            continue
        nk = n_tot[g].sum()
        if nk <= 0:
            continue
        pbar = (p[g] * n_tot[g]).sum() / nk
        obar_k = n_succ[g].sum() / nk
        rel += nk * (pbar - obar_k) ** 2
        res += nk * (obar_k - obar) ** 2
    bs = brier(p, n_succ, n_tot).sum() / N
    return bs, rel / N, res / N, obar * (1 - obar)


def _sampler(m, M):
    """Precompute per-seed cell blocks so a bootstrap draw is one O(N) gather."""
    order = np.argsort(m, kind='stable')
    counts = np.bincount(m, minlength=M)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    return order, starts, counts, np.flatnonzero(counts > 0)


def _draw(rng, order, starts, counts, live):
    pick = live[rng.integers(0, len(live), len(live))]
    c, s = counts[pick], starts[pick]
    off = np.r_[0, np.cumsum(c)[:-1]]
    return order[np.repeat(s - off, c) + np.arange(c.sum())]


def boot_decompose(p_a, p_b, n_succ, n_tot, m, M, reps=300, n_bins=50, seed=0):
    """Seed-clustered CI for the reliability and resolution differences (a - b).

    Resolution is the claim that matters -- reliability can be improved just by
    widening the forecast, resolution cannot -- so it needs an interval, not a
    point estimate.
    """
    rng = np.random.default_rng(seed)
    samp = _sampler(m, M)
    drel = np.empty(reps); dres = np.empty(reps)
    for r in range(reps):
        ix = _draw(rng, *samp)
        _, ra, sa, _ = decompose(p_a[ix], n_succ[ix], n_tot[ix], n_bins)
        _, rb, sb, _ = decompose(p_b[ix], n_succ[ix], n_tot[ix], n_bins)
        drel[r], dres[r] = ra - rb, sa - sb
    q = lambda x: (float(np.mean(x)), float(np.percentile(x, 2.5)),
                   float(np.percentile(x, 97.5)))
    return q(drel), q(dres)


def sub_problems(p_neg, p_neu, p_pos, ev, eps=1e-12):
    """The two binary questions the ordinal forecast answers.

    Splitting them matters because they are substantively different: how
    engaged a seed is with a target, and which side it takes given that it
    engages.
    """
    p_op = np.clip(1.0 - p_neu, eps, None)
    return {
        'neutrality': (p_neu, ev['n_neu'], ev['n']),
        'polarity': ((p_pos + eps) / p_op, ev['n_pos'], ev['n_neg'] + ev['n_pos']),
    }
