"""Ordered-probit cell likelihood and its Gaussian site approximation.

Every post in a cell shares the same predictor f, so the exact cell likelihood
depends on the data only through the three category counts -- the same size as
the Gaussian sufficient statistics. Thresholds are symmetric (-c, +c); a global
shift is absorbed by the per-target intercept b_j and a global scale by W.

The cell likelihood is log-concave in f, so the site precision below is always
positive and no clipping heuristic is needed.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr

GH_N = 12
_x, _w = np.polynomial.hermite_e.hermegauss(GH_N)
GH_X = jnp.asarray(_x)
GH_W = jnp.asarray(_w / np.sqrt(2 * np.pi))


def _log_diff_ndtr(a, b):
    """log(Phi(b) - Phi(a)) for b > a, evaluated in whichever tail is stable."""
    swap = (a + b) > 0
    lo = jnp.where(swap, -b, a)
    hi = jnp.where(swap, -a, b)
    l_lo, l_hi = log_ndtr(lo), log_ndtr(hi)
    return l_hi + jnp.log1p(-jnp.exp(jnp.minimum(l_lo - l_hi, -1e-10)))


def log_probs(f, c):
    """Log P(-1|f), P(0|f), P(+1|f)."""
    return (log_ndtr(-c - f),
            _log_diff_ndtr(-c - f, c - f),
            log_ndtr(f - c))


def cell_loglik(f, c, n_neg, n_neu, n_pos):
    lneg, lneu, lpos = log_probs(f, c)
    return n_neg * lneg + n_neu * lneu + n_pos * lpos


def _quad_nodes(m, v):
    return m[..., None] + jnp.sqrt(jnp.maximum(v, 1e-12))[..., None] * GH_X


def expected_loglik(m, v, c, n_neg, n_neu, n_pos):
    """E[log p(cell | f)] under the current Gaussian marginal f ~ N(m, v)."""
    ll = cell_loglik(_quad_nodes(m, v), c,
                     n_neg[..., None], n_neu[..., None], n_pos[..., None])
    return (GH_W * ll).sum(-1)


def _total(m, v, c, nn, nu_, np_):
    return expected_loglik(m, v, c, nn, nu_, np_).sum()


_d1 = jax.grad(_total, argnums=0)
_d2 = jax.grad(lambda *a: _d1(*a).sum(), argnums=0)
_dc = jax.grad(_total, argnums=2)
_dcc = jax.grad(_dc, argnums=2)


@jax.jit
def sites(m, v, c, n_neg, n_neu, n_pos):
    """Gaussian site (precision, pseudo-observation) matching E[log p] to 2nd order.

    This is the variational (CVI) update: derivatives are taken of the expected
    log-likelihood under the current marginal, not of the likelihood at a point.
    """
    g1 = _d1(m, v, c, n_neg, n_neu, n_pos)
    g2 = _d2(m, v, c, n_neg, n_neu, n_pos)
    tau = jnp.maximum(-g2, 1e-8)
    return tau, m + g1 / tau


@jax.jit
def _threshold_derivs(m, v, c, n_neg, n_neu, n_pos):
    return (_dc(m, v, c, n_neg, n_neu, n_pos),
            _dcc(m, v, c, n_neg, n_neu, n_pos))


def newton_threshold(m, v, c, n_neg, n_neu, n_pos, chunk=750_000):
    """One damped Newton step on the shared threshold.

    The objective sums over every cell, so the derivatives are accumulated
    chunk by chunk rather than materialising the quadrature for all of them.
    """
    g = h = 0.0
    for i in range(0, m.shape[0], chunk):
        s = slice(i, i + chunk)
        gi, hi = _threshold_derivs(m[s], v[s], c, n_neg[s], n_neu[s], n_pos[s])
        g += float(gi); h += float(hi)
    step = -g / h if h < -1e-12 else 0.0
    return float(np.clip(c + np.clip(step, -0.25, 0.25), 0.05, 5.0))


@jax.jit
def predictive(m, v, c):
    """Posterior-averaged category probabilities for one cell."""
    lp = log_probs(_quad_nodes(m, v), c)
    return tuple((GH_W * jnp.exp(l)).sum(-1) for l in lp)


def _chunked(fn, n, chunk, *arrs, scalars=()):
    """Apply a jitted per-cell function in slices; quadrature is memory-hungry."""
    if n <= chunk:
        return fn(*arrs, *scalars)
    outs = [fn(*(a[i:i + chunk] for a in arrs), *scalars)
            for i in range(0, n, chunk)]
    return tuple(jnp.concatenate([o[k] for o in outs]) for k in range(len(outs[0])))


def sites_chunked(m, v, c, n_neg, n_neu, n_pos, chunk=750_000):
    return _chunked(lambda *a: sites(a[0], a[1], c, a[2], a[3], a[4]),
                    m.shape[0], chunk, m, v, n_neg, n_neu, n_pos)


def predictive_chunked(m, v, c, chunk=750_000):
    return _chunked(lambda *a: predictive(a[0], a[1], c), m.shape[0], chunk, m, v)


def init_threshold(n_neg, n_neu, n_pos):
    """Match the marginal neutral share at f = 0."""
    from scipy.stats import norm
    share = float(n_neu.sum() / (n_neg.sum() + n_neu.sum() + n_pos.sum()))
    return float(norm.ppf(0.5 + share / 2))
