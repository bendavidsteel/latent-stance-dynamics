"""Ordered-probit latent-GP factor model: fitting, inference and scoring.

    f_mjt = W_j . z_m(t) + b_j          W, b, c global; z per-seed

Non-conjugate, but the cell likelihood depends on z only through the scalar f,
so each cell is replaced by a Gaussian site matching the expected
log-likelihood to second order (CVI). The per-bin assembly and the smoother are
then the linear-Gaussian ones.

`fit` learns the global parameters; `infer` runs the same E-step with them held
fixed, which is how trajectories outside the training split get a state
estimate without informing the representation.
"""

import numpy as np
import jax
import jax.numpy as jnp

from . import core, metrics, ordinal


def prior_components(K, n_fast, fast_tau, slow_kind='const', slow_tau=2560.0, var=1.0):
    """Per-dimension prior: n_fast drifting dimensions, the rest slow or frozen.

    Homogeneous mixes are returned in the shared form, which leaves the latent
    basis free to rotate; a genuine mix is per-dimension and so fixes the basis.
    """
    fast = dict(kind='wiener', tau=float(fast_tau), var=var)
    slow = (dict(kind='const', var=var) if slow_kind == 'const'
            else dict(kind='wiener', tau=float(slow_tau), var=var))
    if n_fast <= 0:
        return [slow]
    if n_fast >= K:
        return [fast]
    return [[fast]] * n_fast + [[slow]] * (K - n_fast)


def _marginal_f(d, W, b, Ez, Ezz, K):
    Wj = W[d['j']]
    ez = Ez.reshape(-1, K)[d['flat']]
    cov = Ezz.reshape(-1, K, K)[d['flat']]
    m = (Wj * ez).sum(-1) + b[d['j']]
    v = jnp.maximum(jnp.einsum('ci,cij,cj->c', Wj, cov, Wj), 1e-10)
    return m, v


def fit(d, comps, dt, K, n_iter=25, damping=0.6, seed=0):
    """EM with a variational Newton E-step; learns W, b, c and the posterior z."""
    key = jax.random.PRNGKey(seed)
    W = jax.random.normal(key, (d['J'], K)) / np.sqrt(K)
    b = jnp.zeros(d['J'])
    c = ordinal.init_threshold(d['n_neg'], d['n_neu'], d['n_pos'])
    F, Q, P0, S = core.build_ssm(dt, comps, K)
    Sj = jnp.asarray(S)
    smoother = core.make_smoother(F, Q, P0, S)

    m_f = jnp.zeros(d['j'].shape[0])
    v_f = jnp.ones(d['j'].shape[0])
    tau = h = None
    Ez = Ezz = None

    for _ in range(n_iter):
        t_new, nu_new = ordinal.sites_chunked(m_f, v_f, c, d['n_neg'], d['n_neu'], d['n_pos'])
        if tau is None:
            tau, h = t_new, t_new * nu_new
        else:   # damp in natural parameters, as variational Newton requires
            tau = (1 - damping) * tau + damping * t_new
            h = (1 - damping) * h + damping * (t_new * nu_new)
        nu = h / tau

        G, g = core.assemble(d, W, tau, tau * (nu - b[d['j']]), K)
        Ez, Ezz = smoother(*core.to_information(G, g, Sj))

        W, b = core.m_step(d, Ez, Ezz, tau, nu, K)
        m_f, v_f = _marginal_f(d, W, b, Ez, Ezz, K)
        c = ordinal.newton_threshold(m_f, v_f, c, d['n_neg'], d['n_neu'], d['n_pos'])

    if not core.is_heterogeneous(comps):
        W, Ez = core.identify(W, Ez, K)
    return dict(W=np.asarray(W), b=np.asarray(b), c=c,
                Ez=np.asarray(Ez), Ezz=np.asarray(Ezz))


def infer(d, comps, dt, K, W, b, c, n_iter=15, damping=0.6, filtered=False):
    """E-step only: posterior over z with the global parameters frozen.

    Trajectories held out of the fit get their state this way, so their data
    never reaches W, b or c. With filtered=True the state at t sees only
    observations up to t.
    """
    W = jnp.asarray(W); b = jnp.asarray(b)
    F, Q, P0, S = core.build_ssm(dt, comps, K)
    Sj = jnp.asarray(S)
    smoother = core.make_smoother(F, Q, P0, S, filtered=filtered)

    m_f = jnp.zeros(d['j'].shape[0])
    v_f = jnp.ones(d['j'].shape[0])
    tau = h = None
    Ez = Ezz = None

    for _ in range(n_iter):
        t_new, nu_new = ordinal.sites_chunked(m_f, v_f, c, d['n_neg'], d['n_neu'], d['n_pos'])
        if tau is None:
            tau, h = t_new, t_new * nu_new
        else:
            tau = (1 - damping) * tau + damping * t_new
            h = (1 - damping) * h + damping * (t_new * nu_new)
        nu = h / tau

        G, g = core.assemble(d, W, tau, tau * (nu - b[d['j']]), K)
        Ez, Ezz = smoother(*core.to_information(G, g, Sj))
        m_f, v_f = _marginal_f(d, W, b, Ez, Ezz, K)

    return np.asarray(Ez), np.asarray(Ezz)


# ------------------------------------------------------------------ scoring

def predict(r, ev):
    """Posterior-averaged category probabilities for every held-out cell."""
    W, b, Ez, Ezz = r['W'], r['b'], r['Ez'], r['Ezz']
    Wj = W[ev['j']]
    m = (Wj * Ez[ev['m'], ev['t']]).sum(-1) + b[ev['j']]
    v = np.maximum(np.einsum('ci,cij,cj->c', Wj, Ezz[ev['m'], ev['t']], Wj), 1e-10)
    return tuple(np.asarray(x) for x in ordinal.predictive_chunked(
        jnp.asarray(m), jnp.asarray(v), r['c']))


def cell_scores(r, ev):
    """Per-cell RPS, log score and squared error of the predicted mean.

    RPS and the log score are both proper; RPS is the one that respects the
    category ordering. Squared error is kept only because it is comparable
    across likelihoods.
    """
    p_neg, p_neu, p_pos = predict(r, ev)
    eps = 1e-12
    ll = (ev['n_neg'] * np.log(p_neg + eps) + ev['n_neu'] * np.log(p_neu + eps)
          + ev['n_pos'] * np.log(p_pos + eps))
    return dict(rps=metrics.rps(p_neg, p_neu, p_pos, ev), ll=ll,
                se=((p_pos - p_neg) - ev['mean']) ** 2,
                p=(p_neg, p_neu, p_pos))


def score(r, ev):
    sc = cell_scores(r, ev)
    n = ev['n'].sum()
    out = dict(rps=float(sc['rps'].sum() / n), ll=float(sc['ll'].sum() / n),
               mse=float(np.average(sc['se'], weights=ev['n'])))
    for name, (p, ns, nt) in metrics.sub_problems(*sc['p'], ev).items():
        bs, rel, res, unc = metrics.decompose(p, ns, nt)
        out[name] = dict(bs=bs, rel=rel, res=res, unc=unc)
    return out


def boot(tot_a, tot_b, ev, M, reps=2000, seed=0):
    """Seed-clustered bootstrap of the per-post difference (a - b).

    Each metric is a ratio of per-seed sums, so the seeds are reduced once and
    a replicate is a sum over resampled seeds -- no re-indexing of the millions
    of held-out cells.
    """
    agg = lambda x: np.bincount(ev['m'], weights=x, minlength=M)
    a, b, den = agg(tot_a), agg(tot_b), agg(ev['n'])
    live = np.flatnonzero(den > 0)
    rng = np.random.default_rng(seed)
    idx = live[rng.integers(0, len(live), (reps, len(live)))]
    d = (a[idx].sum(1) - b[idx].sum(1)) / den[idx].sum(1)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
