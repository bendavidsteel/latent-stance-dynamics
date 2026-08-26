"""Correctness checks for the ordered-probit site approximation."""

import numpy as np
import jax
import jax.numpy as jnp

from . import ordinal

jax.config.update('jax_enable_x64', True)


def test_probs_sum_to_one():
    f = jnp.linspace(-6, 6, 41)
    for c in (0.1, 0.5, 1.1, 3.0):
        p = sum(jnp.exp(l) for l in ordinal.log_probs(f, c))
        err = float(jnp.abs(p - 1).max())
        assert err < 1e-12, f'c={c} err={err}'
    print('probabilities sum to 1                       OK')


def test_log_diff_matches_naive():
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    a = rng.uniform(-4, 3, 500)
    b = a + rng.uniform(0.05, 4, 500)
    got = np.asarray(ordinal._log_diff_ndtr(jnp.asarray(a), jnp.asarray(b)))
    want = np.log(norm.cdf(b) - norm.cdf(a))
    err = np.abs(got - want).max()
    assert err < 1e-9, err
    print(f'log(Phi(b)-Phi(a)) vs naive                  OK  (max {err:.2e})')


def test_sites_match_numerical_derivatives():
    rng = np.random.default_rng(1)
    n = 200
    m = jnp.asarray(rng.uniform(-2.5, 2.5, n))
    v = jnp.asarray(rng.uniform(0.01, 1.5, n))
    nn = jnp.asarray(rng.integers(0, 30, n).astype(float))
    nu_ = jnp.asarray(rng.integers(0, 80, n).astype(float))
    np_ = jnp.asarray(rng.integers(0, 30, n).astype(float))
    c = 1.1

    tau, nu = ordinal.sites(m, v, c, nn, nu_, np_)
    h = 1e-5
    f0 = ordinal.expected_loglik(m, v, c, nn, nu_, np_)
    fp = ordinal.expected_loglik(m + h, v, c, nn, nu_, np_)
    fm = ordinal.expected_loglik(m - h, v, c, nn, nu_, np_)
    d1 = (fp - fm) / (2 * h)
    d2 = (fp - 2 * f0 + fm) / h ** 2

    e_tau = float(jnp.abs(tau - (-d2)).max() / jnp.abs(d2).max())
    e_nu = float(jnp.abs((nu - m) * tau - d1).max() / jnp.abs(d1).max())
    assert e_tau < 1e-4, e_tau
    assert e_nu < 1e-6, e_nu
    assert float(tau.min()) > 0, 'log-concavity violated'
    print(f'site precision vs finite differences         OK  (rel {e_tau:.2e})')
    print(f'site mean vs finite differences              OK  (rel {e_nu:.2e})')
    print(f'log-concavity: min site precision {float(tau.min()):.3e}   OK')


def test_predictive_is_a_distribution():
    rng = np.random.default_rng(2)
    m = jnp.asarray(rng.uniform(-3, 3, 300))
    v = jnp.asarray(rng.uniform(0.0, 2.0, 300))
    p = ordinal.predictive(m, v, 1.1)
    tot = sum(p)
    assert float(jnp.abs(tot - 1).max()) < 1e-6
    assert all(float(x.min()) >= 0 for x in p)
    print('posterior-averaged predictive is a pmf       OK')


def test_threshold_newton_recovers_truth():
    """With f fixed and known, the Newton step should find the generating c."""
    rng = np.random.default_rng(3)
    n = 4000
    f = jnp.asarray(rng.normal(0, 1.0, n))
    c_true = 1.35
    lp = ordinal.log_probs(f, c_true)
    pr = np.stack([np.asarray(jnp.exp(l)) for l in lp], 1)
    counts = np.stack([rng.multinomial(200, p) for p in pr])
    nn, nu_, np_ = (jnp.asarray(counts[:, i].astype(float)) for i in range(3))
    v = jnp.zeros(n)
    c = 0.4
    for _ in range(60):
        c = ordinal.newton_threshold(f, v, c, nn, nu_, np_)
    err = abs(float(c) - c_true)
    assert err < 0.02, (float(c), c_true)
    print(f'threshold Newton recovers c                  OK  '
          f'({float(c):.4f} vs {c_true})')


if __name__ == '__main__':
    test_probs_sum_to_one()
    test_log_diff_matches_naive()
    test_sites_match_numerical_derivatives()
    test_predictive_is_a_distribution()
    test_threshold_newton_recovers_truth()
    print('\nall ordinal checks passed')
