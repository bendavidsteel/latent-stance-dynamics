"""Stationarity tests for trajectory data.

This module implements several tests for stationarity of dynamical systems:

1. Mean Squared Displacement (MSD) Analysis
   - For stationary processes, MSD plateaus at large lag times
   - For random walks/diffusion, MSD grows linearly with lag time: MSD ∝ 2dDτ
   Reference: Einstein, A. (1905). "On the Movement of Small Particles Suspended
   in Stationary Liquids Required by the Molecular-Kinetic Theory of Heat."
   See also: https://docs.mdanalysis.org/stable/documentation_pages/analysis/msd.html

2. Rolling Window Statistics
   - Compares mean and variance across time windows
   - Non-stationarity indicated by drift in these statistics
   Reference: Common diagnostic, see e.g. Hyndman & Athanasopoulos (2021)
   "Forecasting: Principles and Practice" https://otexts.com/fpp3/

3. Augmented Dickey-Fuller (ADF) Test
   - H0: Unit root present (non-stationary)
   - Reject H0 -> evidence for stationarity
   Reference: Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the Estimators
   for Autoregressive Time Series with a Unit Root." JASA, 74, 427-431.

4. KPSS Test
   - H0: Series is stationary
   - Reject H0 -> evidence for non-stationarity
   Reference: Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., Shin, Y. (1992).
   "Testing the null hypothesis of stationarity against the alternative of a
   unit root." Journal of Econometrics, 54, 159-178.

5. Ensemble Spread Analysis
   - Measures average distance of trajectories from the ensemble centroid
     over time. Constant spread is consistent with a translating cloud
     (drift without dispersion); growing spread suggests divergence.
   - Note: this is *not* a Lyapunov-exponent measurement, which would
     require tracking the separation of nearby trajectory pairs.
"""
import os

import hydra
import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import combine_pvalues
from statsmodels.tsa.stattools import adfuller, kpss
from tqdm import tqdm

from nn_potential import INITIAL_DATE, UNIT_DAYS


def _compute_msd_curve(trajectories: list[np.ndarray], max_lag: int = None) -> tuple:
    """Compute time- and ensemble-averaged MSD(τ) over a sampled lag grid.

    MSD(τ) = <|r(t+τ) - r(t)|²>_{t, ensemble}, summed over spatial dimensions.

    Returns:
        (lags, msd, msd_stderr) — lags is shape (K,), the others same shape.
    """
    min_len = min(len(t) for t in trajectories)
    if max_lag is None:
        max_lag = min_len // 2

    lags = np.arange(1, max_lag + 1, max(1, max_lag // 20))
    msd_values = []
    msd_stderr = []

    for lag in lags:
        squared_displacements = []
        for traj in trajectories:
            T = len(traj)
            if T > lag:
                disp = traj[lag:] - traj[:T - lag]
                sq_disp = np.sum(disp ** 2, axis=1)
                squared_displacements.append(sq_disp)
        if squared_displacements:
            all_sq = np.concatenate(squared_displacements)
            msd_values.append(np.mean(all_sq))
            msd_stderr.append(np.std(all_sq) / np.sqrt(len(all_sq)))
        else:
            msd_values.append(np.nan)
            msd_stderr.append(np.nan)

    msd_values = np.array(msd_values)
    msd_stderr = np.array(msd_stderr)
    valid = ~np.isnan(msd_values)
    return lags[valid], msd_values[valid], msd_stderr[valid]


def test_msd(trajectories: list[np.ndarray], max_lag: int = None, label: str = "") -> dict:
    """Compute Mean Squared Displacement (MSD) and characterize the dynamics.

    MSD(τ) = <|r(t+τ) - r(t)|²> describes how far trajectories travel from
    their start as a function of lag τ. For d-dimensional motion:

    - Pure diffusion:        MSD = 2dDτ                    (linear)
    - Pure ballistic drift:  MSD = (vτ)²                    (quadratic)
    - Drift + diffusion:     MSD = 2dDτ + (vτ)²             (mixed)
    - Confined / OU:         MSD saturates at long τ        (plateau)
    - Anomalous:             MSD ∝ τ^α with α ≠ 1           (sub/super)

    IMPORTANT: MSD shape alone does NOT determine stationarity.
    An Ornstein–Uhlenbeck process is stationary but exhibits MSD ∝ τ at
    short lags before saturating. A pure random walk is non-stationary
    and has MSD ∝ τ at all lags. The shape characterizes the *dynamics*;
    stationarity is decided by ADF/KPSS and by whether MSD saturates.

    This function reports:
      1. A power-law fit MSD ∝ τ^α (α as a free parameter).
      2. A drift+diffusion fit MSD = 2dDτ + (vτ)², separating the diffusion
         coefficient D from a global drift speed |v|.
      3. A simple saturation check (does the long-lag MSD plateau?).

    Reference:
        Einstein, A. (1905). Annalen der Physik, 17, 549-560.
        Qian, H. et al. (1991). Biophys J, 60(4), 910-921.

    Args:
        trajectories: List of trajectory arrays, each shape (T, D).
        max_lag: Maximum lag time (default: min trajectory length // 2).
        label: Optional label for the printed header (e.g. "drift-removed").

    Returns:
        Dictionary with MSD curve, fit parameters, and a regime label.
        is_stationary is intentionally None — see ADF/KPSS for that verdict.
    """
    lags_valid, msd_valid, msd_stderr_valid = _compute_msd_curve(trajectories, max_lag)
    n_dim = trajectories[0].shape[1]

    # Linear fit: MSD = slope * τ + intercept
    slope, intercept, r_value, p_value, _ = stats.linregress(lags_valid, msd_valid)

    # Power law fit: log MSD = α log τ + const
    log_lags = np.log(lags_valid)
    log_msd = np.log(msd_valid + 1e-10)
    alpha, log_const, r_alpha, _, _ = stats.linregress(log_lags, log_msd)

    # Drift+diffusion fit: MSD = a*τ + b*τ²  with a = 2dD, b = |v|²
    # Solve via non-negative-constrained least squares so D and |v|² stay ≥ 0.
    design = np.column_stack([lags_valid, lags_valid ** 2])
    coeffs, *_ = np.linalg.lstsq(design, msd_valid, rcond=None)
    a_fit, b_fit = coeffs
    D_fit = max(a_fit, 0.0) / (2 * n_dim)
    v_fit = np.sqrt(max(b_fit, 0.0))
    msd_pred = design @ coeffs
    ss_res = np.sum((msd_valid - msd_pred) ** 2)
    ss_tot = np.sum((msd_valid - msd_valid.mean()) ** 2)
    r2_drift_diff = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Saturation check: compare MSD over the last third of lags to a flat line.
    n_tail = max(3, len(lags_valid) // 3)
    tail_lags = lags_valid[-n_tail:]
    tail_msd = msd_valid[-n_tail:]
    tail_slope, _, tail_r, tail_p, _ = stats.linregress(tail_lags, tail_msd)
    # Normalize tail slope by the mean tail MSD per unit lag — small => plateau.
    tail_growth_rate = tail_slope * (tail_lags[-1] - tail_lags[0]) / (np.mean(tail_msd) + 1e-10)
    saturates = abs(tail_growth_rate) < 0.1 and tail_p > 0.05

    header = f"=== Mean Squared Displacement (MSD) Analysis{f' [{label}]' if label else ''} ==="
    print(f"\n{header}")
    print(f"MSD at lag {lags_valid[0]}: {msd_valid[0]:.4f} (PC²)")
    print(f"MSD at lag {lags_valid[-1]}: {msd_valid[-1]:.4f} (PC²)")
    print(f"Linear fit:           MSD = {slope:.6f} τ + {intercept:.4f}   R² = {r_value**2:.4f}")
    print(f"Power law fit:        MSD ∝ τ^{alpha:.3f}                       R² = {r_alpha**2:.4f}")
    print(f"Drift+diffusion fit:  MSD = 2·{n_dim}·D·τ + (v·τ)²")
    print(f"                      D = {D_fit:.6f} PC²/lag,  |v| = {v_fit:.6f} PC/lag   R² = {r2_drift_diff:.4f}")
    print(f"Tail (last {n_tail} lags) growth rate over tail span: {tail_growth_rate:+.3f} "
          f"(p = {tail_p:.2e}) -> {'saturates' if saturates else 'still growing'}")

    # Characterize the dynamical regime (NOT stationarity).
    if saturates:
        regime = 'confined_or_OU'
        print("Regime: MSD saturates -> confined dynamics (e.g. Ornstein–Uhlenbeck).")
    elif alpha > 1.5 and r_alpha**2 > 0.9:
        regime = 'ballistic_or_drift_dominated'
        print(f"Regime: α ≈ {alpha:.2f} -> drift-dominated (super-diffusive).")
    elif alpha > 1.1 and r_alpha**2 > 0.9:
        regime = 'drift_plus_diffusion'
        print(f"Regime: α ≈ {alpha:.2f} -> drift + diffusion mix.")
    elif 0.9 <= alpha <= 1.1 and r_alpha**2 > 0.9:
        regime = 'diffusive'
        print(f"Regime: α ≈ {alpha:.2f} -> diffusive scaling. NOTE: this alone does not "
              "distinguish a random walk from an OU process at short lags.")
    elif alpha < 0.9 and r_alpha**2 > 0.8:
        regime = 'subdiffusive'
        print(f"Regime: α ≈ {alpha:.2f} -> sub-diffusive (confined / trapped).")
    else:
        regime = 'unclassified'
        print("Regime: power-law fit poor; see drift+diffusion fit and saturation check.")

    return {
        'lags': lags_valid,
        'msd': msd_valid,
        'msd_stderr': msd_stderr_valid,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'alpha': alpha,
        'alpha_r_squared': r_alpha ** 2,
        'D': D_fit,
        'v': v_fit,
        'drift_diff_r_squared': r2_drift_diff,
        'tail_growth_rate': tail_growth_rate,
        'saturates': saturates,
        'regime': regime,
        'is_stationary': None,  # MSD shape alone does not determine this.
    }


def compute_drift_removed_trajectories(
    df: pl.DataFrame,
    coord_cols: list[str],
    n_centroid_bins: int = 50,
    min_length: int = 100,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Subtract the time-varying ensemble centroid from each trajectory.

    Bins time into ``n_centroid_bins`` equal-width windows, computes the
    ensemble mean of each coord column inside each bin, then subtracts the
    bin's centroid from every observation falling in that bin. The result is
    a set of residual trajectories r_i(t) = x_i(t) - c(t).

    If the picture is "moving school of fish", residual MSD should saturate
    even when raw MSD grows linearly.

    Returns:
        (residual_trajectories, centroid_bin_centers, centroid_values)
        where centroid_values has shape (n_centroid_bins, len(coord_cols)).
    """
    df = df.sort('createtime')
    min_time = df['createtime'].min()
    max_time = df['createtime'].max()
    time_range_seconds = (max_time - min_time).total_seconds()
    bin_seconds = max(1, time_range_seconds / n_centroid_bins)

    df_with_bin = df.with_columns(
        ((pl.col('createtime') - min_time).dt.total_seconds() / bin_seconds)
        .floor()
        .clip(0, n_centroid_bins - 1)
        .cast(pl.Int64)
        .alias('_bin_idx')
    )

    centroid_df = df_with_bin.group_by('_bin_idx').agg(
        [pl.col(c).mean().alias(f'{c}_centroid') for c in coord_cols]
    ).sort('_bin_idx')

    df_residual = df_with_bin.join(centroid_df, on='_bin_idx', how='left').with_columns(
        [(pl.col(c) - pl.col(f'{c}_centroid')).alias(c) for c in coord_cols]
    )

    # Partition by filter_value to recover per-user residual trajectories.
    residual_trajectories = [
        f.sort('createtime').select(coord_cols).to_numpy()
        for f in df_residual.partition_by('filter_value')
        if f.height > min_length
    ]

    centroid_bin_centers = centroid_df['_bin_idx'].to_numpy()
    centroid_values = centroid_df.select([f'{c}_centroid' for c in coord_cols]).to_numpy()
    return residual_trajectories, centroid_bin_centers, centroid_values


def test_rolling_statistics(df: pl.DataFrame, coord_cols: list[str], n_windows: int = 5) -> dict:
    """Test if mean and variance are consistent across time windows.

    A stationary process has constant mean and variance over time. This test
    splits trajectories into non-overlapping time windows and checks whether
    the ensemble statistics differ significantly between windows.

    This is an informal but widely-used diagnostic for non-stationarity.

    Reference:
        Hyndman, R.J. & Athanasopoulos, G. (2021). "Forecasting: Principles
        and Practice" (3rd ed). OTexts. https://otexts.com/fpp3/

    Args:
        df: Polars DataFrame with 'createtime' and coordinate columns
        coord_cols: List of column names containing coordinates
        n_windows: Number of time windows to compare

    Returns:
        Dictionary with window statistics and stationarity assessment
    """
    # Bin every observation by integer window index (deterministic alignment).
    min_time = df['createtime'].min()
    max_time = df['createtime'].max()
    time_range_seconds = (max_time - min_time).total_seconds()
    window_seconds = time_range_seconds / n_windows

    df_w = df.sort('createtime').with_columns(
        ((pl.col('createtime') - min_time).dt.total_seconds() / window_seconds)
        .floor()
        .clip(0, n_windows - 1)
        .cast(pl.Int64)
        .alias('_win')
    )

    window_stats = df_w.group_by('_win').agg(
        [pl.col(c).mean().alias(f'{c}_mean') for c in coord_cols] +
        [pl.col(c).var().alias(f'{c}_var') for c in coord_cols] +
        [pl.len().alias('_n')]
    ).sort('_win')

    window_means = window_stats.select([f'{c}_mean' for c in coord_cols]).to_numpy()
    window_vars = window_stats.select([f'{c}_var' for c in coord_cols]).to_numpy()
    window_n = window_stats['_n'].to_numpy()

    initial_mean = window_means[0]
    initial_var = window_vars[0]
    final_var = window_vars[-1]
    mean_drift = window_means[-1] - window_means[0]
    var_change = window_vars[-1] - window_vars[0]

    # Cohen's d per dim: drift in pooled-std units (drift relative to spread).
    pooled_std = np.sqrt((initial_var + final_var) / 2 + 1e-10)
    cohens_d = mean_drift / pooled_std

    # Variance change relative to initial variance (per dim).
    var_change_rel = var_change / (initial_var + 1e-10)

    # ANOVA across windows per dim: are window means significantly different?
    # F-test for variance equality across windows (Levene's test, more robust
    # than Bartlett to non-normality).
    n_dims = len(coord_cols)
    anova_p = np.full(n_dims, np.nan)
    anova_f = np.full(n_dims, np.nan)
    levene_p = np.full(n_dims, np.nan)
    for d, c in enumerate(coord_cols):
        samples = [
            df_w.filter(pl.col('_win') == w)[c].to_numpy()
            for w in range(n_windows)
        ]
        samples = [s for s in samples if len(s) > 1]
        if len(samples) >= 2:
            anova_f[d], anova_p[d] = stats.f_oneway(*samples)
            try:
                _, levene_p[d] = stats.levene(*samples)
            except ValueError:
                pass

    print("\n=== Rolling Statistics Analysis ===")
    print(f"Window duration: {window_seconds / 86400:.1f} days, "
          f"{len(window_stats)} windows, n per window: "
          f"{window_n.min()}–{window_n.max()}")
    cohens_label = "Cohen's d"
    print("Per-dimension drift (PC units; drift = last − first window):")
    print(f"  {'dim':>3} | {'init μ':>9} | {'Δμ':>9} | {'init σ²':>9} | {'Δσ²':>9} | "
          f"{'Δσ²/σ²₀':>9} | {cohens_label:>10} | {'ANOVA p':>10} | {'Levene p':>10}")
    for d in range(n_dims):
        print(f"  {d:>3} | {initial_mean[d]:>+9.4f} | {mean_drift[d]:>+9.4f} | "
              f"{initial_var[d]:>9.4f} | {var_change[d]:>+9.4f} | "
              f"{var_change_rel[d]:>+9.2%} | "
              f"{cohens_d[d]:>+10.3f} | {anova_p[d]:>10.2e} | {levene_p[d]:>10.2e}")

    # Across-dim summaries.
    avg_abs_d = float(np.mean(np.abs(cohens_d)))
    max_abs_d = float(np.max(np.abs(cohens_d)))
    avg_var_change_rel = float(np.mean(var_change_rel))
    n_sig_mean = int(np.sum((anova_p < 0.05) & (np.abs(cohens_d) > 0.2)))
    n_sig_var = int(np.sum(levene_p < 0.05))

    print(f"\nMean drift summary:  avg |d| = {avg_abs_d:.3f}, max |d| = {max_abs_d:.3f}")
    print(f"  Cohen's d guidelines: 0.2 = small, 0.5 = medium, 0.8 = large effect")
    print(f"  Dims with significant ANOVA (p<0.05) AND |d|>0.2: {n_sig_mean}/{n_dims}")
    print(f"Variance change summary: avg Δσ²/σ²₀ = {avg_var_change_rel:+.2%}")
    print(f"  Dims with significant Levene test (p<0.05): {n_sig_var}/{n_dims}")

    # Decision: require both statistical significance AND non-trivial effect.
    # Variance: any dim with significant Levene + |Δσ²/σ²₀| > 0.5 → non-stationary.
    # Mean: any dim with significant ANOVA + |d| > 0.2 (small effect) → non-stationary.
    var_nonstat = bool(np.any((levene_p < 0.05) & (np.abs(var_change_rel) > 0.5)))
    mean_nonstat = n_sig_mean > 0

    if var_nonstat and mean_nonstat:
        print("Result: Both mean and variance drift significantly -> NON-STATIONARY (mean and variance)")
        is_stationary = False
    elif var_nonstat:
        print("Result: Variance changes significantly -> NON-STATIONARY (in variance)")
        is_stationary = False
    elif mean_nonstat:
        print(f"Result: Mean drifts significantly in {n_sig_mean}/{n_dims} dim(s) "
              f"(avg |d| = {avg_abs_d:.2f}) -> NON-STATIONARY (in mean)")
        is_stationary = False
    else:
        print("Result: No significant drift in mean or variance -> Consistent with STATIONARITY")
        is_stationary = True

    return {
        'window_means': window_means,
        'window_vars': window_vars,
        'window_n': window_n,
        'initial_mean': initial_mean,
        'initial_var': initial_var,
        'mean_drift': mean_drift,
        'var_change': var_change,
        'var_change_rel': var_change_rel,
        'cohens_d': cohens_d,
        'anova_f': anova_f,
        'anova_p': anova_p,
        'levene_p': levene_p,
        'n_sig_mean_dims': n_sig_mean,
        'n_sig_var_dims': n_sig_var,
        'is_stationary': is_stationary,
    }


def test_adf_kpss(trajectories: list[np.ndarray], n_sample: int = 100) -> dict:
    """Run Augmented Dickey-Fuller and KPSS tests on trajectory components.

    These are complementary unit root tests:

    ADF (Augmented Dickey-Fuller):
        - H0: Unit root present (series is non-stationary)
        - H1: No unit root (series is stationary)
        - Reject H0 (p < 0.05) -> evidence FOR stationarity

    KPSS (Kwiatkowski-Phillips-Schmidt-Shin):
        - H0: Series is stationary around a constant
        - H1: Series has a unit root (non-stationary)
        - Reject H0 (p < 0.05) -> evidence AGAINST stationarity

    Using both tests together:
        - ADF rejects, KPSS doesn't reject -> Stationary
        - ADF doesn't reject, KPSS rejects -> Non-stationary
        - Both reject or neither rejects -> Inconclusive (may be trend-stationary)

    P-values from individual tests are combined using Fisher's method (Maddala & Wu, 1999),
    which is a panel unit root test: P = -2 * sum(ln(p_i)) ~ chi-squared(2N).

    References:
        Dickey, D.A. & Fuller, W.A. (1979). "Distribution of the Estimators for
        Autoregressive Time Series with a Unit Root." JASA, 74(366), 427-431.

        Kwiatkowski, D. et al. (1992). "Testing the null hypothesis of stationarity
        against the alternative of a unit root." J. Econometrics, 54(1-3), 159-178.

        Maddala, G.S. & Wu, S. (1999). "A comparative study of unit root tests with
        panel data and a new simple test." Oxford Bulletin of Econ. & Stats., 61, 631-652.

    Args:
        trajectories: List of trajectory arrays
        n_sample: Number of trajectories to sample for testing (for efficiency)

    Returns:
        Dictionary with test statistics, p-values, and stationarity assessment
    """

    n_dims = trajectories[0].shape[1]
    n_test_dims = min(3, n_dims)

    # Store results per dimension (combine across trajectories, not dimensions)
    adf_pvalues_by_dim = {d: [] for d in range(n_test_dims)}
    kpss_pvalues_by_dim = {d: [] for d in range(n_test_dims)}

    print("\n=== ADF and KPSS Tests (Fisher/Maddala-Wu panel unit root) ===")
    print(f"Testing {len(trajectories)} trajectories, first {n_test_dims} dimensions (combined per-dimension)")

    import warnings
    num_adf_errors = 0
    num_kpss_errors = 0
    num_constant = 0
    for traj in tqdm(trajectories, desc="Processing trajectories"):
        for d in range(n_test_dims):
            series = traj[:, d]

            # Skip constant series (causes errors in both tests)
            if np.std(series) < 1e-10:
                num_constant += 1
                continue

            # ADF test
            try:
                adf_stat, adf_p, _, _, _, _ = adfuller(series, autolag='AIC')
                adf_pvalues_by_dim[d].append(adf_p)
            except ValueError:
                num_adf_errors += 1
                pass  # Skip series that cause numerical issues

            # KPSS test (suppress interpolation warnings for extreme values)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kpss_stat, kpss_p, _, _ = kpss(series, regression='c', nlags='auto')
                kpss_pvalues_by_dim[d].append(kpss_p)
            except ValueError:
                num_kpss_errors += 1
                pass  # Skip series that cause numerical issues

    print(f"Completed tests with {num_adf_errors} ADF errors and {num_kpss_errors} KPSS errors, {num_constant} constant series skipped.")

    # Combine p-values per dimension using Fisher's method (Maddala-Wu panel unit root test)
    # P = -2 * sum(ln(p_i)) ~ chi-squared(2N)
    adf_combined_by_dim = {}
    kpss_combined_by_dim = {}

    print(f"\nPer-dimension results:")
    for d in range(n_test_dims):
        if adf_pvalues_by_dim[d]:
            adf_stat, adf_p = combine_pvalues(adf_pvalues_by_dim[d], method='fisher')
            adf_combined_by_dim[d] = {'statistic': adf_stat, 'pvalue': adf_p, 'n_tests': len(adf_pvalues_by_dim[d])}
        if kpss_pvalues_by_dim[d]:
            kpss_stat, kpss_p = combine_pvalues(kpss_pvalues_by_dim[d], method='fisher')
            kpss_combined_by_dim[d] = {'statistic': kpss_stat, 'pvalue': kpss_p, 'n_tests': len(kpss_pvalues_by_dim[d])}

        adf_p = adf_combined_by_dim[d]['pvalue']
        kpss_p = kpss_combined_by_dim[d]['pvalue']
        print(f"  Dim {d}: ADF p={adf_p:.4e} ({'reject' if adf_p < 0.05 else 'fail'}), "
              f"KPSS p={kpss_p:.4e} ({'reject' if kpss_p < 0.05 else 'fail'})")

    # Overall decision: require majority of dimensions to agree
    adf_rejects_by_dim = [adf_combined_by_dim[d]['pvalue'] < 0.05 for d in range(n_test_dims)]
    kpss_rejects_by_dim = [kpss_combined_by_dim[d]['pvalue'] < 0.05 for d in range(n_test_dims)]

    adf_majority_rejects = sum(adf_rejects_by_dim) > n_test_dims / 2
    kpss_majority_rejects = sum(kpss_rejects_by_dim) > n_test_dims / 2

    print(f"\nADF test (H0: unit root / non-stationary):")
    print(f"  Dimensions rejecting: {sum(adf_rejects_by_dim)}/{n_test_dims}")
    print(f"  Majority rejects: {adf_majority_rejects} (reject -> evidence for stationarity)")

    print(f"\nKPSS test (H0: stationary):")
    print(f"  Dimensions rejecting: {sum(kpss_rejects_by_dim)}/{n_test_dims}")
    print(f"  Majority rejects: {kpss_majority_rejects} (reject -> evidence for non-stationarity)")

    # Interpretation based on majority of dimensions.
    # Four-way table (ADF H0: unit root; KPSS H0: stationary):
    #   ADF rejects, KPSS doesn't  -> stationary
    #   ADF doesn't, KPSS rejects  -> unit root / random-walk non-stationary
    #   neither rejects            -> inconclusive (low test power)
    #   both reject                -> trend-stationary or structural breaks
    #                                 (structured non-stationarity, NOT random walk)
    if adf_majority_rejects and not kpss_majority_rejects:
        print("\nResult: ADF rejects unit root, KPSS doesn't reject stationarity -> STATIONARY")
        is_stationary = True
        regime = 'stationary'
    elif not adf_majority_rejects and kpss_majority_rejects:
        print("\nResult: ADF doesn't reject unit root, KPSS rejects stationarity -> NON-STATIONARY (unit root / random walk)")
        is_stationary = False
        regime = 'unit_root'
    elif not adf_majority_rejects and not kpss_majority_rejects:
        print("\nResult: Neither test rejects -> INCONCLUSIVE (low test power)")
        is_stationary = None
        regime = 'inconclusive'
    else:
        print("\nResult: Both tests reject -> TREND-STATIONARY or STRUCTURAL BREAKS")
        print("        (structured non-stationarity around a deterministic component, NOT random walk)")
        is_stationary = False
        regime = 'trend_stationary_or_breaks'

    return {
        'adf_combined_by_dim': adf_combined_by_dim,
        'kpss_combined_by_dim': kpss_combined_by_dim,
        'adf_rejects_by_dim': adf_rejects_by_dim,
        'kpss_rejects_by_dim': kpss_rejects_by_dim,
        'is_stationary': is_stationary
    }


def test_ensemble_spread(df: pl.DataFrame, coord_cols: list[str], n_windows: int = 20) -> dict:
    """Test if the ensemble of trajectories spreads out over time.

    Computes the average distance of trajectories from the ensemble centroid
    at each time point. Constant spread with a moving centroid is consistent
    with rigid translation of the cloud (correlated drift without dispersion);
    growing spread indicates dispersion / divergence.

    Note: This measures ensemble dispersion, not individual trajectory
    divergence. It is NOT a Lyapunov-exponent estimate — that would require
    tracking the separation of initially-nearby trajectory pairs over time.
    Two trajectories can have a positive Lyapunov exponent (locally diverging)
    while the ensemble spread stays constant if the divergence is bounded by
    a confining potential.

    Args:
        df: Polars DataFrame with 'createtime', 'filter_value', and coordinate columns
        coord_cols: List of column names containing coordinates
        n_windows: Number of time windows to sample

    Returns:
        Dictionary with spread measurements and stationarity assessment
    """
    # Compute window duration
    min_time = df['createtime'].min()
    max_time = df['createtime'].max()
    time_range_seconds = (max_time - min_time).total_seconds()
    window_seconds = int(time_range_seconds / n_windows)
    window_duration = f"{window_seconds}s"

    # Get one point per trajectory per time window using group_by_dynamic
    df_sampled = df.sort('createtime').group_by_dynamic(
        'createtime',
        every=window_duration,
        group_by='filter_value'
    ).agg([pl.col(c).first() for c in coord_cols])

    # Compute distance from centroid using window function
    df_with_dist = df_sampled.with_columns(
        pl.sum_horizontal([(pl.col(c) - pl.col(c).mean().over('createtime')).pow(2) for c in coord_cols]).sqrt().alias('dist_from_centroid')
    )

    # Aggregate mean distance per window
    window_stats = df_with_dist.group_by('createtime').agg([
        pl.col('dist_from_centroid').mean().alias('mean_dist'),
        pl.len().alias('n_trajectories')
    ]).filter(pl.col('n_trajectories') >= 10).sort('createtime')

    if window_stats.height < 3:
        print("\n=== Trajectory Spread Analysis ===")
        print("Not enough overlapping time points to analyze spread")
        return {
            'time_points': np.array([]),
            'mean_distances': np.array([]),
            'slope': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'spread_ratio': np.nan,
            'is_stationary': None
        }

    mean_distances = window_stats['mean_dist'].to_numpy()
    valid_time_points = np.arange(len(mean_distances))

    if len(valid_time_points) < 3:
        print("\n=== Trajectory Spread Analysis ===")
        print("Not enough overlapping time points to analyze spread")
        return {
            'time_points': valid_time_points,
            'mean_distances': mean_distances,
            'slope': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'spread_ratio': np.nan,
            'is_stationary': None
        }

    # Fit linear trend
    slope, intercept, r_value, p_value, _ = stats.linregress(valid_time_points, mean_distances)

    print("\n=== Trajectory Spread Analysis ===")
    print(f"Mean distance from centroid at t={valid_time_points[0]}: {mean_distances[0]:.4f}")
    print(f"Mean distance from centroid at t={valid_time_points[-1]}: {mean_distances[-1]:.4f}")
    print(f"Linear fit: distance = {slope:.6f} * t + {intercept:.4f}")
    print(f"R² = {r_value**2:.4f}, p-value = {p_value:.2e}")

    spread_ratio = mean_distances[-1] / (mean_distances[0] + 1e-10)
    print(f"Spread ratio (final/initial): {spread_ratio:.2f}x")

    if slope > 0 and p_value < 0.05 and spread_ratio > 1.5:
        print("Result: Trajectories are spreading out over time -> NON-STATIONARY (fanning out)")
        is_stationary = False
    elif slope > 0 and p_value < 0.05:
        print("Result: Slight trajectory spread detected -> Possibly NON-STATIONARY")
        is_stationary = False
    else:
        print("Result: Trajectories maintain consistent spread -> Consistent with STATIONARITY")
        is_stationary = True

    return {
        'time_points': valid_time_points,
        'mean_distances': mean_distances,
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'spread_ratio': spread_ratio,
        'is_stationary': is_stationary
    }


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    print("Loading data...")

    target_path = os.path.join(cfg.trend_path, 'ppca_coords.parquet.zstd')
    target_df = pl.read_parquet(target_path)
    coord_col = [c for c in target_df.columns if c.startswith('coord_')][0]
    n_pca_dims = int(coord_col.split('_')[1][:-1])

    target_df = target_df.filter(pl.col('filter_value') != '')
    target_df = target_df.select(['createtime', 'filter_value', coord_col]) \
        .sort(['filter_value', 'createtime']) \
        .with_columns(((pl.col('createtime') - INITIAL_DATE).dt.total_days() / UNIT_DAYS).alias('t0')) \
        .rename({coord_col: 'x0'})

    target_df = target_df \
        .with_columns([pl.col('x0').arr.get(i).alias(f'x0_{i}') for i in range(n_pca_dims)])

    # Partition into separate trajectories by filter_value
    filter_dfs = target_df.partition_by('filter_value')

    # Filter trajectories by minimum length
    min_length = 100
    trajectories = [
        filter_df.sort('createtime').select([f'x0_{i}' for i in range(n_pca_dims)]).to_numpy()
        for filter_df in filter_dfs
        if filter_df.height > min_length
    ]
    print(f"Loaded {len(trajectories)} trajectories (length > {min_length})")
    print(f"Trajectory lengths: min={min(len(t) for t in trajectories)}, max={max(len(t) for t in trajectories)}")

    # Run stationarity tests
    results = {}

    coord_cols = [f'x0_{i}' for i in range(n_pca_dims)]

    # Test 1: Mean Squared Displacement (MSD) on raw trajectories
    results['msd'] = test_msd(trajectories, label="raw")

    # Test 2: Rolling statistics (uses calendar time alignment)
    results['rolling_stats'] = test_rolling_statistics(target_df, coord_cols)

    # Test 3: ADF and KPSS tests
    results['adf_kpss'] = test_adf_kpss(trajectories)

    # Test 4: Ensemble spread (uses calendar time alignment)
    results['ensemble_spread'] = test_ensemble_spread(target_df, coord_cols)

    # Test 5: MSD on drift-removed (centroid-subtracted) trajectories.
    # If raw MSD grows linearly but residual MSD saturates, the dynamics are
    # "deterministic drift + bounded diffusion" rather than a random walk.
    print("\n--- Computing centroid-removed trajectories ---")
    residual_trajectories, _, _ = compute_drift_removed_trajectories(
        target_df, coord_cols, n_centroid_bins=50, min_length=min_length
    )
    print(f"{len(residual_trajectories)} residual trajectories after centroid removal.")
    results['msd_drift_removed'] = test_msd(residual_trajectories, label="drift-removed")

    # Reconciliation: compare raw vs residual MSD growth.
    raw_saturates = results['msd']['saturates']
    res_saturates = results['msd_drift_removed']['saturates']
    raw_alpha = results['msd']['alpha']
    res_alpha = results['msd_drift_removed']['alpha']
    raw_v = results['msd']['v']
    res_v = results['msd_drift_removed']['v']

    print("\n=== Raw vs drift-removed MSD reconciliation ===")
    print(f"  raw:           α = {raw_alpha:.2f}, |v| = {raw_v:.4f}, saturates = {raw_saturates}")
    print(f"  drift-removed: α = {res_alpha:.2f}, |v| = {res_v:.4f}, saturates = {res_saturates}")
    if not raw_saturates and res_saturates:
        print("  -> Raw MSD grows but residual MSD saturates: dynamics are")
        print("     deterministic drift + bounded diffusion (cloud translates rigidly).")
    elif raw_saturates and res_saturates:
        print("  -> Both saturate: bounded dynamics with no global drift.")
    elif not raw_saturates and not res_saturates:
        print("  -> Neither saturates: dispersion grows even after removing global drift.")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    stationarity_votes = []
    for test_name, test_result in results.items():
        is_stat = test_result.get('is_stationary')
        status = "STATIONARY" if is_stat else ("NON-STATIONARY" if is_stat is False else "INCONCLUSIVE / N/A")
        print(f"  {test_name}: {status}")
        if is_stat is not None:
            stationarity_votes.append(is_stat)

    if stationarity_votes:
        stationary_fraction = np.mean(stationarity_votes)
        if stationary_fraction > 0.5:
            print(f"\nOverall: System appears STATIONARY ({stationary_fraction:.0%} of tests)")
        else:
            print(f"\nOverall: System appears NON-STATIONARY ({1-stationary_fraction:.0%} of tests)")


if __name__ == '__main__':
    main()
