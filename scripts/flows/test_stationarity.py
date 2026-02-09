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
   - Measures divergence of trajectories from ensemble centroid over time
   - Related to Lyapunov exponent analysis for chaotic systems
   Reference: Wolf, A. et al. (1985). "Determining Lyapunov exponents from a
   time series." Physica D, 16, 285-317.
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


def test_msd(trajectories: list[np.ndarray], max_lag: int = None) -> dict:
    """Compute Mean Squared Displacement (MSD) to test for diffusive behavior.

    The MSD measures the average squared displacement as a function of lag time τ:
        MSD(τ) = <|r(t+τ) - r(t)|²>

    For different types of motion:
    - Normal diffusion (random walk): MSD ∝ τ (linear growth)
    - Stationary/confined: MSD plateaus at large τ
    - Superdiffusion: MSD ∝ τ^α with α > 1
    - Subdiffusion: MSD ∝ τ^α with α < 1

    This implementation uses time-averaged MSD for each trajectory, then
    averages across the ensemble.

    Reference:
        Einstein, A. (1905). Annalen der Physik, 17, 549-560.
        Qian, H. et al. (1991). Biophys J, 60(4), 910-921.

    Args:
        trajectories: List of trajectory arrays, each shape (T, D)
        max_lag: Maximum lag time to compute (default: min trajectory length // 2)

    Returns:
        Dictionary with MSD values, linear fit parameters, and stationarity assessment
    """
    min_len = min(len(t) for t in trajectories)
    if max_lag is None:
        max_lag = min_len // 2

    # Sample lag times (use ~20 points for efficiency)
    lags = np.arange(1, max_lag + 1, max(1, max_lag // 20))
    msd_values = []
    msd_stderr = []

    for lag in lags:
        # Time-averaged MSD: average over all time origins t
        # MSD(τ) = <|r(t+τ) - r(t)|²>_t,ensemble
        squared_displacements = []

        for traj in trajectories:
            T = len(traj)
            if T > lag:
                # Compute displacements for all valid time origins
                for t0 in range(T - lag):
                    disp = traj[t0 + lag] - traj[t0]
                    sq_disp = np.sum(disp ** 2)  # Sum over dimensions
                    squared_displacements.append(sq_disp)

        if squared_displacements:
            msd = np.mean(squared_displacements)
            stderr = np.std(squared_displacements) / np.sqrt(len(squared_displacements))
            msd_values.append(msd)
            msd_stderr.append(stderr)
        else:
            msd_values.append(np.nan)
            msd_stderr.append(np.nan)

    msd_values = np.array(msd_values)
    msd_stderr = np.array(msd_stderr)
    valid_mask = ~np.isnan(msd_values)
    lags_valid = lags[valid_mask]
    msd_valid = msd_values[valid_mask]

    # Fit linear regression: MSD = D * lag + b
    # For random walk: D > 0 (diffusion coefficient), R² high
    # For stationary: D ≈ 0 or MSD plateaus
    slope, intercept, r_value, p_value, std_err = stats.linregress(lags_valid, msd_valid)

    # Also fit power law: MSD ∝ τ^α to get anomalous exponent
    # log(MSD) = α * log(τ) + const
    log_lags = np.log(lags_valid)
    log_msd = np.log(msd_valid + 1e-10)
    alpha, log_const, r_alpha, p_alpha, _ = stats.linregress(log_lags, log_msd)

    print("\n=== Mean Squared Displacement (MSD) Analysis ===")
    print(f"MSD at lag 1: {msd_valid[0]:.4f}")
    print(f"MSD at lag {lags_valid[-1]}: {msd_valid[-1]:.4f}")
    print(f"Linear fit: MSD = {slope:.6f} * τ + {intercept:.4f}")
    print(f"  R² = {r_value**2:.4f}, p-value = {p_value:.2e}")
    print(f"Power law fit: MSD ∝ τ^{alpha:.2f}")
    print(f"  R² = {r_alpha**2:.4f}")

    # Interpretation based on anomalous exponent α
    if alpha > 0.9 and r_alpha**2 > 0.9:
        if alpha < 1.1:
            print("Result: MSD ∝ τ (α ≈ 1) -> Normal diffusion / random walk -> NON-STATIONARY")
        else:
            print(f"Result: MSD ∝ τ^{alpha:.1f} (α > 1) -> Superdiffusion -> NON-STATIONARY")
        is_stationary = False
    elif alpha < 0.5 and r_alpha**2 > 0.8:
        print(f"Result: MSD ∝ τ^{alpha:.1f} (α < 0.5) -> Subdiffusion/confined -> Likely STATIONARY")
        is_stationary = True
    elif slope > 0 and p_value < 0.05:
        print("Result: MSD grows with lag time -> NON-STATIONARY")
        is_stationary = False
    else:
        print("Result: MSD does not grow significantly -> Consistent with STATIONARITY")
        is_stationary = True

    return {
        'lags': lags_valid,
        'msd': msd_valid,
        'msd_stderr': msd_stderr[valid_mask],
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'alpha': alpha,
        'alpha_r_squared': r_alpha**2,
        'is_stationary': is_stationary
    }


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
    # Compute window duration from time range
    min_time = df['createtime'].min()
    max_time = df['createtime'].max()
    time_range_seconds = (max_time - min_time).total_seconds()
    window_seconds = int(time_range_seconds / n_windows)
    window_duration = f"{window_seconds}s"

    # Use group_by_dynamic for time-based windowing
    window_stats = df.sort('createtime').group_by_dynamic(
        'createtime',
        every=window_duration
    ).agg(
        [pl.col(c).mean().alias(f'{c}_mean') for c in coord_cols] +
        [pl.col(c).var().alias(f'{c}_var') for c in coord_cols]
    )

    window_means = window_stats.select([f'{c}_mean' for c in coord_cols]).to_numpy()
    window_vars = window_stats.select([f'{c}_var' for c in coord_cols]).to_numpy()

    # Test if means differ across windows (ANOVA-like)
    mean_variation = np.std(window_means, axis=0)
    var_variation = np.std(window_vars, axis=0)

    # Compare first and last window
    mean_drift = window_means[-1] - window_means[0]
    var_change = window_vars[-1] - window_vars[0]

    print("\n=== Rolling Statistics Analysis ===")
    print(f"Window duration: {window_seconds / 86400:.1f} days, {len(window_stats)} windows")
    print(f"Mean drift (last - first window): {np.mean(np.abs(mean_drift)):.4f} (avg across dims)")
    print(f"Variance change (last - first window): {np.mean(var_change):.4f} (avg across dims)")
    print(f"Std of window means: {np.mean(mean_variation):.4f} (avg across dims)")
    print(f"Std of window variances: {np.mean(var_variation):.4f} (avg across dims)")

    # Relative changes
    rel_mean_drift = np.mean(np.abs(mean_drift)) / (np.mean(np.abs(window_means[0])) + 1e-10)
    rel_var_change = np.mean(var_change) / (np.mean(window_vars[0]) + 1e-10)

    print(f"Relative mean drift: {rel_mean_drift:.2%}")
    print(f"Relative variance change: {rel_var_change:.2%}")

    if rel_var_change > 0.5:
        print("Result: Variance increases substantially over time -> NON-STATIONARY")
        is_stationary = False
    elif rel_mean_drift > 0.5:
        print("Result: Mean drifts substantially over time -> NON-STATIONARY")
        is_stationary = False
    else:
        print("Result: Statistics relatively stable -> Consistent with STATIONARITY")
        is_stationary = True

    return {
        'window_means': window_means,
        'window_vars': window_vars,
        'mean_drift': mean_drift,
        'var_change': var_change,
        'rel_mean_drift': rel_mean_drift,
        'rel_var_change': rel_var_change,
        'is_stationary': is_stationary
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

    # Interpretation based on majority of dimensions
    if adf_majority_rejects and not kpss_majority_rejects:
        print("\nResult: ADF rejects unit root, KPSS doesn't reject stationarity -> STATIONARY")
        is_stationary = True
    elif not adf_majority_rejects and kpss_majority_rejects:
        print("\nResult: ADF doesn't reject unit root, KPSS rejects stationarity -> NON-STATIONARY")
        is_stationary = False
    elif not adf_majority_rejects and not kpss_majority_rejects:
        print("\nResult: Both tests inconclusive -> Possibly TREND-STATIONARY")
        is_stationary = None
    else:
        print("\nResult: Conflicting results -> INCONCLUSIVE")
        is_stationary = None

    return {
        'adf_combined_by_dim': adf_combined_by_dim,
        'kpss_combined_by_dim': kpss_combined_by_dim,
        'adf_rejects_by_dim': adf_rejects_by_dim,
        'kpss_rejects_by_dim': kpss_rejects_by_dim,
        'is_stationary': is_stationary
    }


def test_ensemble_spread(df: pl.DataFrame, coord_cols: list[str], n_windows: int = 20) -> dict:
    """Test if ensemble of trajectories spreads out (fans out) over time.

    Computes the average distance of trajectories from the ensemble centroid
    at each time point. If this increases, trajectories are diverging, which
    indicates non-stationary dynamics.

    This is related to Lyapunov exponent analysis: positive Lyapunov exponents
    indicate exponential divergence of nearby trajectories (chaos), while for
    stationary systems the spread should remain bounded.

    Note: This measures ensemble spread, not individual trajectory divergence.
    Trajectories can drift together (correlated motion) without increasing
    spread from the centroid.

    Reference:
        Wolf, A., Swift, J.B., Swinney, H.L., Vastano, J.A. (1985).
        "Determining Lyapunov exponents from a time series."
        Physica D, 16(3), 285-317.

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

    n_pca_dims = 21

    target_path = os.path.join(cfg.trend_path, 'pca_coords.parquet.zstd')
    target_df = pl.read_parquet(target_path, columns=['createtime', 'filter_value', f'coord_{n_pca_dims}d'])

    target_df = target_df.filter(pl.col('filter_value') != '')
    target_df = target_df.select(['createtime', 'filter_value', f'coord_{n_pca_dims}d']) \
        .sort(['filter_value', 'createtime']) \
        .with_columns(((pl.col('createtime') - INITIAL_DATE).dt.total_days() / UNIT_DAYS).alias('t0')) \
        .rename({f'coord_{n_pca_dims}d': 'x0'})

    target_df = target_df \
        .with_columns([pl.col('x0').arr.get(i).alias(f'x0_{i}') for i in range(n_pca_dims)]) \
        # .with_columns([pl.col(f'x0_{i}').rolling_mean(cfg.rolling_mean_window).over('filter_value') for i in range(n_pca_dims)]) \
        # .drop_nulls([f'x0_{i}' for i in range(n_pca_dims)])

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

    # Test 1: Mean Squared Displacement (MSD)
    results['msd'] = test_msd(trajectories)

    # Test 2: Rolling statistics (uses calendar time alignment)
    results['rolling_stats'] = test_rolling_statistics(target_df, coord_cols)

    # Test 3: ADF and KPSS tests
    results['adf_kpss'] = test_adf_kpss(trajectories)

    # Test 4: Ensemble spread (uses calendar time alignment)
    results['ensemble_spread'] = test_ensemble_spread(target_df, coord_cols)

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    stationarity_votes = []
    for test_name, test_result in results.items():
        is_stat = test_result.get('is_stationary')
        status = "STATIONARY" if is_stat else ("NON-STATIONARY" if is_stat is False else "INCONCLUSIVE")
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
