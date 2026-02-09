import os

import hydra
import numpy as np
import polars as pl
from tqdm import tqdm

from deeptime.clustering import RegularSpace
from deeptime.decomposition import VAMP
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM, BayesianMSM
from deeptime.util.validation import implied_timescales

from nn_potential import INITIAL_DATE, UNIT_DAYS

def find_converged_lagtime(
    its_data,
    rel_tol: float = 0.1,
    n_timescales: int = 3
) -> int:
    """Find the minimum lagtime where implied timescales have converged.

    Convergence is defined as the point where the relative change in the
    slowest n_timescales is below rel_tol for consecutive lagtimes.

    Args:
        its_data: ImpliedTimescales object from deeptime
        rel_tol: Relative tolerance for convergence (default 0.1 = 10%)
        n_timescales: Number of slowest timescales to check for convergence

    Returns:
        The minimum converged lagtime
    """
    lagtimes = its_data.lagtimes
    n_processes = its_data.max_n_processes

    # Build timescales matrix: shape (n_lagtimes, n_processes)
    # timescales_for_process returns shape (n_lagtimes,)
    timescales = np.column_stack([
        its_data.timescales_for_process(p) for p in range(n_processes)
    ])

    print(f"Implied timescales shape: {timescales.shape}")
    print(f"Lagtimes: {lagtimes}")

    # Check convergence for each lagtime
    n_ts = min(n_timescales, timescales.shape[1])

    for i in range(1, len(lagtimes)):
        # Compare with previous lagtime
        prev_ts = timescales[i-1, :n_ts]
        curr_ts = timescales[i, :n_ts]

        # Calculate relative change
        rel_change = np.abs(curr_ts - prev_ts) / (np.abs(prev_ts) + 1e-10)
        max_rel_change = np.max(rel_change)

        print(f"  Lagtime {lagtimes[i]}: max relative change = {max_rel_change:.4f}")

        if max_rel_change < rel_tol:
            print(f"Convergence detected at lagtime {lagtimes[i]} (relative change {max_rel_change:.4f} < {rel_tol})")
            return int(lagtimes[i])

    # If no convergence, return the lagtime with smallest relative change
    print("Warning: No clear convergence detected, using lagtime with smallest change")
    rel_changes = []
    for i in range(1, len(lagtimes)):
        prev_ts = timescales[i-1, :n_ts]
        curr_ts = timescales[i, :n_ts]
        rel_change = np.max(np.abs(curr_ts - prev_ts) / (np.abs(prev_ts) + 1e-10))
        rel_changes.append(rel_change)

    best_idx = np.argmin(rel_changes) + 1
    return int(lagtimes[best_idx])


def test_equilibrium(trajectories: list[np.ndarray], chosen_lag: int, n_clusters: int = 10):
    # Step 1: Cluster in VAMP space (accepts list of trajectories)
    cluster = RegularSpace(
        dmin=10.0,
        max_centers=50,
        n_jobs=8
    )
    clustering_model = cluster.fit(trajectories).fetch_model()
    discrete_trajs = [clustering_model.transform(traj) for traj in trajectories]

    # 1. Fit REVERSIBLE MSM (assumes equilibrium)
    msm_reversible = MaximumLikelihoodMSM(
        lagtime=chosen_lag,
        reversible=True  # enforces detailed balance
    )
    model_rev = msm_reversible.fit(discrete_trajs).fetch_model()

    # 2. Fit NON-REVERSIBLE MSM (no equilibrium assumption)
    msm_nonreversible = MaximumLikelihoodMSM(
        lagtime=chosen_lag,
        reversible=False  # no detailed balance constraint
    )
    model_nonrev = msm_nonreversible.fit(discrete_trajs).fetch_model()

    # 3. Compare VAMP-2 scores
    score_rev = model_rev.score(discrete_trajs, r='VAMP2')
    score_nonrev = model_nonrev.score(discrete_trajs, r='VAMP2')

    print(f"Reversible MSM VAMP-2 score: {score_rev}")
    print(f"Non-reversible MSM VAMP-2 score: {score_nonrev}")

    # Interpretation:
    if score_rev >= score_nonrev * 0.95:  # within ~5%
        print("System appears to be at equilibrium (reversible)")
    else:
        print("System NOT at equilibrium (use non-reversible model)")

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
        .with_columns([pl.col(f'x0_{i}').rolling_mean(cfg.rolling_mean_window).over('filter_value') for i in range(n_pca_dims)]) \
        .drop_nulls([f'x0_{i}' for i in range(n_pca_dims)])

    # Partition into separate trajectories by filter_value
    filter_dfs = target_df.partition_by('filter_value')

    # Use maximum lagtime to filter trajectories (ensures all lagtimes have same trajectories)
    lagtimes = [50, 75, 100, 150, 200, 250, 300]
    max_lagtime = max(lagtimes)
    trajectories_data = [
        filter_df.sort('createtime').select([f'x0_{i}' for i in range(n_pca_dims)]).to_numpy()
        for filter_df in filter_dfs
        if filter_df.height > max_lagtime
    ]
    print(f"Loaded {len(trajectories_data)} trajectories (length > {max_lagtime})")

    # Step 1: Cluster trajectories to get discrete states for implied timescale computation
    print("\nClustering trajectories...")
    cluster = RegularSpace(
        dmin=1.0,
        max_centers=100,
        n_jobs=8
    )
    clustering_model = cluster.fit(trajectories_data).fetch_model()
    discrete_trajs = [clustering_model.transform(traj) for traj in trajectories_data]
    print(f"Created {clustering_model.n_clusters} clusters")

    # Step 2: Fit MSMs at each lagtime and compute implied timescales
    print("\nFitting MSMs at each lagtime...")
    models = []
    for lag in lagtimes:
        print(f"  Fitting MSM at lagtime {lag}...")
        counts = TransitionCountEstimator(lagtime=lag, count_mode='effective').fit_fetch(discrete_trajs)
        msm = BayesianMSM(n_samples=50).fit_fetch(counts)
        models.append(msm)

    print("\nComputing implied timescales...")
    its = implied_timescales(models, n_its=10)

    # Print implied timescales for inspection
    print("\nImplied timescales by lagtime:")
    n_show = min(5, its.max_n_processes)
    for i, lag in enumerate(its.lagtimes):
        ts_vals = [its.timescales_for_process(p)[i] for p in range(n_show)]
        ts_str = ", ".join([f"{t:.1f}" for t in ts_vals])
        print(f"  Lagtime {lag}: [{ts_str}, ...]")

    # Step 3: Find converged lagtime
    print("\nFinding converged lagtime...")
    best_lagtime = find_converged_lagtime(its, rel_tol=0.1, n_timescales=3)
    print(f"\nSelected lagtime: {best_lagtime}")

    # Step 4: Apply VAMP projection at the selected lagtime
    print(f"\nApplying VAMP projection at lagtime {best_lagtime}...")
    vamp = VAMP(
        lagtime=best_lagtime,
        var_cutoff=0.6,
    )
    vamp_model = vamp.fit(trajectories_data).fetch_model()
    vamp_trajectories = [vamp_model.transform(traj) for traj in tqdm(trajectories_data, desc="Transforming trajectories")]

    print(f"VAMP singular values: {vamp_model.singular_values}")
    print(f"VAMP output dimensions: {vamp_trajectories[0].shape[1]}")

    # Step 5: Test equilibrium using VAMP-projected trajectories
    print("\nTesting equilibrium...")
    test_equilibrium(vamp_trajectories, best_lagtime)

if __name__ == '__main__':
    main()