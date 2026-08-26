"""Checks for the nested trajectory x time split.

Run as: python test_splits.py
"""

import datetime

import numpy as np
import polars as pl

import splits


def make_pairs(n_traj=200, n_per=40, start=datetime.datetime(2020, 1, 1)):
    rows = []
    for i in range(n_traj):
        for k in range(n_per):
            t = start + datetime.timedelta(days=20 * k)
            rows.append((f'seed{i:04d}', t, t + datetime.timedelta(days=20)))
    return pl.DataFrame(rows, schema=['filter_value', 'createtime', 'future_createtime'],
                        orient='row')


def test_fractions():
    spec = splits.SplitSpec(holdout_days=365)
    ids = [f'seed{i:05d}' for i in range(20_000)]
    got = splits.assign_trajectory_split(ids, spec)
    share = got.group_by('traj_split').agg((pl.len() / len(ids)).alias('f'))
    got_f = dict(zip(share['traj_split'], share['f']))
    for name, want in (('train', 0.70), ('val', 0.10), ('test', 0.20)):
        assert abs(got_f[name] - want) < 0.02, (name, got_f)
    print('fractions', {k: round(v, 4) for k, v in sorted(got_f.items())})


def test_assignment_is_stable_under_filtering():
    """A platform filter must not reshuffle who is in which split."""
    spec = splits.SplitSpec(holdout_days=365)
    ids = [f'seed{i:04d}' for i in range(2_000)]
    full = splits.assign_trajectory_split(ids, spec)
    subset = splits.assign_trajectory_split(ids[::3], spec)
    joined = subset.join(full, on='filter_value', how='inner', suffix='_full')
    assert (joined['traj_split'] == joined['traj_split_full']).all()
    print('assignment stable under subsetting')


def test_seed_changes_assignment():
    ids = [f'seed{i:04d}' for i in range(2_000)]
    a = splits.assign_trajectory_split(ids, splits.SplitSpec(holdout_days=365, seed=1))
    b = splits.assign_trajectory_split(ids, splits.SplitSpec(holdout_days=365, seed=2))
    frac = float((a['traj_split'] == b['traj_split']).mean())
    assert 0.4 < frac < 0.8, frac          # agreement by chance is 0.54
    print(f'a different seed reassigns {1 - frac:.0%} of trajectories')


def test_labels_and_leakage():
    pairs = make_pairs()
    for holdout in splits.HOLDOUT_DAYS:
        spec = splits.SplitSpec(holdout_days=holdout)
        labelled = splits.label_pairs(pairs, spec)
        train = splits.training_rows(labelled)
        assert len(train) > 0, holdout

        for traj in splits.TRAJ_SPLITS:
            for time in splits.TIME_SPLITS:
                cell = splits.select(labelled, traj, time)
                splits.check_leakage(train, cell, splits.scenario_name(traj, time))

        # every training target must predate the holdout window
        cutoff = splits.time_cutoff(pairs['future_createtime'], spec)
        assert train['future_createtime'].max() < cutoff
        n_out = len(splits.select(labelled, splits.TRAJ_SPLITS, 'out'))
        print(f'holdout {holdout:4d}d  cutoff {cutoff:%Y-%m-%d}  '
              f'train pairs {len(train):6d}  out-of-time pairs {n_out:6d}')


def test_leakage_check_actually_fires():
    """A deliberately contaminated evaluation set must be rejected."""
    spec = splits.SplitSpec(holdout_days=365)
    labelled = splits.label_pairs(make_pairs(), spec)
    train = splits.training_rows(labelled)
    bad = splits.select(labelled, 'test', 'out').with_columns(
        pl.lit(train['filter_value'][0]).alias('filter_value'))
    try:
        splits.check_leakage(train, bad, 'test_out')
    except AssertionError as e:
        print('contaminated set rejected:', str(e).split(',')[0])
        return
    raise AssertionError('check_leakage passed a trajectory that is in train')


def test_time_split_keys_on_target():
    """A pair starting before the cutoff but landing after it is out-of-time."""
    spec = splits.SplitSpec(holdout_days=365)
    pairs = make_pairs()
    labelled = splits.label_pairs(pairs, spec)
    cutoff = splits.time_cutoff(pairs['future_createtime'], spec)
    straddling = labelled.filter((pl.col('createtime') < cutoff)
                                 & (pl.col('future_createtime') >= cutoff))
    assert len(straddling) > 0
    assert (straddling['time_split'] == 'out').all()
    print(f'{len(straddling)} straddling pairs all labelled out-of-time')


if __name__ == '__main__':
    for fn in (test_fractions, test_assignment_is_stable_under_filtering,
               test_seed_changes_assignment, test_labels_and_leakage,
               test_leakage_check_actually_fires, test_time_split_keys_on_target):
        fn()
    print('\nall split checks passed')
