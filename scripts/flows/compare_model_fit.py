import polars as pl
import wandb

def main():
    api = wandb.Api(timeout=120)

    project_name = 'potential_landscape_training'
    runs = api.runs(project_name)
    runs_list = list(runs)

    # Collect run data
    run_data = []
    for run in runs_list:
        config = run.config
        split_type = config.get('split_type', None)

        # Get final losses from summary
        summary = run.summary
        train_loss = summary.get('train_loss', None)
        valid_loss = summary.get('valid_loss', None)

        run_data.append({
            'run_id': run.id,
            'run_name': run.name,
            'split_type': split_type,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'state': run.state,
        })

    if not run_data:
        print("No runs found in project")
        return

    df = pl.DataFrame(run_data)

    print("=" * 60)
    print("Model Fit Comparison by Split Type")
    print("=" * 60)
    print()

    # Filter to only completed runs with valid losses and split_type
    df_valid = df.filter(
        (pl.col('train_loss').is_not_null()) &
        (pl.col('valid_loss').is_not_null()) &
        (pl.col('split_type').is_not_null())
    )

    print(f"Total runs found: {len(df)}")
    print(f"Runs with split_type and valid losses: {len(df_valid)}")
    print()

    # Group by split type and compute statistics
    split_types = ['random', 'filter_value', 'time']

    print("-" * 60)
    print(f"{'Split Type':<15} {'Train Loss':<15} {'Val Loss':<15} {'Gap':<15}")
    print("-" * 60)

    for split_type in split_types:
        split_df = df_valid.filter(pl.col('split_type') == split_type)

        if len(split_df) == 0:
            print(f"{split_type:<15} {'No runs':<15} {'No runs':<15} {'N/A':<15}")
            continue

        train_loss = split_df['train_loss'].mean()
        valid_loss = split_df['valid_loss'].mean()
        gap = valid_loss - train_loss if train_loss is not None and valid_loss is not None else None

        train_str = f"{train_loss:.6f}" if train_loss is not None else "N/A"
        val_str = f"{valid_loss:.6f}" if valid_loss is not None else "N/A"
        gap_str = f"{gap:.6f}" if gap is not None else "N/A"

        print(f"{split_type:<15} {train_str:<15} {val_str:<15} {gap_str:<15}")

        if len(split_df) > 1:
            train_std = split_df['train_loss'].std()
            val_std = split_df['valid_loss'].std()
            print(f"{'  (std)':<15} {f'±{train_std:.6f}':<15} {f'±{val_std:.6f}':<15}")

    print("-" * 60)
    print()

    # Print individual run details
    print("Individual Run Details:")
    print("-" * 60)

    for split_type in split_types:
        split_df = df_valid.filter(pl.col('split_type') == split_type)

        if len(split_df) == 0:
            continue

        print(f"\n{split_type.upper()} split:")
        for row in split_df.iter_rows(named=True):
            train_loss = row['train_loss']
            valid_loss = row['valid_loss']
            gap = valid_loss - train_loss if train_loss and valid_loss else None
            pct_gap = (gap / train_loss * 100) if train_loss and gap is not None else None

            print(f"  Run: {row['run_name']}")
            print(f"    Train Loss: {train_loss:.6f}" if train_loss else "    Train Loss: N/A")
            print(f"    Val Loss:   {valid_loss:.6f}" if valid_loss else "    Val Loss: N/A")
            print(f"    Gap:        {gap:.6f}" if gap else "    Gap: N/A")
            print(f"    % Gap:      {pct_gap:.2f}%" if pct_gap else "    % Gap: N/A")

if __name__ == '__main__':
    main()
