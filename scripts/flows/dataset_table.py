import os

import polars as pl

def main():
    seed_data_path = './data/stance_targets/noun_phrase_bkrr_trends'
    platform_data_path = './data/stance_targets/platform_handle_noun_phrase_bkrr_trends'
    output_path = './out/dataset_table.tex'

    seed_df = pl.read_parquet(os.path.join(seed_data_path, 'loaded_trends.parquet.zstd'))
    platform_df = pl.read_parquet(os.path.join(platform_data_path, 'loaded_trends.parquet.zstd'))

    total_num_posts = seed_df['volume'].sum()
    total_num_users = seed_df['filter_value'].n_unique()

    platforms = platform_df['filter_value'].unique().str.split('-').list.get(1).unique().sort()

    rows = []
    for platform in platforms:
        platform_seed_df = platform_df.filter(pl.col('filter_value').str.contains(f'-{platform}-'))
        platform_num_posts = platform_seed_df['volume'].sum()
        platform_num_users = platform_seed_df['filter_value'].n_unique()
        rows.append((platform.capitalize(), platform_num_posts, platform_num_users))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("\\begin{tabular}{lrr}\n")
        f.write("\\toprule\n")
        f.write("Platform & Num Posts & Num Users \\\\\n")
        f.write("\\midrule\n")

        for platform, num_posts, num_users in rows:
            f.write(f"{platform} & {num_posts:,} & {num_users:,} \\\\\n")

        f.write("\\midrule\n")
        f.write(f"Total & {total_num_posts:,} & {total_num_users:,} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Dataset statistics by platform.}\n")
        f.write("\\label{tab:dataset}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table written to {output_path}")

if __name__ == '__main__':
    main()