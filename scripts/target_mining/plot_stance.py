import os

import polars as pl

def main():
    stance_path = './data/stance_targets/doc_stance'
    df = None
    for filename in os.listdir(stance_path):
        file_df = pl.read_parquet(os.path.join(stance_path, filename))
        df = pl.concat([df, file_df], how='diagonal_relaxed') if df is not None else file_df

    pass

if __name__ == '__main__':
    main()