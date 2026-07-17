import os
import shutil

import polars as pl
from tqdm import tqdm

def main():
    trends_path = './data/stance_targets/new_noun_phrase_kernelreg_trends'
    platform_handle_path = './data/stance_targets/platform_handle_trends'
    os.makedirs(platform_handle_path, exist_ok=True)

    filenames = [f for f in os.listdir(trends_path) if 'trends.parquet' in f]

    for filename in tqdm(filenames):
        src_path = os.path.join(trends_path, filename)
        dst_path = os.path.join(platform_handle_path, filename)
        df = pl.scan_parquet(src_path)
        df = df.filter(pl.col('filter_type') == 'PlatformHandleID')
        df = df.collect()
        if df.height == 0:
            continue
        df.write_parquet(dst_path, compression='zstd')

if __name__ == "__main__":
    main()