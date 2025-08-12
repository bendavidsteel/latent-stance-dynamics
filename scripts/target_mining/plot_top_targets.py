import os
import re
import matplotlib.pyplot as plt
import polars as pl
import sentence_transformers
import umap
import numpy as np
from matplotlib import cm

from matplotlib.patches import Circle


def main():
    data_path = './data/stance_targets'
    df = pl.DataFrame()
    for filename in os.listdir(data_path):
        if re.search('\d{4}_\d{1,2}_doc_targets_with_stance.parquet.zstd', filename):
            year = int(filename.split('_')[0])
            month = int(filename.split('_')[1])
            # TODO remove
            if year == 2024 and month < 40:
                continue
            month_stance = pl.read_parquet(
                os.path.join(data_path, filename),
                columns=['id', 'Document', 'createtime', 'seed_id', 'Targets', 'Polarities']
            )
            df = pl.concat([df, month_stance], how='diagonal_relaxed')
    
    df = df.filter(pl.col('Targets').list.len() == pl.col('Polarities').list.len())
    
    fig = stancemining.plot.plot_semantic_map(df)

    fig.savefig('./figs/target_map.png', dpi=300, bbox_inches='tight')
    print(f"Figure saved to './figs/target_map.png'")

if __name__ == '__main__':
    main()