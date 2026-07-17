import datetime
import os
from typing import List, Tuple

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.patches import Polygon
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from deduplicate_targets import process_data, remove_low_count_targets

# Example usage with enhanced features
if __name__ == "__main__":
    df, target_count_df, unique_platforms, unique_parties, unique_main_types = process_data()
    print(f"Processed data for {len(target_count_df)} targets")

    # remove targets with low counts
    df, target_count_df = remove_low_count_targets(df, target_count_df, 50)
    
    # Create the enhanced continuous stream diagram
    fig = plot_stream_map(
        df,
        figsize=(20, 14),
        max_stream_width=4.0,
        min_transition_count=4  # Only show substantial focus shifts
    )
    
    # To save the figure
    fig.savefig('./figs/enhanced_continuous_stance_streams.png', dpi=300, bbox_inches='tight')
