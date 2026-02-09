import os
import re
import matplotlib.pyplot as plt
import polars as pl
import sentence_transformers
import umap
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Circle
from adjustText import adjust_text

def main():
    data_path = './data/twitter/doc_stance'
    df = pl.DataFrame()
    for filename in os.listdir(data_path):
        if re.search('\d{4}_\d{1,2}_doc_targets_with_stance.parquet.zstd', filename):
            year = int(filename.split('_')[0])
            month = int(filename.split('_')[1])
            month_stance = pl.read_parquet(
                os.path.join(data_path, filename),
                columns=['id', 'Document', 'createtime', 'seed', 'Targets', 'Polarities']
            )
            df = pl.concat([df, month_stance], how='diagonal_relaxed')
    
    first_target_only = False
    if first_target_only:
        # Only keep the first target for each document
        df = df.with_columns([
            pl.col('Targets').list.slice(0, 1).alias('Targets'),
            pl.col('Polarities').list.slice(0, 1).alias('Polarities')
        ])

    doc_target_df = df.explode(['Targets', 'Polarities']).drop_nulls('Targets').rename({'Targets': 'Target', 'Polarities': 'Stance'})
    
    target_df = doc_target_df.group_by('Target').agg([
        pl.col('Stance').mean().alias('stance_mean'),
        pl.col('Stance').count().alias('count')
    ]).sort('count', descending=True).head(30)
    
    encoder = sentence_transformers.SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = encoder.encode(target_df['Target'].to_list())
    
    umap_model = umap.UMAP(spread=0.3)
    coordinates = umap_model.fit_transform(embeddings)
    
    # Create figure with larger size for better visibility
    figsize = (8, 4)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define custom green to red color map and normalization
    colors = [(0.7, 0.0, 0.0), (0.9, 0.9, 0.9), (0.0, 0.7, 0.0)]  # green, light gray, red
    cmap = LinearSegmentedColormap.from_list("red_to_green", colors)
    norm = Normalize(vmin=-1, vmax=1)  # Assuming stance ranges from -1 to 1
    
    # Define size scaling based on count
    count_values = target_df['count'].to_numpy()
    size_min, size_max = 100, 500  # Min and max circle sizes
    sizes = size_min + (size_max - size_min) * (count_values - count_values.min()) / (count_values.max() - count_values.min() + 1e-10)
    
    # Create scatter plot
    scatter = ax.scatter(
        coordinates[:, 0], 
        coordinates[:, 1],
        c=target_df['stance_mean'].to_numpy(),
        s=sizes,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        edgecolors='black'
    )
    
    # Prepare labels for adjustText
    targets = target_df['Target'].to_list()
    texts = []
    for i, (x, y) in enumerate(coordinates):
        # Truncate long target names
        target_label = targets[i]
            
        # Create text objects
        text = ax.annotate(target_label, (x, y), 
                      fontsize=12,
                      ha='center', 
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
        texts.append(text)
    
    # Use adjustText to prevent label overlap
    adjust_text(texts, 
                force_points=0.2, 
                force_text=0.5,
                expand_points=(1.5, 1.5),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    # Add colorbar for stance
    if figsize[0] > figsize[1]:
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.01, aspect=40, location='right')
    else:
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.01, aspect=40, location='bottom')
    cbar.set_label('Stance Mean (Against to Favor)')
    
    # Add labels to colorbar extremes
    # cbar.ax.text(0, -0.05, 'Negative (-1)', ha='left', va='top', transform=cbar.ax.transAxes, color='darkred')
    # cbar.ax.text(1, -0.05, 'Positive (1)', ha='right', va='top', transform=cbar.ax.transAxes, color='darkgreen')
    
    # Add legend for sizes (3 representative sizes with even numbers)
    # Find min and max counts, and round to even numbers
    min_count = 10 ** (len(str(int(count_values.min()))))
    max_count = 10 ** (len(str(int(count_values.max()))))
    
    size_legend_values = [10000, 50000]
    size_legend_labels = [f'Count: {int(val)}' for val in size_legend_values]
    
    # Calculate the actual sizes that would be used in the plot for these counts
    size_legend_sizes = size_min + (size_max - size_min) * (
        np.array(size_legend_values) - count_values.min()) / (count_values.max() - count_values.min() + 1e-10)
    
    # Create dummy scatter points for legend with correct sizes
    legend_elements = []
    for size, label in zip(size_legend_sizes, size_legend_labels):
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', label=label,
                      markerfacecolor='gray', markersize=np.sqrt(size))
        )
    
    ax.legend(handles=legend_elements, title="Target Frequency", loc="upper right", bbox_to_anchor=(1.0, 1.1))
    
    # Set labels and title
    # ax.set_title('Semantic Map of Targets with Stance and Frequency', fontsize=16)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')

    if figsize[0] < figsize[1]:
        ax.xaxis.set_label_position('top') 
    
    # Remove ticks as they don't have meaningful values
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Save figure
    fig.tight_layout()
    fig_path = './figs/twitter/target_map.png'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to '{fig_path}'")

if __name__ == '__main__':
    main()