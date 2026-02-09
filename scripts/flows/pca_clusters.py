import collections
import os

from adjustText import adjust_text
import bertopic
from bertopic.backend._utils import select_backend
from bertopic.representation import KeyBERTInspired, TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer
import hydra
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import numpy as np
import polars as pl
import skimage.feature
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline, AutoTokenizer

from pca_density import load_df, pivot_and_impute, do_pca, create_kde_background, plot_streamplot

def load_text_df(cfg, columns=['id', 'createtime', 'seed', 'Document']):
    dir_path = cfg.base_stance_path
    file_paths = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) if file_name.endswith('.parquet.zstd')]
    df = pl.read_parquet(file_paths, columns=columns)
    return df

def get_prompt(tokenizer):
    
    # System prompt describes information given to all conversations

    no_think_msg = '/no_think' if 'qwen' in tokenizer.name_or_path.lower() else ''

    messages = [
        {'role': 'system', 'content': 'You are a helpful, respectful and honest assistant for labeling topics.' + no_think_msg},
        {
            'role': 'user', 
            'content': (
                "I have a topic that contains the following documents:\n"
                "- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.\n"
                "- Meat, but especially beef, is the word food in terms of emissions.\n"
                "- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.\n\n"
                "The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.\n\n"
                "Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more."
            )
        },
        {'role': 'assistant', 'content': 'Environmental impacts of eating meat'},
        {
            'role': 'user', 
            'content': (
                "I have a topic that contains the following documents:\n"
                "[DOCUMENTS]\n\n"
                "The topic is described by the following keywords: '[KEYWORDS]'.\n\n"
                "Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more."
            )
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, add_special_tokens=False, tokenize=False)

    return prompt

def get_clusters(log_density, xx, yy, coords, target_df, cfg):
    # Get peak coordinates
    min_distance = 2
    num_peaks = 10
    peak_indices = skimage.feature.peak_local_max(log_density, min_distance=min_distance, num_peaks=num_peaks)
    if peak_indices.shape[0] < 3:
        min_distance = 1
        peak_indices = skimage.feature.peak_local_max(log_density, min_distance=min_distance, num_peaks=num_peaks)

    peak_x = xx[peak_indices[:, 0], peak_indices[:, 1]]
    peak_y = yy[peak_indices[:, 0], peak_indices[:, 1]]
    peak_coords = np.vstack([peak_x, peak_y]).T

    # find shortest distance between peaks
    dists = np.linalg.norm(peak_coords[:, np.newaxis, :] - peak_coords[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)

    # map coordinates to nearest peak
    dists = np.linalg.norm(coords[:, np.newaxis, :] - peak_coords[np.newaxis, :, :], axis=-1)
    # any points further than half the min_dist are considered noise
    noise_threshold = min_dist / 2
    nearest_peak = np.argmin(dists, axis=1)
    nearest_dist = np.min(dists, axis=1)
    nearest_peak[nearest_dist > noise_threshold] = -1  # Mark as noise
    target_df = target_df.with_columns(pl.Series(name='cluster', values=nearest_peak))

    text_df = load_text_df(cfg)
    text_df = text_df.with_columns(pl.col('seed').struct.field('SeedName')).drop('seed')

    clusters_text_df = pl.DataFrame()
    peaks = []
    for cluster_id in target_df['cluster'].unique().to_list():
        if cluster_id == -1:
            continue
        cluster_df = target_df.filter(pl.col('cluster') == cluster_id)
        period_df = cluster_df.group_by('filter_value')\
            .agg([
                pl.col('createtime').min().alias('start_date'), 
                pl.col('createtime').max().alias('end_date')
            ])
        cluster_text_df = period_df.join_where(
            text_df,
            (pl.col('filter_value') == pl.col('SeedName')) & (pl.col('start_date') <= pl.col('createtime')) & (pl.col('end_date') >= pl.col('createtime'))
        )
        cluster_text_df = cluster_text_df.with_columns(pl.lit(cluster_id).alias('Topic'))

        if not cluster_text_df.is_empty():
            peaks.append(peak_coords[cluster_id])

        clusters_text_df = pl.concat([clusters_text_df, cluster_text_df], how='diagonal_relaxed')

    outliers_df = text_df.join(
            target_df.select(['filter_value']).unique(),
            left_on='SeedName',
            right_on='filter_value',
            how='inner'
        )\
        .select(['id', 'Document'])\
        .join(clusters_text_df.select(['id']), on='id', how='anti')\
        .with_columns(pl.lit(-1).alias('Topic'))
    
    if outliers_df.shape[0] > 10**6:
        outliers_df = outliers_df.sample(10**6, seed=42)

    # Use BERTopic to extract topics for each cluster
    extra_french_stopwords = ['quoi', 'alors', 'pourquoi', 'depuis', 'maintenant', 'où']
    english_french_stop_words = stopwords.words('english') + stopwords.words('french') + ['rt', 'url'] + extra_french_stopwords
    vectorizer_model = CountVectorizer(stop_words=english_french_stop_words, ngram_range=(1, 3))
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    model_name = 'microsoft/Phi-4-mini-instruct'
    generator = pipeline('text-generation', model=model_name, torch_dtype='auto')
    pipeline_kwargs = {
        # 'tokenizer_encode_kwargs': {'enable_thinking': False},
        'max_new_tokens': 20
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_gen_model = TextGeneration(
        generator,
        prompt=get_prompt(tokenizer),
        pipeline_kwargs=pipeline_kwargs,
        nr_docs=16,
        doc_length=200,
        tokenizer=tokenizer,
    )

    representation_model = {
        'Main': [
            KeyBERTInspired(),
            text_gen_model
        ],
        'Keywords': KeyBERTInspired()
    }
    topic_model = bertopic.BERTopic(
        embedding_model='intfloat/multilingual-e5-small',
        vectorizer_model=vectorizer_model, 
        ctfidf_model=ctfidf_model, 
        representation_model=representation_model, 
        verbose=True
    )
    topic_model.embedding_model = select_backend(topic_model.embedding_model, language=topic_model.language, verbose=topic_model.verbose)
    document_df = pl.concat([clusters_text_df.select(['Document', 'Topic']), outliers_df.select(['Document', 'Topic'])]).with_row_index()
    documents_per_topic = document_df.group_by('Topic').agg(pl.col('Document')).with_columns(pl.col('Document').list.join(' ')).to_pandas()
    topic_model.c_tf_idf_, words = topic_model._c_tf_idf(documents_per_topic)
    topic_representations = topic_model._extract_words_per_topic(
        words,
        document_df.to_pandas(),
    )

    # annotate clusters with topic names
    topic_names = []
    for cluster_id, peak_pos in enumerate(peaks):
        topic_name = topic_representations[cluster_id][0][0].split('\n')[-1]
        topic_names.append(topic_name)
        
    return peak_coords, topic_names

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    print("Loading data...")
    filter_val = 'SeedName'

    trend_path = cfg.trend_path
    trend_name = os.path.basename(trend_path.rstrip('/'))
    # keywords = ['climate', 'carbon', 'energy', 'fossil', 'fuel', 'gas', '\boil\b', '\bcoal\b', 'solar']
    # dir_name = f"{trend_name}/climate"
    keywords = None
    dir_name = f"{trend_name}/all"

    target_path = os.path.join(trend_path, 'pca_coords.parquet.zstd')
    if os.path.exists(target_path):
        target_df = pl.read_parquet(target_path)
        target_head_df = pl.read_parquet(target_path, n_rows=1)
        target_df = pl.read_parquet(target_path, columns=['createtime', 'filter_value', 'coord_2d'])
        components = np.load(os.path.join(cfg.trend_path, 'pca_components.npy'))
        stance_cols = [col for col in target_head_df.columns if col.startswith('trend_mean_')]
        target_df = target_df.select(['createtime', 'filter_value', 'coord_2d'])
        coords = target_df['coord_2d'].to_numpy()
    else:
        all_trend_path = os.path.join(trend_path, 'loaded_trends.parquet.zstd')
        if os.path.exists(all_trend_path):
            df = pl.read_parquet(all_trend_path)
        else:
            df = load_df(trend_path, 'SeedName')
            df.write_parquet(all_trend_path, compression='zstd')
        
        target_df, feature_cols, stance_cols, volume_cols = pivot_and_impute(df)
        pca, coords, components = do_pca(target_df, stance_cols)
        target_df = target_df.with_columns(pl.Series(name='coord_2d', values=coords))
        np.save(os.path.join(trend_path, 'pca_components.npy'), components)
        target_df.write_parquet(target_path, compression='zstd')

    # Create scatter plot of PCA coordinates colored by cluster
    fig, ax = plt.subplots(figsize=(10, 8))
    if len(coords) == 0:
        return None, (None, None)
    
    contours, (xx, yy), log_density = create_kde_background(coords, ax)
    plot_streamplot(ax, target_df, xx, yy)

    peak_coords, topic_names = get_clusters(log_density, xx, yy, coords, target_df, cfg)

    texts = []
    for peak_coord, topic_name in zip(peak_coords, topic_names):
        ax.plot(peak_coord[0], peak_coord[1], 'rX', markersize=12, markeredgewidth=2)

        txt = ax.text(peak_coord[0], peak_coord[1], topic_name, fontsize=8, color='darkblue',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='blue', alpha=0.8))
        texts.append(txt)

    adjust_text(
        texts, 
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5),
        force_text=(2.0, 2.0),
        force_static=(1.0, 1.0)
    )

    fig.savefig(f"./figs/{dir_name}/pca_clusters.png", dpi=300)
    
if __name__ == '__main__':
    main()