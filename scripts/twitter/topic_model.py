import os

import bertopic
from sklearn.feature_extraction.text import CountVectorizer
from cuml.preprocessing import normalize
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from nltk.corpus import stopwords
import numpy as np
import polars as pl
import sentence_transformers

def main():
    embedding_file_path = './data/twitter/2024_doc_targets_with_embeddings.parquet.zstd'

    if not os.path.exists(embedding_file_path):
        df = pl.read_parquet('./data/twitter/2024_doc_targets.parquet.zstd')
        embedder = sentence_transformers.SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')
        
        doc_df = df.unique('Document')
        embeddings = embedder.encode(doc_df['Document'].to_list(), batch_size=256, show_progress_bar=True)
        doc_df = doc_df.with_columns(pl.Series('embedding', embeddings))
        doc_df.write_parquet(embedding_file_path, compression='zstd')
    else:
        doc_df = pl.read_parquet(embedding_file_path)

    # doc_df = doc_df.sample(500000)

    stop_words = list(set(stopwords.words('english'))) + list(set(stopwords.words('french')))
    vectorizer_model = CountVectorizer(stop_words=stop_words)

    embeddings = doc_df['embedding'].to_numpy()
    embeddings = normalize(embeddings)

    # Create instances of GPU-accelerated UMAP and HDBSCAN
    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, verbose=True)
    hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True, verbose=True)

    topic_model = bertopic.BERTopic(
        verbose=True, 
        umap_model=umap_model, 
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model
    )
    topic, probs = topic_model.fit_transform(doc_df['Document'].to_list(), embeddings=embeddings)
    topic_info_df = topic_model.get_topic_info()
    topic_info_df = pl.from_pandas(topic_info_df)
    topic_info_df.write_parquet('./data/twitter/topic_info.parquet')

if __name__ == '__main__':
    main()