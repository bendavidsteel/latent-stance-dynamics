import os

import numba
import numpy as np
import polars as pl
from pynndescent import NNDescent
import hdbscan
from cuml import DBSCAN
from tqdm import tqdm

@numba.jit(nopython=True)
def cosine_distance(embed_a, embed_b):
    return 1 - np.dot(embed_a, embed_b) / (np.linalg.norm(embed_a) * np.linalg.norm(embed_b))

def main():
    pl.set_random_seed(42)

    platform_dir = '../sitrep/data/digital_trace/raw_platforms'
    dfs = []
    for filename in tqdm(os.listdir(platform_dir)):
        if filename.startswith('tiktok') and filename.endswith('.parquet.zstd'):
            try:
                file_df = pl.read_parquet(os.path.join(platform_dir, filename), columns=['transcripts', 'speaker_embeddings', 'video_id', 'author_name', 'seed'])
                dfs.append(file_df)
            except pl.exceptions.ColumnNotFoundError:
                print(f"File: {filename} did not continue all necessary columns")
                continue
    df = pl.concat(dfs, how='diagonal_relaxed')
    
    segment_df = df.with_columns(pl.col('transcripts').struct.field('segments'))\
        .explode('segments')\
        .drop_nulls('segments')\
        .with_columns([
            pl.col('segments').struct.field('speaker').str.split('_').list.get(-1).cast(pl.UInt32).alias('speaker_index'),
            pl.col('segments').struct.unnest()
        ])

    speaker_embedding_df = df.filter(pl.col('speaker_embeddings').is_not_null())\
        .explode('speaker_embeddings')\
        .select(['video_id', 'author_name', 'seed', pl.col('speaker_embeddings').struct.unnest()])\
        .with_columns(pl.col('embedding').cast(pl.Array(pl.Float32, 256)))\
        .filter(pl.col('embedding').arr.get(0).is_not_nan())
    speaker_embeddings = speaker_embedding_df['embedding'].to_numpy().copy()

    dataset = [
        ('7088426673405824261', 1, '7067276646365154566', 0, True),
        ('7079979155835718918', 2, '7088473393200172294', 0, False),
        ('7201863266191166725', 0, '7067276646365154566', 0, True),
        ('7118832382433561862', 4, '7237216570659851525', 2, False),
        ('7150549788776238342', 0, '7069119384811293957', 0, True),
        ('7242069707858267397', 2, '7220107651781381382', 0, False),
        ('7372278178213285126', 1, '7288776373269761286', 4, False),
        ('7050281009283108102', 0, '7398313537547898155', 0, False),
        ('7188255302368644357', 1, '7377485052436892934', 1, False),
        ('7119266586812501294', 0, '7132637749982563589', 0, True),
        ('7088407419927219461', 0, '7058413097769291013', 0, True),
        ('7346951783631179013', 0, '7347029775640186118', 0, True),
        ('7142863259060702506', 0, '7107597756587330858', 0, False)
    ]

    # normalize cluster embeddings
    speaker_embeddings = speaker_embeddings / np.linalg.norm(speaker_embeddings, axis=1, keepdims=True)
    
    highest_accuracy = 0
    best_eps = None
    for eps in np.linspace(0.1, 0.5, 20):
        clusterer = DBSCAN(eps=eps, metric='cosine', min_samples=2)
        embed_clusters = clusterer.fit_predict(speaker_embeddings)
        # embed_clusters = h_cluster.dbscan_clustering(cut_distance=eps, min_cluster_size=2)
        speaker_embedding_df = speaker_embedding_df.with_columns(pl.Series('cluster', embed_clusters))

        # check if clusters are correct based on dataset
        num_correct = 0
        num_present = 0
        for video_id_a, speaker_idx_a, video_id_b, speaker_idx_b, same_speaker in dataset:
            cluster_as = speaker_embedding_df.filter((pl.col('video_id') == video_id_a) & (pl.col('speaker_index') == speaker_idx_a))['cluster'].to_numpy()
            cluster_bs = speaker_embedding_df.filter((pl.col('video_id') == video_id_b) & (pl.col('speaker_index') == speaker_idx_b))['cluster'].to_numpy()

            if len(cluster_as) == 0 or len(cluster_bs) == 0:
                print(f"Warning: Missing embedding for {video_id_a}:{speaker_idx_a} or {video_id_b}:{speaker_idx_b}")
                continue
            num_present += 1

            cluster_a = cluster_as[0]
            cluster_b = cluster_bs[0]

            if same_speaker:
                num_correct += int(cluster_a == cluster_b and cluster_a != -1)
            else:
                num_correct += int(cluster_a != cluster_b or (cluster_a == -1 and cluster_b == -1))
        accuracy = num_correct / num_present
        print(f"Eps: {eps:.3f}, Accuracy: {accuracy:.3f} ({num_correct}/{num_present})")
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_eps = eps

    print(f"Best eps: {best_eps:.3f} with accuracy: {highest_accuracy:.3f}")
    embed_clusters = DBSCAN(eps=best_eps, metric='cosine', min_samples=2).fit_predict(speaker_embeddings)
    new_cluster_id = embed_clusters.max() + 1
    for i in range(len(embed_clusters)):
        if embed_clusters[i] == -1:
            embed_clusters[i] = new_cluster_id
            new_cluster_id += 1
    speaker_embedding_df = speaker_embedding_df.with_columns(pl.Series('cluster', embed_clusters))

    speaker_embedding_df = speaker_embedding_df.join(segment_df.unique(['video_id', 'speaker_index']).select(['video_id', 'speaker_index', 'start', 'speaker', 'text']), on=['video_id', 'speaker_index'], how='left')

    # print details by author
    print("Details by author:")
    for author_df in speaker_embedding_df.drop_nulls('text').sample(fraction=1.0).partition_by('seed')[:3]:
        segments = author_df.unique('video_id').head(2)
        if len(segments) < 2:
            continue
        for segment in segments.to_dicts():
            url = f"https://www.tiktok.com/@user/video/{segment['video_id']}"
            print(f"Video: {url}, Speaker: {segment['speaker_index']}, Text: {segment['text']}, Cluster: {segment['cluster']}")

    # print details by cluster
    print("Details by cluster:")
    for cluster_df in speaker_embedding_df.drop_nulls('text').filter(pl.col('cluster') != -1).sample(fraction=1.0).partition_by('cluster')[-3:]:
        for segment in cluster_df.unique('video_id').head(2).to_dicts():
            url = f"https://www.tiktok.com/@user/video/{segment['video_id']}"
            print(f"Video: {url}, Speaker: {segment['speaker_index']}, Text: {segment['text']}, Cluster: {segment['cluster']}")

    cluster_df = speaker_embedding_df.group_by('author_name')\
        .agg(pl.col('video_id').n_unique().alias('num_account_videos'))\
        .join(
            speaker_embedding_df.group_by(['author_name', 'cluster'])\
                .agg(pl.col('video_id').n_unique().alias('num_account_cluster_videos')),
            on='author_name', 
            how='left'
        )\
        .with_columns((pl.col('num_account_cluster_videos') / pl.col('num_account_videos')).alias('cluster_ratio'))\
        .filter((pl.col('cluster_ratio') > 0.5) & (pl.col('num_account_videos') > 2))\

    speaker_embedding_df = speaker_embedding_df.join(
        cluster_df.select(['author_name', 'cluster', 'cluster_ratio']),
        on=['author_name', 'cluster'],
        how='left'
    )
    speaker_embedding_df = speaker_embedding_df.with_columns(pl.when(pl.col('cluster_ratio').is_not_null()).then(pl.lit(True)).otherwise(pl.lit(False)).alias('is_author'))
    speaker_embedding_df = speaker_embedding_df.drop('cluster_ratio')
    speaker_embedding_df.select(['video_id', 'speaker_index', 'is_author'])\
        .write_parquet(os.path.join('data', 'tiktok', 'speaker_author.parquet.zstd'), compression='zstd')

if __name__ == "__main__":
    main()