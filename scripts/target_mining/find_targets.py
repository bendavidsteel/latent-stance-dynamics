import os

import polars as pl

import stancemining

def main():
    # load base targets from twitter, instagram and tiktok
    tiktok_target_df = pl.read_parquet('./data/tiktok/stance_targets/targets.parquet.zstd')
    tiktok_target_df = tiktok_target_df.with_columns([pl.col('text').alias('Document'), pl.col('createtime').str.to_datetime().dt.convert_time_zone('UTC')])
    # twitter_target_dir = './data/twitter/stance_targets'
    # twitter_target_df = pl.DataFrame()
    # for filename in os.listdir(twitter_target_dir):
    #     if filename.startswith('targets') and filename.endswith('.parquet.zstd'):
    #         twitter_target_file_df = pl.read_parquet(f'{twitter_target_dir}/{filename}')
    #         twitter_target_df = pl.concat([twitter_target_df, twitter_target_file_df], how='diagonal_relaxed')
    twitter_target_df = pl.read_parquet('./data/twitter/stance_targets/targets.parquet.zstd')
    twitter_target_df = twitter_target_df.with_columns([pl.col('rawContent').alias('Document'), pl.col('date').str.to_datetime().alias('createtime')])
    # instagram_target_dir = './data/instagram/stance_targets'
    # instagram_target_df = pl.DataFrame()
    # for filename in os.listdir(instagram_target_dir):
    #     if filename.startswith('targets') and filename.endswith('.parquet.zstd'):
    #         instagram_target_file_df = pl.read_parquet(f'{instagram_target_dir}/{filename}')
    #         instagram_target_df = pl.concat([instagram_target_df, instagram_target_file_df], how='diagonal_relaxed')
    instagram_target_df = pl.read_parquet('./data/instagram/stance_targets/targets.parquet.zstd')
    instagram_target_df = instagram_target_df.with_columns([pl.col('raw_caption').alias('Document'), pl.from_epoch(pl.col('taken_at')).dt.convert_time_zone('UTC').alias('createtime')])
    target_df = pl.concat([tiktok_target_df, twitter_target_df, instagram_target_df], how='diagonal_relaxed')

    finetune_kwargs = {
        'model_name': 'HuggingFaceTB/SmolLM2-360M-Instruct',
        'add_system_message': True,
        'save_model_path': '../stancemining/models/stancemining',
        'prompting_method': 'stancemining',
        'classification_method': 'generation',
        'generation_method': 'list',
        'batch_size': 64
    }

    model = stancemining.StanceMining(
        model_name='microsoft/Phi-4-mini-instruct',
        model_kwargs={'device_map': 'auto', 'trust_remote_code': True, 'torch_dtype': 'auto'},
        finetune_kwargs=finetune_kwargs,
        get_stance=False,
        verbose=True
    )

    target_path = './data/stance_targets/unique_targets.parquet.zstd'
    if not os.path.exists(target_path):
        unique_target_df = target_df.select('Targets').explode('Targets').unique('Targets')
        unique_target_df.write_parquet(target_path, compression='zstd')

        embeddings = model._get_embeddings(unique_target_df['Targets'].to_list())
        # save embeddings
        unique_target_df = unique_target_df.with_columns(pl.Series(name='embeddings', values=embeddings))
        unique_target_df.write_parquet(target_path, compression='zstd')
    else:
        unique_target_df = pl.read_parquet(target_path)
    unique_target_df = unique_target_df.rename({'Targets': 'text', 'embeddings': 'embedding'})

    bertopic_kwargs = {
        'min_topic_size': 50,
        'verbose': True
    }

    target_df = target_df.filter(pl.col('createtime') > pl.lit('2024-12-1').str.to_datetime().dt.convert_time_zone('UTC'))

    target_df = target_df.with_columns(stancemining.utils.filter_stance_targets(pl.col('Targets')))

    doc_target_df = model.fit_transform(target_df, embedding_cache=unique_target_df, bertopic_kwargs=bertopic_kwargs)
    target_info = model.get_target_info()
    doc_target_df.write_parquet('./data/stance_targets/1month_doc_targets.parquet.zstd', compression='zstd')

if __name__ == '__main__':
    main()
