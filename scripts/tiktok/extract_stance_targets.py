import os

import polars as pl
from tqdm import tqdm

import stancemining

def main():
    dir_path = '../sitrep/data/digital_trace/raw_platforms'

    data_files = os.listdir(dir_path)
    file_df = pl.DataFrame({'file': data_files})
    file_df = file_df.filter(pl.col('file').str.starts_with('tiktok'))
    file_df = file_df.filter(pl.col('file').str.ends_with('parquet.zstd'))

    # parse out date from file name
    file_df = file_df.with_columns(pl.col('file').str.split('_').list.get(1).str.split('.').list.get(0).str.split('-').alias('date_numbers'))\
        .with_columns(pl.col('date_numbers').list.get(0).cast(pl.UInt16).alias('year'))\
        .with_columns(pl.col('date_numbers').list.get(1).cast(pl.UInt8).alias('month'))\
        .with_columns(pl.col('date_numbers').list.get(2).cast(pl.UInt8).alias('day'))

    df = pl.read_parquet('./data/tiktok/transcripts.parquet.zstd')

    df = df.with_columns(pl.col('transcript').struct.field('segments').list.eval(pl.col('').struct.field('text')).list.join(' ').str.strip_chars().alias('text'))
    finetune_kwargs = {
        'model_name': 'HuggingFaceTB/SmolLM2-360M-Instruct',
        'add_system_message': True,
        'save_model_path': '../stancemining/models/stancemining',
        'prompting_method': 'stancemining',
        'classification_method': 'generation',
        'generation_method': 'list',
        'batch_size': 64
    }

    save_path = './data/tiktok/stance_targets'
    os.makedirs(save_path, exist_ok=True)

    video_df = pl.DataFrame()
    for file_name in tqdm(file_df['file']):
        batch_df = pl.read_parquet(f'{dir_path}/{file_name}')
        video_df = pl.concat([video_df, batch_df], how='diagonal_relaxed')

    video_df = video_df.with_columns([
        pl.col('video_id').cast(pl.UInt64),
        pl.col('seed').struct.field('MainType')
    ])
    df = df.select(['video_id', 'createtime', 'seed_id', 'text']).join(video_df.select(['video_id', 'MainType']), on='video_id', how='left')

    # filter to politicians and influencers
    df = df.filter(pl.col('MainType').is_in(['influencer', 'politician']))

    unique_df = df.unique('text').filter(pl.col('text').is_not_null())
    docs = unique_df['text'].to_list()

    model = stancemining.StanceMining(
        model_kwargs={'device_map': {'': 1}},
        finetune_kwargs=finetune_kwargs,
        load_generator=False
    )

    doc_df = model.get_base_targets(docs)
    target_df = unique_df.select(['video_id', 'text']).with_columns(doc_df['Targets'])
    target_df = target_df.with_columns(pl.lit(finetune_kwargs).alias('finetune_kwargs'))

    df = df.select(['video_id']).join(target_df, on='video_id', how='left')

    target_df.write_parquet(os.path.join(save_path, f'targets.parquet.zstd'), compression='zstd')

if __name__ == '__main__':
    main()