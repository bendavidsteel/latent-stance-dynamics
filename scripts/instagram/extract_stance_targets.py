import os

import polars as pl

import stancemining

def main():
    dir_path = '../sitrep/data/digital_trace/raw_platforms'

    data_files = os.listdir(dir_path)
    file_df = pl.DataFrame({'file': data_files})
    file_df = file_df.filter(pl.col('file').str.starts_with('instagram'))
    file_df = file_df.filter(pl.col('file').str.ends_with('parquet.zstd'))

    # parse out date from file name
    file_df = file_df.with_columns(pl.col('file').str.split('_').list.get(1).str.split('.').list.get(0).str.split('-').alias('date_numbers'))\
        .with_columns(pl.col('date_numbers').list.get(0).cast(pl.UInt16).alias('year'))\
        .with_columns(pl.col('date_numbers').list.get(1).cast(pl.UInt8).alias('month'))\
        .with_columns(pl.col('date_numbers').list.get(2).cast(pl.UInt8).alias('day'))

    finetune_kwargs = {
        'model_name': 'HuggingFaceTB/SmolLM2-360M-Instruct',
        'add_system_message': True,
        'save_model_path': '../stancemining/models/stancemining',
        'prompting_method': 'stancemining',
        'classification_method': 'generation',
        'generation_method': 'list',
        'batch_size': 64
    }

    save_path = './data/instagram/stance_targets'
    os.makedirs(save_path, exist_ok=True)

    # group by month
    for month_files_df in file_df.sort(['year', 'month', 'day']).partition_by(['year', 'month', 'day']):
        try:
            df = pl.DataFrame()
            for file_name in month_files_df['file']:
                batch_df = pl.read_parquet(f'{dir_path}/{file_name}')
                df = pl.concat([df, batch_df], how='diagonal_relaxed')

            # filter to politicians and influencers
            df = df.filter(pl.col('seed').struct.field('MainType').is_in(['influencer', 'politician']))

            df = df.with_columns(pl.col('caption').struct.field('text').alias('raw_caption'))
            unique_df = df.unique('raw_caption').filter(pl.col('raw_caption').is_not_null())
            docs = unique_df['raw_caption'].to_list()

            model = stancemining.StanceMining(
                model_kwargs={'device_map': {'': 1}},
                finetune_kwargs=finetune_kwargs,
                load_generator=False
            )

            doc_df = model.get_base_targets(docs)
            target_df = unique_df.select(['id', 'raw_caption']).with_columns(doc_df['Targets'])
            target_df = target_df.with_columns(pl.lit(finetune_kwargs).alias('finetune_kwargs'))

            df = df.select(['id', 'seed_id', 'taken_at']).join(target_df, on='id', how='left')

            year = month_files_df['year'][0]
            month = month_files_df['month'][0]
            day = month_files_df['day'][0]
            target_df.write_parquet(os.path.join(save_path, f'targets_{year}_{month}_{day}.parquet.zstd'), compression='zstd')
        except Exception as e:
            print(e)
            continue

if __name__ == '__main__':
    main()