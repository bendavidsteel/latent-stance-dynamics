import os

import polars as pl

from stancemining.main import StanceMining

def save_stance(batch_week_df: pl.DataFrame, miner: StanceMining, week_batch_path, finetune_kwargs):
    if os.path.exists(week_batch_path):
        existing_batch_week_df = pl.read_parquet(week_batch_path)
        existing_target_df = existing_batch_week_df.explode(['Targets', 'Polarities'])
        target_df = batch_week_df.explode('Targets')
        remaining_target_df = target_df.join(existing_target_df, on=['Document', 'Targets'], how='anti')
        existing_target_df = existing_target_df.join(target_df.select(['Document', 'Targets']), on=['Document', 'Targets'], how='inner')
        batch_week_df = remaining_target_df.group_by([col for col in target_df.columns if col != 'Targets'])\
            .agg(pl.col('Targets'))
    else:
        existing_target_df = None

    # take care of duplicate documents (i.e. from retweets)
    unique_week_df = batch_week_df.unique('Document')
    unique_week_df, _ = miner.get_stance(unique_week_df)
    week_stance_df = batch_week_df.drop('Targets')\
        .join(unique_week_df.select(['Document', 'Targets', 'Polarities']), on='Document', how='left')
    week_stance_df = week_stance_df.with_columns(pl.lit(finetune_kwargs).alias('stance_finetune_kwargs'))

    if existing_target_df is not None:
        week_stance_df = pl.concat([week_stance_df.explode(['Targets', 'Polarities']), existing_target_df])\
            .group_by([col for col in target_df.columns if col not in ['Targets', 'Polarities']])\
            .agg([pl.col('Targets'), pl.col('Polarities')])

    week_stance_df.write_parquet(week_batch_path, compression='zstd')

def main():
    # TODO intentionally leaving out parentdocument for emnlp submission
    document_df = pl.read_parquet(f'./data/twitter/2024_doc_targets.parquet.zstd', columns=['id', 'seed', 'createtime', 'platform', 'Document', 'finetune_kwargs', 'Targets'])
    document_df = document_df.filter(pl.col('platform') == 'twitter')

    finetune_kwargs = {
        'model_name': 'HuggingFaceTB/SmolLM2-135M',
        'add_system_message': True,
        'save_model_path': '../stancemining/models/stancemining',
        'prompting_method': 'stancemining',
        'classification_method': 'generation',
        'generation_method': 'list',
        'batch_size': 200
    }

    model_kwargs = {
            'device_map': {'': 1},
        'torch_dtype': 'auto',
        'attn_implementation': 'flash_attention_2'
    }

    miner = StanceMining(finetune_kwargs=finetune_kwargs, model_kwargs=model_kwargs, verbose=True)
    # batch out calls
    week_df = document_df.select([
            pl.col('createtime').dt.year().alias('year'), 
            pl.col('createtime').dt.week().alias('week')
        ])\
        .unique()\
        .filter(pl.col('year') == 2024)\
        .sort(['year', 'week'], descending=True)
    for week in week_df.to_dicts():
        week_batch_path = f'./data/twitter/doc_stance/{week["year"]}_{week["week"]}_doc_targets_with_stance.parquet.zstd'
        print(f"Processing week {week['week']} of year {week['year']} of {len(week_df)} weeks")
        batch_week_df = document_df.filter((pl.col('createtime').dt.year() == week['year']) & (pl.col('createtime').dt.week() == week['week']))
        
        save_stance(batch_week_df, miner, week_batch_path, finetune_kwargs)

if __name__ == "__main__":
    main()
