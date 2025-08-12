import os

import hydra
import polars as pl

from stancemining import StanceMining

def save_stance(batch_week_df: pl.DataFrame, miner: StanceMining, week_batch_path):
    if os.path.exists(week_batch_path):
        original_batch_week_df = batch_week_df.clone()
        existing_batch_week_df = pl.read_parquet(week_batch_path)
        if 'Polarities' in existing_batch_week_df.columns:
            existing_batch_week_df = existing_batch_week_df.rename({'Polarities': 'Stances'})
        if 'stance_finetune_kwargs' not in existing_batch_week_df.columns:
            existing_batch_week_df = existing_batch_week_df.with_columns(pl.lit(None).alias('stance_finetune_kwargs'))
        existing_target_df = existing_batch_week_df.explode(['Targets', 'Stances']).drop_nulls('Targets').unique(['id', 'platform', 'Targets'])
        
        target_df = batch_week_df.explode('Targets').drop_nulls('Targets')
        joined_target_df = target_df.join(existing_target_df.select(['id', 'platform', 'Targets', 'Stances', 'stance_finetune_kwargs']), on=['id', 'platform', 'Targets'], how='left')
        remaining_target_df = joined_target_df.filter(pl.col('Stances').is_null()).drop(['Stances', 'stance_finetune_kwargs'])
        existing_target_df = joined_target_df.filter(pl.col('Stances').is_not_null())
        
        if existing_target_df.is_empty():
            existing_target_df = None
        else:
            assert existing_target_df.shape[0] + remaining_target_df.shape[0] == target_df.shape[0]

            remaining_batch_week_df = remaining_target_df.group_by(['id', 'platform'])\
                .agg([pl.col('Targets')] + [pl.col(c).first() for c in remaining_target_df.columns if c not in ['Targets', 'id', 'platform']])
            
            if remaining_batch_week_df.is_empty():
                return

            batch_week_df = remaining_batch_week_df
    else:
        existing_target_df = None

    # take care of duplicate documents (i.e. from retweets) but keep all targets
    unique_week_df = batch_week_df.explode('Targets')\
        .drop_nulls('Targets')\
        .unique(['Document', 'ParentDocument', 'Targets'])\
        .group_by(['Document', 'ParentDocument'])\
        .agg(pl.col('Targets'))
    # unique_week_df = batch_week_df.unique(['Document', 'ParentDocument'])
    unique_week_df = miner.get_stance(unique_week_df, text_column='Document', parent_text_column='ParentDocument')
    week_stance_df = batch_week_df.drop('Targets')\
        .join(unique_week_df.select(['Document', 'ParentDocument', 'Targets', 'Stances']), on=['Document', 'ParentDocument'], how='left', nulls_equal=True)
    # week_stance_df = week_stance_df.with_columns(pl.lit(finetune_kwargs).alias('stance_finetune_kwargs'))

    if existing_target_df is not None:
        combined_stance_df = pl.concat([
                week_stance_df.explode(['Targets', 'Stances']).drop_nulls('Targets'), 
                existing_target_df.select(week_stance_df.columns)
            ], how='vertical_relaxed')\
            .group_by(['id', 'platform'])\
            .agg([pl.col('Targets'), pl.col('Stances')] + [pl.col(c).first() for c in remaining_target_df.columns if c not in ['Targets', 'Stances', 'id', 'platform']])
        
        combined_stance_df = pl.concat([
            combined_stance_df, 
            original_batch_week_df.filter(pl.col('Targets').list.len() == 0).with_columns([pl.lit([]).alias('Stances'), pl.lit(None).alias('stance_finetune_kwargs')]).select(combined_stance_df.columns)
        ])

        unique_stance_df = combined_stance_df.join(original_batch_week_df.select(['id', 'platform']), on=['id', 'platform'], how='inner')\
            .sort(['id', 'platform', pl.col('Targets').list.len()], descending=True)\
            .unique(['id', 'platform'], keep='first')
        
        if unique_stance_df.shape[0] != original_batch_week_df.shape[0]:
            print(f"Warning: combined_stance_df.shape[0] != original_batch_week_df.shape[0]: {combined_stance_df.shape[0]} != {original_batch_week_df.shape[0]}")
        week_stance_df = unique_stance_df

    week_stance_df.write_parquet(week_batch_path, compression='zstd')

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    document_df = pl.read_parquet(f'./data/stance_targets/all_doc_targets.parquet.zstd')
    document_df = document_df.select(['id', 'Document', 'ParentDocument', 'createtime', 'seed', 'Targets', 'finetune_kwargs', 'platform'])

    # truncate text for now
    # TODO come up with smarter system
    document_df = document_df.with_columns((pl.col('Document').str.len_chars().fill_null(0) + pl.col('ParentDocument').str.len_chars().fill_null(0)).alias('text_len'))\
        .filter(pl.col('text_len') < 8500).drop('text_len')

    miner = StanceMining(
        verbose=True
    )
    # batch out calls
    week_df = document_df.select([pl.col('createtime').dt.year().alias('year'), pl.col('createtime').dt.week().alias('week')]).unique().sort(['year', 'week'], descending=True)
    for week in week_df.to_dicts():
        week_batch_path = f'{config.base_stance_path}/{week["year"]}_{week["week"]}_doc_targets_with_stance.parquet.zstd'
        print(f"Processing week {week['week']} of year {week['year']} of {len(week_df)} weeks")
        batch_week_df = document_df.filter((pl.col('createtime').dt.year() == week['year']) & (pl.col('createtime').dt.week() == week['week']))

        save_stance(batch_week_df, miner, week_batch_path)

if __name__ == "__main__":
    main()
