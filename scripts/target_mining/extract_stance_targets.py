import datetime
import gc
import os
import traceback as tb

import hydra
import polars as pl
import torch
from tqdm import tqdm

from stancemining.main import StanceMining
from find_targets import remove_bad_targets

def add_dialogue_turn(df: pl.DataFrame):
    """
    Add a 'dialogue_turn' column to the dataframe that counts dialogue turns within each video.
    
    A new turn is counted whenever the speaker changes.
    """
    df = df.sort(['video_id', 'start'])
    # Create a column to identify when the speaker changes within each video
    df = df.with_columns([
        # False for the first row of each video, otherwise check if speaker changed
        pl.when(
            # Check if it's the first row of a video
            pl.col('video_id') != pl.col('video_id').shift(1)
        ).then(False).otherwise(
            # Check if the speaker changed
            pl.col('speaker_index') != pl.col('speaker_index').shift(1)
        ).alias('is_speaker_change')
    ])
    
    # Calculate cumulative sum of speaker changes for each video to get dialogue turn
    df = df.with_columns([
        pl.col('is_speaker_change').cast(pl.Int64).cum_sum().over('video_id').fill_null(0).alias('dialogue_turn')
    ])
    
    # Drop the temporary column
    df = df.drop('is_speaker_change')
    
    return df

class PlatformHandler:
    def __init__(self):
        self.tiktok_transcript_df = pl.read_parquet('./data/tiktok/transcripts.parquet.zstd')
        self.tiktok_speaker_author_df = pl.read_parquet('./data/tiktok/speaker_author.parquet.zstd').with_columns(pl.col('video_id').cast(pl.UInt64))

        dir_path = '../sitrep/data/digital_trace/raw_platforms'

        data_files = os.listdir(dir_path)
        file_df = pl.DataFrame({'file': data_files})
        file_df = file_df.filter(pl.col('file').str.starts_with('twitter'))
        file_df = file_df.filter(pl.col('file').str.ends_with('parquet.zstd'))

        # parse out date from file name
        file_df = file_df.with_columns(pl.col('file').str.split('_').list.get(1).str.split('.').list.get(0).str.split('-').alias('date_numbers'))\
            .with_columns(pl.col('date_numbers').list.get(0).cast(pl.UInt16).alias('year'))\
            .with_columns(pl.col('date_numbers').list.get(1).cast(pl.UInt8).alias('month'))\
            .with_columns(pl.col('date_numbers').list.get(2).cast(pl.UInt8).alias('day'))

        self.twitter_df = pl.DataFrame()
        for file_name in tqdm(file_df['file']):
            batch_df = pl.read_parquet(f'{dir_path}/{file_name}', columns=['id', 'rawContent']).with_columns(pl.col('id').cast(pl.UInt64))
            self.twitter_df = pl.concat([self.twitter_df, batch_df], how='diagonal_relaxed')


    def format_platform_data(self, df: pl.DataFrame, platform):
        if platform == 'tiktok':
            # TODO format text with author
            df = df.with_columns(pl.col('video_id').cast(pl.UInt64))
            df = df.join(
                self.tiktok_transcript_df.select([pl.col('video_id'), 'transcript']), 
                on='video_id', 
                how='left'
            ).filter(pl.col('transcript').is_not_null())
            unique_speaker_df = df.select([
                'video_id', 
                (pl.col('transcript').struct.field('segments').list.eval(pl.col('').struct.field('speaker')).list.unique().list.len() > 1).alias('multiple_speakers')
            ])
            # get speaker indexs
            df = df.with_columns(pl.col('transcript').struct.field('segments'))\
                .explode('segments')\
                .with_columns(pl.col('segments').struct.unnest())\
                .with_columns(pl.col('speaker').str.split('_').list.get(-1).cast(pl.UInt32).alias('speaker_index'))
            # find cases where a speaker index is also the author of the video
            df = df.join(self.tiktok_speaker_author_df, on=['video_id', 'speaker_index'])\
                .with_columns([
                    pl.when(pl.col('is_author'))\
                    .then(pl.lit('author'))\
                    .otherwise(pl.col('speaker'))\
                    .alias('speaker'),
                    pl.col('text').str.strip_chars()
                ])
            df = df.join(unique_speaker_df, on='video_id', how='left')
            df = add_dialogue_turn(df)
            # group speaker sections together 
            df = df.sort(['video_id', 'start'])\
                .group_by(['video_id', 'createtime', 'seed', 'multiple_speakers', 'dialogue_turn'], maintain_order=True)\
                .agg([pl.col('text'), pl.col('start').min(), pl.col('is_author').first()])\
                .with_columns(pl.col('text').list.join(' ').str.strip_chars())
            
            df = df.with_columns(
                pl.when(pl.col('is_author') | ~pl.col('multiple_speakers'))\
                    .then(pl.col('text'))\
                    .otherwise(pl.format('"{}"', 'text'))\
                    .alias('formatted_text')
            )
            # TODO fix dialogue formatting
            df = df.sort(['video_id', 'start'])\
                .group_by(['video_id', 'createtime', 'seed'], maintain_order=True)\
                .agg(pl.col('formatted_text'))\
                .with_columns(pl.col('formatted_text').list.join(' ').str.strip_chars().alias('text'))
            df = df.with_columns([
                pl.col('video_id').alias('id').cast(pl.String),
                pl.col('text').alias('Document'), 
                pl.lit(None).alias('ParentDocument'),
                pl.col('createtime').str.to_datetime().dt.convert_time_zone('UTC'), 
                pl.lit('tiktok').alias('platform')]
            )
        elif platform == 'instagram':
            df = df.with_columns([
                pl.lit(None).alias('ParentDocument'),
                pl.col('caption').struct.field('text').alias('Document'), 
                pl.from_epoch(pl.col('taken_at')).dt.convert_time_zone('UTC').alias('createtime'), 
                pl.lit('instagram').alias('platform')
            ])
        elif platform == 'instagram-meta':
            df = df.with_columns([
                pl.lit(None).alias('ParentDocument'),
                pl.col('description').alias('Document'), 
                pl.col('date').str.to_datetime('%Y-%m-%dT%H:%M:%SZ').dt.convert_time_zone('UTC').alias('createtime'), 
                pl.lit('instagram').alias('platform')
            ])
        elif platform == 'twitter':
            df = df.with_columns([
                pl.col('id').cast(pl.UInt64),
                pl.col('inReplyToTweetId').replace("None", None).replace("-1", None).cast(pl.UInt64),
            ])
            df = df.join(self.twitter_df.select(['id', pl.col('rawContent').alias('inReplyToTweet')]), left_on='inReplyToTweetId', right_on='id', how='left')
            if 'rawContent' in df.schema['quotedTweet']:
                df = df.with_columns(
                    pl.when(pl.col('inReplyToTweet').is_not_null())\
                        .then(pl.col('inReplyToTweet'))\
                        .when(pl.col('quotedTweet').struct.field('rawContent').is_not_null())\
                        .then(pl.col('quotedTweet').struct.field('rawContent'))\
                        .otherwise(pl.lit(None))\
                        .alias('ParentDocument')
                )
            else:
                df = df.with_columns(
                    pl.when(pl.col('inReplyToTweet').is_not_null())\
                        .then(pl.col('inReplyToTweet'))\
                        .otherwise(pl.lit(None))\
                        .alias('ParentDocument')
                )

            df = df.with_columns([
                pl.col('id').cast(pl.String),
                pl.col('rawContent').alias('Document'), 
                pl.col('date').str.to_datetime(time_zone='UTC').alias('createtime'), 
                pl.lit('twitter').alias('platform')
            ])
            # TODO remove rt @user and urls
            df = df.with_columns([
                pl.col('Document').str.replace(r'https://t.co/\w+', 'URL').str.replace('rt @\w+', ''),
                pl.col('ParentDocument').str.replace(r'https://t.co/\w+', 'URL').str.replace('rt @\w+', '')
            ])
        elif platform == 'bluesky':
            # TODO add reply.root content as multiple parent documents
            try:
                df = df.with_columns([
                    pl.col('reply').struct.field('parent').struct.field('record').struct.field('text').alias('ParentDocument'),
                    pl.col('text').alias('Document'),
                    pl.col('date').str.to_datetime().alias('createtime'),
                    pl.lit('bluesky').alias('platform')
                ])
            except pl.exceptions.StructFieldNotFoundError:
                df = df.with_columns([
                    pl.lit(None).alias('ParentDocument'),
                    pl.col('text').alias('Document'),
                    pl.col('date').str.to_datetime().alias('createtime'),
                    pl.lit('bluesky').alias('platform')
                ])
        else:
            raise NotImplementedError("Haven't implemented this platform")
        
        return df.select(['createtime', 'Document', 'ParentDocument', 'id', 'platform', 'seed'])

def join_text(df: pl.DataFrame):
    return df.with_columns(
        pl.when(pl.col('ParentDocument').is_not_null())\
            .then(pl.concat_str(['ParentDocument', 'Document'], separator="; "))\
            .otherwise(pl.col('Document'))\
            .alias('AllText')
    )

def process_month(month_files_df: pl.DataFrame, dir_path, save_path, platform_handler: PlatformHandler, finetune_kwargs, config):
    try:
        year = month_files_df['year'][0]
        month = month_files_df['month'][0]
        date_str = f"{year}_{month:02d}"
        print(f"Processing {date_str}")

        df = pl.DataFrame()
        pbar = tqdm(total=len(month_files_df), desc='Loading raw files')
        for platform in month_files_df['platform'].unique().to_list():
            platform_df = pl.DataFrame()
            for file in month_files_df.filter(pl.col('platform') == platform).to_dicts():
                pbar.update(1)
                file_name = file['file']
                try:
                    batch_df = pl.read_parquet(f'{dir_path}/{file_name}')
                    platform_df = pl.concat([platform_df, batch_df], how='diagonal_relaxed')
                except Exception as ex:
                    print(ex)
                    continue
            try:
                platform_df = platform_handler.format_platform_data(platform_df, platform)
                df = pl.concat([df, platform_df], how='diagonal_relaxed')
            except Exception as ex:
                print(ex)
                continue

        df = df.filter(pl.col('seed').struct.field('MainType').is_in(['influencer', 'politician', 'foreign']))

        # concatenate 
        df = join_text(df)

        # filter to politicians and influencers
        df = df.filter(pl.col('AllText').is_not_null())

        if len(df) == 0:
            return

        # check if text already processed
        target_path = os.path.join(save_path, f'targets_{date_str}.parquet.zstd')
        if os.path.exists(target_path):
            target_df = pl.read_parquet(target_path)
            if 'ParentDocument' not in target_df.columns:
                target_df = target_df.with_columns(pl.lit(None).alias('ParentDocument'))
            if 'AllText' not in target_df.columns:
                target_df = join_text(target_df)

            df = df.join(target_df, on=['id', 'platform'], how='left')
            existing_df = df.filter((pl.col('Document').fill_null('') == pl.col('Document_right').fill_null('')) & ((pl.col('ParentDocument').fill_null('') == pl.col('ParentDocument_right').fill_null(''))))\
                .select(['id', 'seed', 'createtime', 'platform', 'Document', 'ParentDocument', 'Targets', 'finetune_kwargs'])
            df = df.filter((pl.col('Document').fill_null('') != pl.col('Document_right').fill_null('')) | ((pl.col('ParentDocument').fill_null('') != pl.col('ParentDocument_right').fill_null(''))))\
                .drop(['Document_right', 'ParentDocument_right', 'AllText_right'])
        else:
            existing_df = None

        unique_df = df.unique('AllText')

        if len(unique_df) == 0:
            return

        docs = unique_df['AllText'].to_list()

        model = StanceMining(
            # target_extraction_model_kwargs={'enforce_eager': True},
            target_extraction_finetune_kwargs=finetune_kwargs,
            stance_target_type=config.stance_target_type,
            verbose=True,
        )

        doc_df = model.get_base_targets(docs)

        # filter out some targets
        text_col = 'text' if 'text' in doc_df.columns else 'Document'
        target_df = doc_df.explode('Targets').rename({'Targets': 'Target'})
        target_df = remove_bad_targets(target_df)
        doc_df = target_df.group_by(['ID', text_col]).agg(pl.col('Target')).rename({'Target': 'Targets'})

        target_df = unique_df.select(['id', 'platform', 'Document', 'AllText']).join(doc_df.select([text_col, 'Targets']), left_on='AllText', right_on=text_col, how='left').drop('AllText')
        target_df = target_df.with_columns(pl.lit(finetune_kwargs).alias('finetune_kwargs'))

        df = df.select(['id', 'seed', 'createtime', 'platform', 'Document', 'ParentDocument']).join(target_df.select(['id', 'platform', 'Targets', 'finetune_kwargs']), on=['id', 'platform'], how='left')
        if existing_df is not None:
            df = pl.concat([df, existing_df])

        df.write_parquet(target_path, compression='zstd')
    except Exception as e:
        print(f"Error processing {date_str}: exc type: {type(e).__name__}, message: {str(e)}")
        print(f"Traceback: {tb.format_exc()}")
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()  # Clear CUDA cache
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    dir_path = '../sitrep/data/digital_trace/raw_platforms'

    platforms = ['twitter', 'instagram', 'instagram-meta', 'tiktok', 'bluesky']

    data_files = os.listdir(dir_path)
    file_df = pl.DataFrame({'file': data_files})
    file_df = file_df.filter(pl.col('file').str.contains_any(platforms))
    file_df = file_df.filter(pl.col('file').str.ends_with('parquet.zstd'))

    # parse out date from file name
    file_df = file_df.with_columns([
            pl.col('file').str.split('_').list.get(1).str.split('.').list.get(0).str.split('-').alias('date_numbers'),
            pl.col('file').str.split('_').list.get(0).alias('platform')
        ])\
        .with_columns([
            pl.col('date_numbers').list.get(0).cast(pl.UInt16).alias('year'),
            pl.col('date_numbers').list.get(1).cast(pl.UInt8).alias('month'),
            pl.col('date_numbers').list.get(2).cast(pl.UInt8).alias('day')
        ])
    
    finetune_kwargs = {
        'model_path': config.model_name,
        'base_model_name': config.base_model_name,
        'prompting_method': 'stancemining',
        'classification_method': 'generation',
        'generation_method': 'list'
    }

    save_path = config.base_target_path

    platform_handler = PlatformHandler()

    # group by month
    file_df = file_df.filter((pl.col('year') <= 2023) & (pl.col('month') <= 11)) # remove
    for month_files_df in file_df.sort(['year', 'month'], descending=True).partition_by(['year', 'month']):
        process_month(month_files_df, dir_path, save_path, platform_handler, finetune_kwargs, config)

if __name__ == '__main__':
    main()