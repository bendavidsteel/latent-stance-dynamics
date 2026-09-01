import datetime
import gc
import os
import time
import traceback as tb

import hydra
import polars as pl
import torch
from tqdm import tqdm

from stancemining.main import StanceMining

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

OUT_COLUMNS = ['id', 'seed', 'createtime', 'platform', 'Document', 'ParentDocument', 'Targets', 'finetune_kwargs']

# Instagram image OCR. ocr_mean_conf is a per-post mean over detected lines; below
# this the text is mostly garbled glyphs. Counting only runs of letters rejects the
# digit-and-punctuation noise ("0v 5 2 j 651 1 4") that a plain word count lets past.
OCR_MIN_CONF = 0.75
OCR_MIN_WORDS = 3
OCR_WORD = r'[^\W\d_]{3,}'
OCR_MAX_CHARS = 1000

# A document with no letters gives the model nothing to read, and it answers with a
# canned target list rather than an empty one, so those rows are dropped up front.
MIN_DOC_LETTERS = 1

# vLLM builds a fresh engine for every month, and that start-up occasionally loses a
# race for GPU memory; without a retry the month is skipped and silently keeps stale targets.
ENGINE_RETRIES = 3


def _blank_to_null(df: pl.DataFrame, column: str):
    """Column as an expression, blanks as null; a literal null if it isn't there."""
    if column not in df.columns:
        return pl.lit(None, dtype=pl.String)
    e = pl.col(column).str.strip_chars()
    return pl.when(e.str.len_chars() > 0).then(e).otherwise(None)


def _joined_blank_to_null(df: pl.DataFrame, columns):
    parts = [_blank_to_null(df, c) for c in columns]
    joined = pl.concat_str(parts, separator=' ', ignore_nulls=True).str.strip_chars()
    return pl.when(joined.str.len_chars() > 0).then(joined).otherwise(None)


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

        self.twitter_df = pl.read_parquet(file_df.select(pl.format('{}/{}', pl.lit(dir_path), pl.col('file')).alias('file_path'))['file_path'].to_list(), columns=['id', 'rawContent'])\
            .with_columns(pl.col('id').cast(pl.UInt64))


    def format_platform_data(self, df: pl.DataFrame, platform):
        if platform == 'tiktok':
            df = df.with_columns(pl.col('video_id').cast(pl.UInt64))
            # every video has a caption but only some have a transcript, so the
            # transcript is assembled on its own and joined back on
            meta_df = df.select(['video_id', 'createtime', 'seed', 'desc']).unique('video_id', keep='first')
            spoken_df = df.filter(pl.col('transcripts').is_not_null())

            transcript_df = pl.DataFrame(schema={'video_id': pl.UInt64, 'text': pl.String})
            if len(spoken_df) > 0:
                unique_speaker_df = spoken_df.select([
                    'video_id',
                    (pl.col('transcripts').struct.field('segments').list.eval(pl.col('').struct.field('speaker')).list.unique().list.len() > 1).alias('multiple_speakers')
                ])
                # get speaker indexs
                seg_df = spoken_df.with_columns(pl.col('transcripts').struct.field('segments'))\
                    .explode('segments')\
                    .with_columns(pl.col('segments').struct.unnest())\
                    .with_columns(pl.col('speaker').str.split('_').list.get(-1).cast(pl.UInt32).alias('speaker_index'))
                # find cases where a speaker index is also the author of the video
                seg_df = seg_df.join(self.tiktok_speaker_author_df, on=['video_id', 'speaker_index'])\
                    .with_columns([
                        pl.when(pl.col('is_author'))\
                        .then(pl.lit('author'))\
                        .otherwise(pl.col('speaker'))\
                        .alias('speaker'),
                        pl.col('text').str.strip_chars()
                    ])
                seg_df = seg_df.join(unique_speaker_df, on='video_id', how='left')
                seg_df = add_dialogue_turn(seg_df)
                # group speaker sections together
                seg_df = seg_df.sort(['video_id', 'start'])\
                    .group_by(['video_id', 'multiple_speakers', 'dialogue_turn'], maintain_order=True)\
                    .agg([pl.col('text'), pl.col('start').min(), pl.col('is_author').first()])\
                    .with_columns(pl.col('text').list.join(' ').str.strip_chars())

                seg_df = seg_df.with_columns(
                    pl.when(pl.col('is_author') | ~pl.col('multiple_speakers'))\
                        .then(pl.col('text'))\
                        .otherwise(pl.format('"{}"', 'text'))\
                        .alias('formatted_text')
                )
                # TODO fix dialogue formatting
                if len(seg_df) > 0:
                    transcript_df = seg_df.sort(['video_id', 'start'])\
                        .group_by('video_id', maintain_order=True)\
                        .agg(pl.col('formatted_text'))\
                        .with_columns(pl.col('formatted_text').list.join(' ').str.strip_chars().alias('text'))\
                        .select(['video_id', 'text'])

            df = meta_df.join(transcript_df, on='video_id', how='left')
            desc = pl.col('desc').str.strip_chars()
            desc = pl.when(desc.str.len_chars() > 0).then(desc).otherwise(None)
            spoken = pl.when(pl.col('text').str.len_chars() > 0).then(pl.col('text')).otherwise(None)
            df = df.with_columns([
                pl.col('video_id').alias('id').cast(pl.String),
                # caption first, then whatever was said in the video
                pl.when(desc.is_not_null() & spoken.is_not_null())
                    .then(pl.concat_str([desc, spoken], separator='\n'))
                    .otherwise(pl.coalesce([desc, spoken]))
                    .alias('Document'),
                pl.lit(None).alias('ParentDocument'),
                pl.col('createtime').str.to_datetime().dt.convert_time_zone('UTC'),
                pl.lit('tiktok').alias('platform')]
            )
        elif platform == 'instagram':
            caption = pl.col('caption').struct.field('text').str.strip_chars()
            caption = pl.when(caption.str.len_chars() > 0).then(caption).otherwise(None)
            if 'ocr_text' in df.columns:
                ocr = pl.when(pl.col('ocr_mean_conf') >= OCR_MIN_CONF)\
                    .then(pl.col('ocr_text').str.replace_all(r'\s+', ' ').str.strip_chars())\
                    .otherwise(None)
                ocr = pl.when(ocr.str.count_matches(OCR_WORD) >= OCR_MIN_WORDS).then(ocr).otherwise(None)
                ocr = ocr.str.slice(0, OCR_MAX_CHARS)
            else:
                ocr = pl.lit(None, dtype=pl.String)
            df = df.with_columns([
                pl.lit(None).alias('ParentDocument'),
                # a post can carry its whole message in the image, so OCR alone still counts
                pl.when(caption.is_not_null() & ocr.is_not_null())
                    .then(pl.concat_str([caption, ocr], separator='\n'))
                    .otherwise(pl.coalesce([caption, ocr]))
                    .alias('Document'),
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
            quoted_dtype = df.schema['quotedTweet']
            if isinstance(quoted_dtype, pl.Struct) and 'rawContent' in quoted_dtype.to_schema():
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
            # replace_all, since a tweet can carry several links; the retweet prefix is
            # anchored and case-insensitive because it is always literally "RT @user:"
            df = df.with_columns([
                pl.col(c).str.replace_all(r'https://t\.co/\w+', 'URL')
                         .str.replace_all(r'(?i)^rt @\w+:?\s*', '')
                         .str.strip_chars()
                for c in ('Document', 'ParentDocument')
            ])
        elif platform == 'bluesky':
            text = pl.col('text').str.strip_chars()
            text = pl.when(text.str.len_chars() > 0).then(text).otherwise(None)
            orig = _blank_to_null(df, 'original_post_text')
            link = _joined_blank_to_null(df, ['embed_title', 'embed_description'])
            parent = pl.lit(None, dtype=pl.String)
            reply_dtype = df.schema.get('reply')
            if isinstance(reply_dtype, pl.Struct) and 'parent' in reply_dtype.to_schema():
                parent = pl.col('reply').struct.field('parent').struct.field('record').struct.field('text')
            df = df.with_columns([
                # a bare repost has no text of its own, so the reposted content is the document
                pl.coalesce([text, orig, link]).alias('Document'),
                # ...but when the post does have text, the quoted post is context, as on twitter
                pl.coalesce([parent, pl.when(text.is_not_null()).then(orig)]).alias('ParentDocument'),
                pl.col('date').str.to_datetime(time_zone='UTC').alias('createtime'),
                pl.lit('bluesky').alias('platform')
            ])
        else:
            raise NotImplementedError("Haven't implemented this platform")
        
        return df.select(['createtime', 'Document', 'ParentDocument', 'id', 'platform', 'seed'])
    
    def get_platform_columns(self, platform):
        if platform == 'tiktok':
            return ['video_id', 'createtime', 'seed', 'transcripts', 'desc']
        elif platform == 'instagram':
            return ['caption', 'taken_at', 'seed', 'id', 'ocr_text', 'ocr_mean_conf']
        elif platform == 'instagram-meta':
            return ['description', 'date', 'seed', 'id']
        elif platform == 'twitter':
            return ['id', 'inReplyToTweetId', 'quotedTweet', 'rawContent', 'date', 'seed']
        elif platform == 'bluesky':
            return ['reply', 'text', 'date', 'seed', 'id',
                    'original_post_text', 'embed_title', 'embed_description']
        else:
            raise NotImplementedError("Haven't implemented this platform")

def join_text(df: pl.DataFrame):
    return df.with_columns(
        pl.when(pl.col('ParentDocument').is_not_null())\
            .then(pl.concat_str(['ParentDocument', 'Document'], separator="; "))\
            .otherwise(pl.col('Document'))\
            .alias('AllText')
    )

def process_month(month_files_df: pl.DataFrame, model: StanceMining, dir_path, save_path, platform_handler: PlatformHandler, finetune_kwargs, config, embedding_model=None):
    date_str = '<unknown>'
    try:
        year = month_files_df['year'][0]
        month = month_files_df['month'][0]
        date_str = f"{year}_{month:02d}"
        print(f"Processing {date_str}")

        df = pl.DataFrame()
        pbar = tqdm(total=len(month_files_df), desc='Loading raw files')
        for platform in month_files_df['platform'].unique().to_list():
            platform_df = pl.DataFrame()
            platform_columns = platform_handler.get_platform_columns(platform)
            file_paths = month_files_df.filter(pl.col('platform') == platform)\
                .select(pl.format('{}/{}', pl.lit(dir_path), pl.col('file')).alias('file_path'))['file_path'].to_list()
            try:
                platform_df = pl.read_parquet(file_paths, columns=platform_columns, missing_columns='insert')
            except Exception:
                dfs = []
                for file_path in file_paths:
                    try:
                        file_df = pl.read_parquet(file_path, columns=platform_columns, missing_columns='insert')
                        dfs.append(file_df)
                    except Exception as ex:
                        print(f"Error reading file {file_path} for platform {platform}: {ex}")
                        continue
                if len(dfs) == 0:
                    print(f"No valid files for platform {platform} in {date_str}")
                    pbar.update(len(file_paths))
                    continue
                platform_df = pl.concat(dfs, how='diagonal_relaxed')
            pbar.update(len(file_paths))
            try:
                platform_df = platform_handler.format_platform_data(platform_df, platform)
                df = pl.concat([df, platform_df], how='diagonal_relaxed')
            except Exception as ex:
                print(f"Error processing platform {platform} for {date_str}: {ex}")
                continue

        df = df.filter(pl.col('seed').struct.field('MainType').is_in(['influencer', 'politician']) | ((pl.col('seed').struct.field('MainType') == 'foreign') & (~pl.col('seed').struct.field('SubType').is_in(['media', 'state']))))

        # concatenate 
        df = join_text(df)

        # filter to politicians and influencers
        df = df.filter(pl.col('AllText').str.count_matches(r'[^\W\d_]') >= MIN_DOC_LETTERS)

        if len(df) == 0:
            return

        # reuse the previous run's extractions wherever both the text and the
        # extracting model are unchanged, so one file never mixes two models
        target_path = os.path.join(save_path, f'targets_{date_str}.parquet.zstd')
        existing_df = None
        if os.path.exists(target_path):
            target_df = pl.read_parquet(target_path)
            if 'ParentDocument' not in target_df.columns:
                target_df = target_df.with_columns(pl.lit(None, dtype=pl.String).alias('ParentDocument'))

            reuse_df = target_df.filter(pl.col('Targets').is_not_null())\
                .select([
                    'id',
                    'platform',
                    pl.col('Document').alias('PriorDocument'),
                    pl.col('ParentDocument').alias('PriorParentDocument'),
                    'Targets',
                    'finetune_kwargs'
                ])\
                .unique(['id', 'platform'])

            df = df.join(reuse_df, on=['id', 'platform'], how='left')
            # eq_missing: null/null counts as unchanged, null/value does not
            reused = pl.col('Targets').is_not_null()\
                & pl.col('Document').eq_missing(pl.col('PriorDocument'))\
                & pl.col('ParentDocument').eq_missing(pl.col('PriorParentDocument'))\
                & (pl.col('finetune_kwargs').struct.field('model_path') == finetune_kwargs['model_path'])
            existing_df = df.filter(reused).select(OUT_COLUMNS)
            df = df.filter(~reused).drop(['PriorDocument', 'PriorParentDocument', 'Targets', 'finetune_kwargs'])
            print(f"{date_str}: reusing {len(existing_df)}, extracting from {len(df)}")

        unique_df = df.unique('AllText')

        if len(unique_df) == 0:
            return

        docs = unique_df['AllText'].to_list()
        for attempt in range(ENGINE_RETRIES):
            try:
                doc_df = model.get_base_targets(docs, embedding_model=embedding_model)
                break
            except Exception as ex:
                if attempt == ENGINE_RETRIES - 1:
                    raise
                print(f'{date_str}: extraction attempt {attempt + 1} failed ({type(ex).__name__}: {ex}), retrying')
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(30)

        # key the results by text, so every document sharing an AllText keeps the extraction
        text_col = 'text' if 'text' in doc_df.columns else 'Document'
        target_df = doc_df.select([pl.col(text_col).alias('AllText'), 'Targets']).unique('AllText')

        df = df.join(target_df, on='AllText', how='left')\
            .with_columns(pl.lit(finetune_kwargs).alias('finetune_kwargs'))\
            .select(OUT_COLUMNS)
        if existing_df is not None and len(existing_df) > 0:
            df = pl.concat([df, existing_df], how='vertical_relaxed')

        # rename last so an interrupted write cannot destroy the previous run's targets
        df.write_parquet(target_path + '.tmp', compression='zstd')
        os.replace(target_path + '.tmp', target_path)
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

    model = StanceMining(
        target_extraction_finetune_kwargs=finetune_kwargs,
        target_extraction_model_kwargs={
            # fraction of the whole card, so it has to leave room for the embedding
            # engine's ~0.1 and for anything else sharing the GPU
            'gpu_memory_utilization': float(config.get('gpu_memory_utilization', 0.85)),
            'max_lora_rank': 8,
            'max_num_seqs': int(config.get('max_num_seqs', 512)),
        },
        stance_target_type=config.stance_target_type,
        verbose=True,
    )
    # built once and passed down; get_base_targets otherwise starts and discards
    # an embedding engine for every month
    embedding_model = model._get_embedding_model()

    # group by month
    month_dfs = file_df.sort(['year', 'month'], descending=True).partition_by(['year', 'month'])
    num_shards = int(config.get('num_shards', 1))
    if num_shards > 1:
        # each shard owns a disjoint set of months, so they never write the same file
        month_dfs = month_dfs[int(config.get('shard', 0))::num_shards]
    for month_files_df in month_dfs:
        process_month(month_files_df, model, dir_path, save_path, platform_handler, finetune_kwargs, config, embedding_model)

if __name__ == '__main__':
    main()