import io
import os

import boto3
from PIL import Image
import polars as pl
from tqdm import tqdm

import gc 
import json
import os

import dotenv
from moviepy import VideoFileClip
import numpy as np
from pyannote.audio import Pipeline
import pandas as pd
import torch
from tqdm import tqdm
import whisperx
from whisperx.audio import SAMPLE_RATE

def to_df(transcript_data):
    batch_transcript_df = pl.DataFrame(
        {
            'video_id': [d['video_id'] for d in transcript_data],
            'transcript': [d['transcript'] for d in transcript_data],
            'speaker_embeddings': [[d['speaker_embeddings'][i].astype(np.float64) for i in range(d['speaker_embeddings'].shape[0])] for d in transcript_data],
        },
        schema={
            'video_id': pl.UInt64,
            'transcript': pl.Struct({
                'segments': pl.List(pl.Struct({
                    'start': pl.Float64,
                    'end': pl.Float64,
                    'speaker': pl.String,
                    'text': pl.String
                }))
            }),
            'speaker_embeddings': pl.List(pl.Array(pl.Float64, 256))
        }
    )
    return batch_transcript_df

def try_create_audio(video_path, audio_file_path):
    try:
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(audio_file_path)
        return True
    except OSError:
        return False
    except AttributeError:
        return False

def apply_whisperx_pipeline(audio, model, diarize_model):
    device = "cuda" 
    batch_size = 16 # reduce if low on GPU mem

    # 1. Transcribe with original whisper (batched)
    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

    
    result = model.transcribe(audio, batch_size=batch_size)

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    # add min/max number of speakers if known
    audio_data = {
        'waveform': torch.from_numpy(audio[None, :]),
        'sample_rate': SAMPLE_RATE
    }
    segments, embeddings = diarize_model(audio_data, return_embeddings=True)
    diarize_segments = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
    diarize_segments['start'] = diarize_segments['segment'].apply(lambda x: x.start)
    diarize_segments['end'] = diarize_segments['segment'].apply(lambda x: x.end)
    
    result = whisperx.assign_word_speakers(diarize_segments, result)
    return result, diarize_segments, embeddings


def main():
    path = '../sitrep/data/digital_trace/raw_platforms'
    data_files = os.listdir(path)
    data_files = [f for f in data_files if f.endswith('.parquet.zstd') and 'tiktok' in f]

    bucket = 'media-data-pipeline-raw-data'
    prefix = 'tiktok/bytes/'

    s3 = boto3.client('s3')

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    media_files = []
    media_pbar = tqdm(desc='Finding media files')
    for page in pages:
        for obj in page['Contents']:
            media_files.append(obj['Key'])
            media_pbar.update(1)
    media_pbar.close()
    media_df = pl.DataFrame({'key': media_files})
    media_df = media_df.with_columns(pl.col('key').str.replace('tiktok/bytes/', '').alias('file_name'))\
        .with_columns(pl.col('file_name').str.split('.').list.get(0).cast(pl.UInt64).alias('video_id'))

    df = pl.DataFrame()
    for data_file in tqdm(data_files, desc='Loading data files'):
        batch_df = pl.read_parquet(f'{path}/{data_file}')
        df = pl.concat([df, batch_df], how='diagonal_relaxed')

    df = df.with_columns(pl.col('video_id').cast(pl.UInt64))
    df = df.join(media_df, on='video_id', how='left')
    df = df.filter(pl.col('file_name').is_not_null())

    transcripts_path = './data/tiktok/transcripts.parquet.zstd'
    if os.path.exists(transcripts_path):
        transcript_df = pl.read_parquet(transcripts_path)
        transcript_df = transcript_df.unique('video_id')
        df = df.filter(~pl.col('video_id').is_in(transcript_df['video_id']))
        # TODO remove already processed videos
        # TODO unique video_id
    else:
        transcript_df = pl.DataFrame()

    device = "cuda" 
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accu
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN).to(torch.device(device))

    tmp_path = './tmp'
    batch_size = 10
    transcript_data = []
    for video_data in tqdm(df.to_dicts(), desc='Extracting video data'):
        # download video file
        key = video_data['key']
        file_byte_string = s3.get_object(Bucket=bucket, Key=key)['Body'].read()

        # save video file
        video_path = os.path.join(tmp_path, video_data['file_name'])
        with open(video_path, 'wb') as f:
            f.write(file_byte_string)
            
        audio_file_path = video_path.replace('.mp4', '.mp3')

        # extract audio
        success = try_create_audio(video_path, audio_file_path)
        if not success:
            continue
        audio = whisperx.load_audio(audio_file_path)

        # delete audio and video files
        os.remove(audio_file_path)
        os.remove(video_path)

        try:
            result, diarize_segments, speaker_embeddings = apply_whisperx_pipeline(audio, model, diarize_model)
            transcript_data.append({
                'video_id': video_data['video_id'],
                'transcript': result,
                'speaker_embeddings': speaker_embeddings
            })
        except:
            continue

        if len(transcript_data) == batch_size:
            batch_transcript_df = to_df(transcript_data)
            transcript_df = pl.concat([transcript_df, batch_transcript_df], how='diagonal_relaxed')
            transcript_df.write_parquet(transcripts_path)
            transcript_data = []

    batch_transcript_df = to_df(transcript_data)
    transcript_df = pl.concat([transcript_df, batch_transcript_df], how='diagonal_relaxed')
    transcript_df.write_parquet(transcripts_path)

if __name__ == '__main__':
    dotenv.load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    main()