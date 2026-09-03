"""Translate the non-English stance targets into English.

Runs between extraction and de-duplication: a French and an English phrasing of the
same target only merge if they are in the same language first.
"""

import datetime
import os
import re

import hydra
import polars as pl
from tqdm import tqdm

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

TRANSLATION_FILE = 'target_translations.parquet.zstd'
START_DATE = datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc)

# Anything outside the Latin script is non-English whatever the detector thinks, so
# the detector only has to separate English from the other Latin-script languages.
NON_LATIN = r'[^\p{Latin}\p{Common}\p{Inherited}]'
HAS_LETTER = r'[^\W\d_]'

# Handles and username-like tokens are what the detector gets wrong most often, and
# there is nothing in them to translate anyway.
HANDLE = r'^@'
DIGIT_TOKEN = r'^\S*\d\S*$'

# Lingua returns no language at all when the runner-up is within this relative
# distance of the winner. Short phrases are easy to detect wrongly, and a wrong
# detection sends an English target through the translator, so leave those alone.
MIN_RELATIVE_DISTANCE = 0.4

# A translation of a short phrase is itself short. Anything longer is the model
# explaining itself, apologising, or looping, none of which is a target.
MAX_OUTPUT_TOKENS = 48
MAX_GROWTH = 3.0
MAX_EXTRA_CHARS = 24

# Asked to translate something untranslatable the model answers about the phrase
# rather than with one, and for a long target that reply is short enough to pass.
META_REPLY = (r"(?i)^(sorry\b|as an ai\b|note:|the phrase\b|this (is|phrase|appears|seems)\b"
              r"|it (is|appears|seems)\b|there is no\b|i (don't|do not|cannot|can't|am not|apologi))")

# Written out every chunk so a crash costs one chunk rather than the whole run.
CHUNK_SIZE = 100_000

# Longer than any real noun phrase, and the prompt has to stay inside max_model_len.
MAX_TARGET_CHARS = 300

PROMPT = (
    'Translate the following {language}phrase into English. '
    'Reply with only the translation: no quotes, no explanation, no notes. '
    'Write proper names the way they are normally written in English. '
    'If the phrase is already English, reply with it unchanged.\n\n'
    'Phrase: {target}'
)


def load_unique_targets(target_dir: str, start_date: datetime.datetime) -> pl.DataFrame:
    """Every distinct target in the monthly base target files, with its frequency."""
    files = [os.path.join(target_dir, f) for f in os.listdir(target_dir)
             if re.match(r'targets_\d{4}_\d{1,2}\.parquet\.zstd$', f)]
    return (pl.scan_parquet(files)
            .select(['createtime', 'Targets'])
            .filter(pl.col('createtime') >= start_date)
            .select(pl.col('Targets').explode().alias('Target'))
            .drop_nulls()
            .group_by('Target').len().rename({'len': 'Count'})
            .collect(engine='streaming'))


def detect_languages(df: pl.DataFrame, low_accuracy: bool = False) -> pl.DataFrame:
    """Add the detected language of each target, null where the detector is unsure."""
    from lingua import LanguageDetectorBuilder

    builder = LanguageDetectorBuilder.from_all_languages()\
        .with_minimum_relative_distance(MIN_RELATIVE_DISTANCE)\
        .with_preloaded_language_models()
    if low_accuracy:
        builder = builder.with_low_accuracy_mode()
    detector = builder.build()

    targets = df['Target'].to_list()
    languages = []
    for i in tqdm(range(0, len(targets), CHUNK_SIZE), desc='detecting'):
        batch = targets[i:i + CHUNK_SIZE]
        languages.extend(l.name.lower() if l is not None else None
                         for l in detector.detect_languages_in_parallel_of(batch))
    return df.with_columns(pl.Series('Language', languages, dtype=pl.String))


def needs_translation(df: pl.DataFrame, min_count: int) -> pl.Series:
    """Targets worth sending to the model: non-English, and made of actual words."""
    return (
        (pl.col('Count') >= min_count)
        & pl.col('Target').str.contains(HAS_LETTER)
        & (pl.col('Target').str.len_chars() <= MAX_TARGET_CHARS)
        & ~pl.col('Target').str.contains(HANDLE)
        & ~pl.col('Target').str.contains(DIGIT_TOKEN)
        & (pl.col('Target').str.contains(NON_LATIN)
           | (pl.col('Language').is_not_null() & (pl.col('Language') != 'english')))
    )


def clean_translation(text: str, target: str) -> str | None:
    """The model's reply as a target, or None if it did not answer with one."""
    # a preamble line ends in a colon ('Sure, here you go:'); the answer is what follows
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    lines = [l for l in lines if not l.endswith(':')] or lines
    text = re.sub(r'(?i)^(translation|english|phrase)\s*:\s*', '', lines[0] if lines else '').strip()
    text = text.strip('"“”‘’\'').strip().lower()
    if not text or not re.search(HAS_LETTER, text):
        return None
    if len(text) > MAX_GROWTH * len(target) + MAX_EXTRA_CHARS:
        return None
    if re.search(META_REPLY, text):
        return None
    return None if text == target.lower() else text


def translate(df: pl.DataFrame, config, out_path: str, done_df: pl.DataFrame) -> pl.DataFrame:
    """Translate every target in df, appending each chunk to the translation file."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=config.get('translate_model_name', 'Qwen/Qwen3.5-4B'),
        gpu_memory_utilization=float(config.get('gpu_memory_utilization', 0.85)),
        max_model_len=int(config.get('translate_max_model_len', 1024)),
        max_num_seqs=int(config.get('max_num_seqs', 512)),
        max_num_batched_tokens=int(config.get('max_num_batched_tokens', 16384)),
        # experimental for this model's linear-attention layers, and it silently
        # returns text belonging to other prompts
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)

    def to_prompt(target: str, language: str | None) -> str:
        content = PROMPT.format(language=f'{language} ' if language else '', target=target)
        return tokenizer.apply_chat_template(
            [{'role': 'user', 'content': content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    # Qwen3.5 reasons by default, which would cost far more tokens than the answer.
    example = to_prompt('les états-unis', 'french')
    assert '<think>\n\n</think>' in example, f'thinking is still on:\n{example}'
    print(f'example prompt:\n{example}\n')

    for i in tqdm(range(0, df.height, CHUNK_SIZE), desc='translating'):
        chunk = df.slice(i, CHUNK_SIZE)
        prompts = [to_prompt(t, l) for t, l in zip(chunk['Target'], chunk['Language'])]
        outputs = llm.generate(prompts, sampling_params)
        chunk = chunk.with_columns(pl.Series(
            'TargetEnglish',
            [clean_translation(o.outputs[0].text, t) for o, t in zip(outputs, chunk['Target'])],
            dtype=pl.String,
        ))
        done_df = pl.concat([done_df, chunk], how='diagonal_relaxed')
        done_df.write_parquet(out_path + '.tmp', compression='zstd')
        os.replace(out_path + '.tmp', out_path)

    return done_df


def apply_translations(df: pl.DataFrame, translation_df: pl.DataFrame) -> pl.DataFrame:
    """Swap each target for its English translation, dropping duplicates a merge creates."""
    translation_df = translation_df.filter(pl.col('TargetEnglish').is_not_null())\
        .select(['Target', 'TargetEnglish']).unique('Target')

    df = df.with_row_index()
    target_df = df.select(['index', 'Targets']).explode('Targets').rename({'Targets': 'Target'})
    target_df = target_df.join(translation_df, on='Target', how='left')\
        .select(['index', pl.coalesce(['TargetEnglish', 'Target']).alias('Target')])\
        .drop_nulls('Target')\
        .unique(['index', 'Target'])\
        .group_by('index').agg(pl.col('Target').alias('Targets'))
    return df.drop('Targets').join(target_df, on='index', how='left')\
        .with_columns(pl.col('Targets').fill_null([]))\
        .drop('index')


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    target_dir = config.base_target_path
    out_path = os.path.join(os.path.dirname(target_dir.rstrip('/')), TRANSLATION_FILE)

    df = load_unique_targets(target_dir, START_DATE)
    print(f'{df.height:,} unique targets, {df["Count"].sum():,} occurrences')

    df = detect_languages(df, low_accuracy=bool(config.get('translate_low_accuracy', False)))
    print(df.group_by('Language').agg(pl.len().alias('targets'), pl.col('Count').sum().alias('occurrences'))
            .sort('targets', descending=True).head(20))

    df = df.filter(needs_translation(df, int(config.get('translate_min_count', 1))))
    print(f'{df.height:,} targets to translate, {df["Count"].sum():,} occurrences')

    done_df = pl.read_parquet(out_path) if os.path.exists(out_path) else \
        pl.DataFrame(schema={'Target': pl.String, 'Count': pl.UInt32,
                             'Language': pl.String, 'TargetEnglish': pl.String})
    if done_df.height:
        df = df.join(done_df.select('Target'), on='Target', how='anti')
        print(f'{done_df.height:,} already translated, {df.height:,} left')

    if df.height == 0:
        print('nothing left to translate')
        return

    # most frequent first, so an interrupted run still covers the targets that matter
    df = translate(df.sort('Count', descending=True), config, out_path, done_df)

    translated = df.filter(pl.col('TargetEnglish').is_not_null())
    print(f'wrote {out_path}: {translated.height:,} of {df.height:,} targets translated')
    print(translated.sort('Count', descending=True).head(30))


if __name__ == '__main__':
    main()
