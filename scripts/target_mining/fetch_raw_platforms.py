"""Download MEO platform data into the raw_platforms layout read by extract_stance_targets.py.

Replaces the sitrep dashboard-API fetch (scripts/data_prep/fetch_data.py): it queries
Elasticsearch directly, pulls only the fields the stance pipeline consumes, and writes
one {platform}_{YYYY-MM-DD}.parquet.zstd per platform per day.
"""

import argparse
import datetime
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
import requests
from tqdm import tqdm

# Kibana's search proxy caps a single response at this many hits and exposes no
# point-in-time API, so windows are bisected until they fit rather than paginated.
WINDOW_LIMIT = 10000
MAX_RETRIES = 5

SEED_FIELDS = {
    'ID': pl.Int64,
    'SeedID': pl.Int64,
    'SeedName': pl.String,
    'MainType': pl.String,
    'SubType': pl.String,
    'Platform': pl.String,
    'Handle': pl.String,
    'PlatformHandleID': pl.String,
    'Collection': pl.String,
    'Province': pl.String,
    'ElectoralDistrict': pl.String,
    'Party': pl.String,
    'FederalParty': pl.String,
    'ProvincialParty': pl.String,
    'NewsOutletCategory': pl.String,
    'SeedStatus': pl.Int64,
    'HandleStatus': pl.Int64,
    'InfoStartDate': pl.String,
    'InfoEndDate': pl.String,
    'LoggedDate': pl.String,
}
SEED_SCHEMA = pl.Struct(SEED_FIELDS)


def coerce_seed(seed):
    seed = seed or {}
    row = {}
    for name, dtype in SEED_FIELDS.items():
        value = seed.get(name)
        if value is None or value == '':
            row[name] = None if dtype != pl.String else value
        elif dtype == pl.Int64:
            try:
                row[name] = int(value)
            except (TypeError, ValueError):
                row[name] = None
        else:
            row[name] = str(value)
    return row


def _get(source, *path):
    for key in path:
        if not isinstance(source, dict):
            return None
        source = source.get(key)
    return source


def twitter_row(src):
    return {
        'id': src.get('id'),
        'date': src.get('date'),
        'rawContent': src.get('rawContent'),
        'inReplyToTweetId': src.get('inReplyToTweetId'),
        'quotedTweet': {'rawContent': _get(src, 'quotedTweet', 'rawContent')},
        'seed': coerce_seed(src.get('seed')),
    }


def bluesky_row(src):
    return {
        'id': src.get('id'),
        'date': src.get('date'),
        'text': src.get('text'),
        'reply': {'parent': {'record': {'text': _get(src, 'reply', 'parent', 'record', 'text')}}},
        # a bare repost carries no text of its own; a quote post carries both
        'original_post_text': src.get('original_post_text'),
        'embed_title': _get(src, 'embed', 'external', 'title'),
        'embed_description': _get(src, 'embed', 'external', 'description'),
        'seed': coerce_seed(src.get('seed')),
    }


def instagram_row(src):
    conf = src.get('ocr_mean_conf')
    return {
        'id': src.get('id'),
        'date': src.get('date'),
        'caption': {'text': _get(src, 'caption', 'text')},
        'taken_at': src.get('taken_at'),
        'seed': coerce_seed(src.get('seed')),
        'ocr_text': src.get('ocr_text'),
        'ocr_mean_conf': float(conf) if conf is not None else None,
        'ocr_n_lines': src.get('ocr_n_lines'),
        'ocr_n_images': src.get('ocr_n_images'),
    }


def instagram_meta_row(src):
    return {
        'id': src.get('id'),
        'date': src.get('date'),
        'description': src.get('description'),
        'seed': coerce_seed(src.get('seed')),
    }


def tiktok_row(src):
    segments = _get(src, 'transcripts', 'segments')
    embeds = src.get('speaker_embeddings')
    return {
        'video_id': src.get('id.keyword') or src.get('video_id'),
        'date': src.get('date'),
        'createtime': src.get('createtime'),
        'desc': src.get('desc'),
        'author_name': src.get('author_name'),
        'transcripts': {'segments': [
            {'start': float(s['start']) if s.get('start') is not None else None,
             'text': s.get('text'), 'speaker': s.get('speaker')}
            for s in segments
        ]} if segments else None,
        # unmapped in Elasticsearch, so it only ever comes back through _source
        'speaker_embeddings': [
            {'speaker_index': e.get('speaker_index'),
             'embedding': [float(x) for x in (e.get('embedding') or [])]}
            for e in embeds
        ] if embeds else None,
        'seed': coerce_seed(src.get('seed')),
    }


PLATFORMS = {
    'twitter': {
        'index': 'phh_twitter',
        'source': ['id', 'date', 'rawContent', 'inReplyToTweetId', 'quotedTweet.rawContent', 'seed'],
        'row': twitter_row,
        'schema': {
            'id': pl.String,
            'date': pl.String,
            'rawContent': pl.String,
            'inReplyToTweetId': pl.String,
            'quotedTweet': pl.Struct({'rawContent': pl.String}),
            'seed': SEED_SCHEMA,
        },
    },
    'bluesky': {
        'index': 'phh_bluesky',
        'source': ['id', 'date', 'text', 'reply.parent.record.text', 'original_post_text',
                   'embed.external.title', 'embed.external.description', 'seed'],
        'row': bluesky_row,
        'schema': {
            'id': pl.String,
            'date': pl.String,
            'text': pl.String,
            'reply': pl.Struct({'parent': pl.Struct({'record': pl.Struct({'text': pl.String})})}),
            'original_post_text': pl.String,
            'embed_title': pl.String,
            'embed_description': pl.String,
            'seed': SEED_SCHEMA,
        },
    },
    'instagram': {
        'index': 'phh_instagram_scraper',
        'source': ['id', 'date', 'caption.text', 'taken_at', 'seed',
                   'ocr_text', 'ocr_mean_conf', 'ocr_n_lines', 'ocr_n_images'],
        'row': instagram_row,
        'schema': {
            'id': pl.String,
            'date': pl.String,
            'caption': pl.Struct({'text': pl.String}),
            'taken_at': pl.Int64,
            'seed': SEED_SCHEMA,
            'ocr_text': pl.String,
            'ocr_mean_conf': pl.Float64,
            'ocr_n_lines': pl.Int64,
            'ocr_n_images': pl.Int64,
        },
    },
    'instagram-meta': {
        'index': 'phh_instagram',
        'source': ['id', 'date', 'description', 'seed'],
        'row': instagram_meta_row,
        'schema': {
            'id': pl.String,
            'date': pl.String,
            'description': pl.String,
            'seed': SEED_SCHEMA,
        },
    },
    'tiktok': {
        'index': 'phh_tiktok_embed',
        'source': ['video_id', 'date', 'createtime', 'desc', 'author_name',
                   'transcripts.segments.start', 'transcripts.segments.text',
                   'transcripts.segments.speaker', 'speaker_embeddings', 'seed'],
        'docvalues': ['id.keyword'],
        'row': tiktok_row,
        'schema': {
            'video_id': pl.String,
            'date': pl.String,
            'createtime': pl.String,
            'desc': pl.String,
            'author_name': pl.String,
            'transcripts': pl.Struct({'segments': pl.List(pl.Struct({
                'start': pl.Float64, 'text': pl.String, 'speaker': pl.String}))}),
            'speaker_embeddings': pl.List(pl.Struct({
                'speaker_index': pl.Int64, 'embedding': pl.List(pl.Float32)})),
            'seed': SEED_SCHEMA,
        },
    },
}

# Mirrors the seed filter extract_stance_targets.py applies after loading.
SEED_QUERY = {
    'bool': {
        'should': [
            {'terms': {'seed.MainType.keyword': ['influencer', 'politician']}},
            {'bool': {
                'must': [{'term': {'seed.MainType.keyword': 'foreign'}}],
                'must_not': [{'terms': {'seed.SubType.keyword': ['media', 'state']}}],
            }},
        ],
        'minimum_should_match': 1,
    }
}


def scope(query, start_ms, end_ms):
    window = {'range': {'date': {'gte': start_ms, 'lt': end_ms, 'format': 'epoch_millis'}}}
    return {'bool': {'must': [query, window]}} if query else window


class Elastic:
    def __init__(self, url, key):
        self.url = url
        self.headers = {
            'Authorization': f'ApiKey {key}',
            'kbn-xsrf': 'true',
            'elastic-api-version': '1',
            'Content-Type': 'application/json',
        }
        self._local = threading.local()

    @property
    def session(self):
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
        return self._local.session

    def search(self, index, body):
        payload = json.dumps({'params': {'index': index, 'ignore_unavailable': False, 'body': body}})
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.post(self.url, headers=self.headers, data=payload, timeout=180)
            except requests.RequestException:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
                continue
            if response.status_code in (429, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            if response.status_code >= 400:
                raise RuntimeError(f'{response.status_code} from Elasticsearch: {response.text[:300]}')
            return response.json()['rawResponse']

    def count(self, index, query):
        return self.search(index, {'size': 0, 'query': query})['hits']['total']

    def collect(self, index, query, source, start_ms, end_ms, total, docvalues=None):
        """Return every document in [start_ms, end_ms), halving the window until it fits one response."""
        if total == 0:
            return []
        if total <= WINDOW_LIMIT:
            return self._page(index, query, source, start_ms, end_ms, total, docvalues)
        mid_ms = start_ms + (end_ms - start_ms) // 2
        if mid_ms <= start_ms or mid_ms >= end_ms:
            print(f'{index}: {total} docs share timestamp {start_ms}, keeping {WINDOW_LIMIT}')
            return self._page(index, query, source, start_ms, end_ms, WINDOW_LIMIT, docvalues)
        left_total = self.count(index, scope(query, start_ms, mid_ms))
        return (self.collect(index, query, source, start_ms, mid_ms, left_total, docvalues)
                + self.collect(index, query, source, mid_ms, end_ms, total - left_total, docvalues))

    def _page(self, index, query, source, start_ms, end_ms, size, docvalues):
        body = {'size': size, '_source': source, 'query': scope(query, start_ms, end_ms)}
        if docvalues:
            body['docvalue_fields'] = docvalues
        return [merge_docvalues(hit) for hit in self.search(index, body)['hits']['hits']]


def merge_docvalues(hit):
    """Fold docvalue_fields into _source.

    The Kibana proxy parses responses in JavaScript, so an id stored as a JSON
    number loses precision past 2^53 (tiktok's 19-digit ids round to the nearest
    representable double). Docvalues on the keyword sub-field come back as strings
    and survive intact.
    """
    src = dict(hit.get('_source') or {})
    for name, values in (hit.get('fields') or {}).items():
        if values:
            src[name] = values[0]
    return src


def epoch_ms(day):
    return int(datetime.datetime.combine(day, datetime.time(), datetime.timezone.utc).timestamp() * 1000)


def fetch_day(es, platform, day, out_dir, seed_filter):
    spec = PLATFORMS[platform]
    out_path = os.path.join(out_dir, f'{platform}_{day.isoformat()}.parquet.zstd')
    start_ms, end_ms = epoch_ms(day), epoch_ms(day + datetime.timedelta(days=1))
    query = SEED_QUERY if seed_filter else None
    total = es.count(spec['index'], scope(query, start_ms, end_ms))
    sources = es.collect(spec['index'], query, spec['source'], start_ms, end_ms, total, spec.get('docvalues'))
    rows = [spec['row'](src) for src in sources]
    df = pl.DataFrame(rows, schema=spec['schema'])
    # rename last so an interrupted run leaves no half-written file for the next run to skip
    df.write_parquet(out_path + '.tmp', compression='zstd')
    os.replace(out_path + '.tmp', out_path)
    return platform, day, len(df)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-date', required=True, help='first day to fetch, YYYY-MM-DD')
    parser.add_argument('--end-date', required=True, help='last day to fetch (exclusive), YYYY-MM-DD')
    parser.add_argument('--out-dir', default='../sitrep/data/digital_trace/raw_platforms')
    parser.add_argument('--platforms', nargs='+', default=list(PLATFORMS), choices=list(PLATFORMS))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true', help='refetch days that already have a file')
    parser.add_argument('--seed-filter', action='store_true',
                        help='fetch only the seeds extract_stance_targets.py keeps; shrinks the '
                             'twitter table it uses to resolve reply parents')
    args = parser.parse_args()

    key, url = os.environ.get('ES_KEY'), os.environ.get('ES_URL')
    env_path = os.path.expanduser('~/.claude/.elastic.env')
    if not key and os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if '=' not in line:
                    continue
                name, _, value = line.replace('export ', '', 1).strip().partition('=')
                if name == 'ES_KEY':
                    key = value.strip('"\'')
                elif name == 'ES_URL' and not url:
                    url = value.strip('"\'')
    if not key or not url:
        raise SystemExit('set ES_KEY and ES_URL, or configure ~/.claude/.elastic.env')

    start = datetime.date.fromisoformat(args.start_date)
    end = datetime.date.fromisoformat(args.end_date)
    days = [start + datetime.timedelta(days=i) for i in range((end - start).days)]
    if not days:
        raise SystemExit('--end-date must be after --start-date')

    os.makedirs(args.out_dir, exist_ok=True)
    tasks = [(platform, day) for platform in args.platforms for day in days
             if args.overwrite or not os.path.exists(
                 os.path.join(args.out_dir, f'{platform}_{day.isoformat()}.parquet.zstd'))]
    if not tasks:
        print('nothing to fetch')
        return

    es = Elastic(url, key)
    counts = {platform: 0 for platform in args.platforms}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_day, es, platform, day, args.out_dir, args.seed_filter): (platform, day)
                   for platform, day in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc='fetching'):
            platform, day = futures[future]
            try:
                _, _, n = future.result()
                counts[platform] += n
            except Exception as ex:
                print(f'{platform} {day}: {type(ex).__name__}: {ex}')

    for platform, n in counts.items():
        print(f'{platform}: {n} documents')


if __name__ == '__main__':
    main()
