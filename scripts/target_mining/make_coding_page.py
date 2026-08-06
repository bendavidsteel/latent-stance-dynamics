"""Build a self-hosted HTML page for manually coding sampled stance targets.

Takes the CSV written by ``sample_classified_posts.py`` and emits a single
self-contained HTML file: one screen per post, with the post embedded as a
platform iframe and, beside it, each of that post's sampled stance targets with
two questions -- is the target relevant to the post (binary), and what is the
post's stance on the target.

Annotations are kept in the browser's localStorage as you go. Clicking "Autosave to
CSV" once picks a file that every later change is written to, so no manual export
step can be forgotten; the file is remembered across reloads. The manual CSV/JSON
export (and JSON re-import, to resume elsewhere) still works.

Autosave needs the File System Access API: a Chromium-based browser in a secure
context, which includes http://localhost. Serving the page from a plain http:// LAN
address disables it, and the page then warns before you leave with unsaved codes.

Example:
    python scripts/target_mining/make_coding_page.py \
        --input ./out/classified_post_sample.csv \
        --output ./out/coding_page.html
"""

import argparse
import csv
import html
import json
import os
import re
from collections import OrderedDict

STANCE_OPTIONS = ['FAVOR', 'AGAINST', 'NEUTRAL']

# only these are needed per post; the rest of the CSV columns come along as context
BSKY_RE = re.compile(r'bsky\.app/profile/([^/]+)/post/([^/?#]+)')
IG_RE = re.compile(r'instagram\.com/(?:p|reel|tv)/([^/?#]+)')
TIKTOK_RE = re.compile(r'tiktok\.com/@([^/]+)/video/(\d+)')
TWITTER_RE = re.compile(r'(?:twitter|x)\.com/([^/]+)/status/(\d+)')

# per-platform starting iframe height in px; embeds that report their own size
# override this at runtime
EMBED_HEIGHTS = {
    'bluesky': 420,
    'instagram': 760,
    'tiktok': 760,
    'twitter': 560,
}


def embed_url(platform, post_url):
    """Return the iframe src for a post, or None if we can't build one."""
    if platform == 'bluesky':
        m = BSKY_RE.search(post_url)
        if m:
            did, rkey = m.groups()
            return f'https://embed.bsky.app/embed/{did}/app.bsky.feed.post/{rkey}'
    elif platform == 'instagram':
        m = IG_RE.search(post_url)
        if m:
            return f'https://www.instagram.com/p/{m.group(1)}/embed/captioned/'
    elif platform == 'tiktok':
        m = TIKTOK_RE.search(post_url)
        if m:
            return f'https://www.tiktok.com/embed/v2/{m.group(2)}'
    elif platform == 'twitter':
        m = TWITTER_RE.search(post_url)
        if m:
            return f'https://platform.twitter.com/embed/Tweet.html?id={m.group(2)}&dnt=true'
    return None


def load_posts(input_path):
    """Group the (post, target) rows of the sample CSV into one record per post."""
    with open(input_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    posts = OrderedDict()
    for row in rows:
        key = (row['platform'], row['post_id'])
        post = posts.get(key)
        if post is None:
            src = embed_url(row['platform'], row['post_url'])
            post = posts[key] = {
                'post_key': f"{row['platform']}::{row['post_id']}",
                'post_id': row['post_id'],
                'platform': row['platform'],
                'post_url': row['post_url'],
                'embed_url': src,
                'embed_height': EMBED_HEIGHTS.get(row['platform'], 600),
                'createtime': row['createtime'],
                'year': row['year'],
                'seed_name': row['seed_name'],
                'handle': row['handle'],
                'main_type': row['main_type'],
                'sub_type': row['sub_type'],
                'party': row['party'],
                'actor_group': row['actor_group'],
                'province': row['province'],
                'electoral_district': row['electoral_district'],
                'post_text': row['post_text'],
                'parent_text': row['parent_text'],
                # sampling weights, absent from samples drawn before they were added
                'n_post_targets': row.get('n_post_targets', ''),
                'n_sampled_targets': row.get('n_sampled_targets', ''),
                'pair_weight': row.get('pair_weight', ''),
                'targets': [],
            }
        post['targets'].append({
            'pair_id': f"{row['platform']}::{row['post_id']}::{row['target']}",
            'target': row['target'],
            'model_stance': row['stance'],
        })

    return list(posts.values())


def render_page(posts, title):
    """Inline the posts as JSON into the single-file coding app."""
    payload = json.dumps({'posts': posts, 'stance_options': STANCE_OPTIONS},
                         ensure_ascii=False)
    # </script> inside the JSON would end the tag early
    payload = payload.replace('</', '<\\/')
    return PAGE_TEMPLATE.replace('__TITLE__', html.escape(title)) \
                        .replace('__DATA__', payload)


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #f5f5f4; --panel: #fff; --ink: #1c1c1a; --muted: #6b6b66;
    --line: #dcdcd8; --accent: #2563eb; --yes: #15803d; --no: #b91c1c;
    --favor: #15803d; --against: #b91c1c; --neutral: #6b6b66;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #17171a; --panel: #202024; --ink: #ededea; --muted: #9a9a94;
      --line: #34343a; --accent: #60a5fa; --yes: #4ade80; --no: #f87171;
      --favor: #4ade80; --against: #f87171; --neutral: #9a9a94;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  header {
    position: sticky; top: 0; z-index: 10; background: var(--panel);
    border-bottom: 1px solid var(--line); padding: 10px 16px;
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  }
  header h1 { font-size: 15px; margin: 0 8px 0 0; font-weight: 600; }
  .grow { flex: 1; }
  button, select, input[type=text] {
    font: inherit; color: inherit; background: var(--panel);
    border: 1px solid var(--line); border-radius: 6px; padding: 5px 10px;
    cursor: pointer;
  }
  input[type=text] { cursor: text; }
  button:hover { border-color: var(--accent); }
  button:disabled { opacity: .4; cursor: default; }
  .bar { height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; width: 160px; }
  .bar > div { height: 100%; background: var(--accent); width: 0; transition: width .2s; }
  main { max-width: 1400px; margin: 0 auto; padding: 16px; }
  .cols { display: grid; grid-template-columns: minmax(360px, 560px) 1fr; gap: 16px; align-items: start; }
  @media (max-width: 900px) { .cols { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
  .embed { padding: 0; overflow: hidden; }
  .embed iframe { width: 100%; border: 0; display: block; }
  .meta { font-size: 13px; color: var(--muted); }
  .meta b { color: var(--ink); font-weight: 600; }
  .chips { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
  .chip {
    font-size: 12px; border: 1px solid var(--line); border-radius: 999px;
    padding: 2px 9px; color: var(--muted);
  }
  .ptext { white-space: pre-wrap; margin-top: 10px; font-size: 14px; }
  .ptext.parent { border-left: 3px solid var(--line); padding-left: 10px; color: var(--muted); }
  .target { border: 1px solid var(--line); border-radius: 10px; padding: 12px; margin-bottom: 12px; background: var(--panel); }
  .target.active { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(37,99,235,.15); }
  .target h3 { margin: 0 0 2px; font-size: 17px; }
  .q { margin-top: 12px; }
  .q > span { display: block; font-size: 12px; color: var(--muted); margin-bottom: 5px; }
  .opts { display: flex; gap: 8px; flex-wrap: wrap; }
  .opt[aria-pressed=true] { color: #fff; border-color: transparent; }
  .opt.yes[aria-pressed=true] { background: var(--yes); }
  .opt.no[aria-pressed=true] { background: var(--no); }
  .opt.FAVOR[aria-pressed=true] { background: var(--favor); }
  .opt.AGAINST[aria-pressed=true] { background: var(--against); }
  .opt.NEUTRAL[aria-pressed=true] { background: var(--neutral); }
  .opt kbd {
    font: 11px monospace; opacity: .6; margin-left: 5px;
    border: 1px solid currentColor; border-radius: 3px; padding: 0 3px;
  }
  .pred { font-size: 12px; color: var(--muted); margin-top: 8px; }
  .pred .hidden-val { filter: blur(5px); cursor: pointer; }
  .note { width: 100%; margin-top: 10px; min-height: 34px; font: inherit;
          background: var(--bg); color: inherit; border: 1px solid var(--line);
          border-radius: 6px; padding: 6px 8px; resize: vertical; }
  footer { text-align: center; color: var(--muted); font-size: 12px; padding: 24px 16px 40px; }
  a { color: var(--accent); }
  .done { color: var(--yes); font-weight: 600; }
  .status { font-size: 12px; color: var(--muted); white-space: nowrap; }
  .status.good { color: var(--yes); }
  .status.warn { color: var(--no); }
</style>
</head>
<body>
<header>
  <h1>Stance coding</h1>
  <button id="prev">&larr; Prev</button>
  <span id="counter" class="meta"></span>
  <button id="next">Next &rarr;</button>
  <div class="bar" title="coded pairs"><div id="progress"></div></div>
  <span id="progressText" class="meta"></span>
  <span class="grow"></span>
  <label class="meta">coder <input type="text" id="coder" size="10" placeholder="initials"></label>
  <button id="jumpNext" title="Jump to the first post with uncoded targets">Next uncoded</button>
  <button id="reveal" aria-pressed="false" title="Show the model's stance prediction">Show model</button>
  <button id="autosave" title="Pick a CSV file once; every code is written straight to it">Autosave to CSV&hellip;</button>
  <span id="saveStatus" class="status"></span>
  <button id="export" title="Download a copy of the CSV now">Export CSV</button>
  <button id="exportJson">JSON</button>
  <button id="import">Import</button>
  <input type="file" id="importFile" accept=".json" hidden>
</header>

<main>
  <div class="cols">
    <div>
      <div class="card embed" id="embedCard"></div>
      <div class="card" id="postCard" style="margin-top:16px"></div>
    </div>
    <div id="targets"></div>
  </div>
</main>

<footer>
  Keys: <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> post &middot; <kbd>&uarr;</kbd>/<kbd>&darr;</kbd> target &middot;
  <kbd>1</kbd> relevant, <kbd>2</kbd> not relevant &middot; <kbd>f</kbd>/<kbd>a</kbd>/<kbd>n</kbd> stance &middot;
  <kbd>0</kbd> clear.
  <br>Click <b>Autosave to CSV</b> once and every code is written straight to that file
  &mdash; otherwise codes live in this browser only, so export before closing the tab.
</footer>

<script>
const DATA = __DATA__;
const POSTS = DATA.posts;
const STANCES = DATA.stance_options;
const STORE_KEY = 'stance-coding-v1';

const PAIRS = POSTS.flatMap((p, pi) => p.targets.map(t => ({...t, post_index: pi})));
let codes = load();
let idx = 0;
let activeTarget = 0;
let reveal = false;

function load() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; }
  catch (e) { return {}; }
}
function save() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(codes)); }
  catch (e) { console.warn('could not save to localStorage', e); }
  scheduleAutosave();
}
function codeFor(pairId) {
  return codes[pairId] || (codes[pairId] = {});
}
function isDone(t) {
  const c = codes[t.pair_id];
  return !!c && (c.relevant === true || c.relevant === false) && !!c.stance;
}
function esc(s) {
  return (s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

/* ---------- rendering ---------- */

function renderEmbed(post) {
  const card = document.getElementById('embedCard');
  if (!post.embed_url) {
    card.innerHTML = `<div style="padding:14px" class="meta">No embed available for this platform.
      <a href="${esc(post.post_url)}" target="_blank" rel="noopener">Open the post &rarr;</a></div>`;
    return;
  }
  card.innerHTML = `<iframe id="embedFrame" src="${esc(post.embed_url)}" height="${post.embed_height}"
    scrolling="no" allowfullscreen allow="encrypted-media; fullscreen; picture-in-picture"
    referrerpolicy="strict-origin-when-cross-origin" loading="eager"></iframe>`;
}

function renderPost(post) {
  const chips = [post.platform, post.year, post.main_type, post.sub_type, post.party,
                 post.province, post.electoral_district]
    .filter(Boolean).map(v => `<span class="chip">${esc(v)}</span>`).join('');
  document.getElementById('postCard').innerHTML = `
    <div class="meta"><b>${esc(post.seed_name)}</b> ${post.handle ? '@' + esc(post.handle) : ''}
      &middot; ${esc((post.createtime || '').slice(0, 10))}
      &middot; <a href="${esc(post.post_url)}" target="_blank" rel="noopener">open original &rarr;</a></div>
    <div class="chips">${chips}</div>
    ${post.parent_text ? `<div class="ptext parent">${esc(post.parent_text)}</div>` : ''}
    ${post.post_text ? `<div class="ptext">${esc(post.post_text)}</div>`
                     : '<div class="ptext meta">(no text captured)</div>'}`;
}

function renderTargets(post) {
  const wrap = document.getElementById('targets');
  wrap.innerHTML = post.targets.map((t, i) => {
    const c = codes[t.pair_id] || {};
    const stanceOpts = STANCES.map(s =>
      `<button class="opt ${s}" data-act="stance" data-i="${i}" data-val="${s}"
         aria-pressed="${c.stance === s}">${s.toLowerCase()}
         <kbd>${s[0].toLowerCase()}</kbd></button>`).join('');
    return `
      <div class="target ${i === activeTarget ? 'active' : ''}" data-i="${i}">
        <div class="meta">stance target ${i + 1} of ${post.targets.length}${isDone(t) ? ' &middot; <span class="done">coded</span>' : ''}</div>
        <h3>${esc(t.target)}</h3>
        <div class="q">
          <span>Is this target relevant to the post?</span>
          <div class="opts">
            <button class="opt yes" data-act="rel" data-i="${i}" data-val="1"
              aria-pressed="${c.relevant === true}">relevant <kbd>1</kbd></button>
            <button class="opt no" data-act="rel" data-i="${i}" data-val="0"
              aria-pressed="${c.relevant === false}">not relevant <kbd>2</kbd></button>
          </div>
        </div>
        <div class="q">
          <span>What is the post's stance on it?</span>
          <div class="opts">${stanceOpts}</div>
        </div>
        <div class="pred">model said:
          <span class="${reveal ? '' : 'hidden-val'}" data-act="peek">${esc(t.model_stance)}</span></div>
        <textarea class="note" data-act="note" data-i="${i}"
          placeholder="note (optional)">${esc(c.note || '')}</textarea>
      </div>`;
  }).join('');
}

function renderHeader() {
  const post = POSTS[idx];
  document.getElementById('counter').textContent = `post ${idx + 1} / ${POSTS.length}`;
  const done = PAIRS.filter(isDone).length;
  document.getElementById('progress').style.width = (100 * done / PAIRS.length) + '%';
  document.getElementById('progressText').textContent = `${done} / ${PAIRS.length} pairs`;
  document.getElementById('prev').disabled = idx === 0;
  document.getElementById('next').disabled = idx === POSTS.length - 1;
  document.title = `${done}/${PAIRS.length} · ${post.platform} · __TITLE__`;
}

function render({embed = true} = {}) {
  const post = POSTS[idx];
  if (embed) renderEmbed(post);
  renderPost(post);
  renderTargets(post);
  renderHeader();
  location.hash = 'p' + (idx + 1);
}

/* ---------- coding actions ---------- */

// relevance and stance are coded independently -- a target judged irrelevant still
// gets a stance, so the two answers can be compared against the model's
function setRelevant(i, val) {
  const t = POSTS[idx].targets[i];
  const c = codeFor(t.pair_id);
  c.relevant = c.relevant === val ? null : val;
  stamp(c, t);
  save();
  renderTargets(POSTS[idx]);
  renderHeader();
  if (isDone(t)) advance(i);
}

function setStance(i, val) {
  const t = POSTS[idx].targets[i];
  const c = codeFor(t.pair_id);
  c.stance = c.stance === val ? null : val;
  stamp(c, t);
  save();
  renderTargets(POSTS[idx]);
  renderHeader();
  if (isDone(t)) advance(i);
}

function clearCode(i) {
  const t = POSTS[idx].targets[i];
  delete codes[t.pair_id];
  save();
  renderTargets(POSTS[idx]);
  renderHeader();
}

function stamp(c, t) {
  c.post_id = POSTS[idx].post_id;
  c.platform = POSTS[idx].platform;
  c.target = t.target;
  c.model_stance = t.model_stance;
  c.coder = document.getElementById('coder').value.trim();
  c.coded_at = new Date().toISOString();
}

function advance(i) {
  const post = POSTS[idx];
  const nextUncoded = post.targets.findIndex((t, j) => j > i && !isDone(t));
  if (nextUncoded !== -1) {
    activeTarget = nextUncoded;
    renderTargets(post);
  } else if (post.targets.every(isDone) && idx < POSTS.length - 1) {
    go(idx + 1);
  }
}

function go(i) {
  idx = Math.max(0, Math.min(POSTS.length - 1, i));
  activeTarget = 0;
  render();
  window.scrollTo({top: 0, behavior: 'smooth'});
}

/* ---------- export / import ---------- */

const EXPORT_COLS = ['platform', 'post_id', 'post_url', 'seed_name', 'handle', 'main_type',
                     'sub_type', 'party', 'actor_group', 'year', 'createtime', 'target',
                     'model_stance', 'coded_relevant', 'coded_stance', 'note', 'coder', 'coded_at',
                     'n_post_targets', 'n_sampled_targets', 'pair_weight'];

function exportRows() {
  const rows = [];
  POSTS.forEach(p => p.targets.forEach(t => {
    const c = codes[t.pair_id] || {};
    rows.push({
      platform: p.platform, post_id: p.post_id, post_url: p.post_url,
      seed_name: p.seed_name, handle: p.handle, main_type: p.main_type,
      sub_type: p.sub_type, party: p.party, actor_group: p.actor_group,
      year: p.year, createtime: p.createtime, target: t.target,
      model_stance: t.model_stance,
      coded_relevant: c.relevant === true ? 1 : (c.relevant === false ? 0 : ''),
      coded_stance: c.stance || '', note: c.note || '',
      coder: c.coder || '', coded_at: c.coded_at || '',
      n_post_targets: p.n_post_targets, n_sampled_targets: p.n_sampled_targets,
      pair_weight: p.pair_weight,
    });
  }));
  return rows;
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], {type}));
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

function csvCell(v) {
  v = v === null || v === undefined ? '' : String(v);
  return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
}

function buildCsv() {
  const rows = exportRows();
  return [EXPORT_COLS.join(',')]
    .concat(rows.map(r => EXPORT_COLS.map(c => csvCell(r[c])).join(','))).join('\n') + '\n';
}

function suffix() {
  const who = document.getElementById('coder').value.trim().replace(/\W+/g, '') || 'coder';
  return `${who}_${new Date().toISOString().slice(0, 10)}`;
}

/* ---------- autosave straight to a CSV on disk ----------
   The File System Access API lets the page hold a writable handle to one file the
   coder picks. Every change is then written to that file, so nothing depends on
   remembering to export. Needs a secure context (localhost counts) and a
   Chromium-based browser; elsewhere we fall back to a leave-the-page warning. */

let fileHandle = null;      // granted handle we can write to
let pendingHandle = null;   // remembered handle still needing a permission click
let autosaveTimer = null;
let writing = false;
let writeAgain = false;
let unsaved = false;

const IDB_NAME = 'stance-coding';
const IDB_STORE = 'handles';

function idb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(IDB_NAME, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(IDB_STORE);
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

async function idbSet(k, v) {
  const db = await idb();
  return new Promise((res, rej) => {
    const tx = db.transaction(IDB_STORE, 'readwrite');
    tx.objectStore(IDB_STORE).put(v, k);
    tx.oncomplete = () => res();
    tx.onerror = () => rej(tx.error);
  });
}

async function idbGet(k) {
  const db = await idb();
  return new Promise((res, rej) => {
    const tx = db.transaction(IDB_STORE, 'readonly');
    const q = tx.objectStore(IDB_STORE).get(k);
    q.onsuccess = () => res(q.result);
    q.onerror = () => rej(q.error);
  });
}

function setStatus(text, cls = '') {
  const el = document.getElementById('saveStatus');
  el.textContent = text;
  el.className = 'status ' + cls;
}

function setAutosaveLabel(text) {
  document.getElementById('autosave').innerHTML = text;
}

async function writeCsv() {
  if (!fileHandle) return;
  if (writing) { writeAgain = true; return; }   // coalesce a burst of keystrokes
  writing = true;
  try {
    const w = await fileHandle.createWritable();
    await w.write(buildCsv());
    await w.close();
    unsaved = false;
    const done = PAIRS.filter(isDone).length;
    setStatus(`saved ${done}/${PAIRS.length} to ${fileHandle.name}`, 'good');
  } catch (e) {
    unsaved = true;   // keep the leave-the-page warning armed
    setStatus('autosave failed: ' + e.message + ' -- use Export CSV', 'warn');
    console.error('autosave write failed', e);
  } finally {
    writing = false;
    if (writeAgain) { writeAgain = false; writeCsv(); }
  }
}

function scheduleAutosave() {
  unsaved = true;
  if (!fileHandle) {
    if (pendingHandle) setStatus('autosave paused -- click "Reconnect autosave"', 'warn');
    else if (window.showSaveFilePicker) setStatus('not autosaving -- pick a CSV file', 'warn');
    return;
  }
  setStatus('saving...');
  clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(writeCsv, 500);
}

async function connectAutosave() {
  await autosaveReady.catch(() => {});   // don't race the reload-time restore
  // a remembered file only needs its permission re-granted, not re-picking
  if (pendingHandle) {
    try {
      if (await pendingHandle.requestPermission({mode: 'readwrite'}) === 'granted') {
        fileHandle = pendingHandle;
        pendingHandle = null;
        setAutosaveLabel('Autosaving &#10003;');
        await writeCsv();
        return;
      }
    } catch (e) { console.warn('could not reconnect', e); }
    pendingHandle = null;
  }
  if (!window.showSaveFilePicker) {
    setStatus('this browser cannot autosave to a file -- use Export CSV', 'warn');
    return;
  }
  try {
    fileHandle = await window.showSaveFilePicker({
      suggestedName: `stance_coding_${suffix()}.csv`,
      types: [{description: 'CSV', accept: {'text/csv': ['.csv']}}],
    });
  } catch (e) {
    if (e.name !== 'AbortError') setStatus('could not open that file: ' + e.message, 'warn');
    return;
  }
  await idbSet('handle', fileHandle).catch(e => console.warn('could not remember the file', e));
  setAutosaveLabel('Autosaving &#10003;');
  await writeCsv();
}

async function restoreAutosave() {
  if (!window.showSaveFilePicker) {
    setStatus('no autosave in this browser -- export before you close the tab', 'warn');
    document.getElementById('autosave').disabled = true;
    return;
  }
  let h = null;
  try { h = await idbGet('handle'); } catch (e) { /* first run, or no idb */ }
  if (fileHandle) return;   // a click beat us to it; leave that connection alone
  if (!h) { setStatus('autosave off -- pick a CSV file', 'warn'); return; }
  if (await h.queryPermission({mode: 'readwrite'}) === 'granted') {
    fileHandle = h;
    setAutosaveLabel('Autosaving &#10003;');
    setStatus(`autosaving to ${h.name}`, 'good');
  } else {
    pendingHandle = h;
    setAutosaveLabel('Reconnect autosave');
    setStatus(`click to keep autosaving to ${h.name}`, 'warn');
  }
}

// last line of defence when autosave is not on
window.addEventListener('beforeunload', e => {
  if (unsaved && Object.keys(codes).length) { e.preventDefault(); e.returnValue = ''; }
});

/* ---------- wiring ---------- */

document.getElementById('prev').onclick = () => go(idx - 1);
document.getElementById('next').onclick = () => go(idx + 1);
document.getElementById('jumpNext').onclick = () => {
  const p = PAIRS.find(x => !isDone(x));
  if (!p) { alert('Everything is coded.'); return; }
  go(p.post_index);
};
document.getElementById('reveal').onclick = e => {
  reveal = !reveal;
  e.currentTarget.setAttribute('aria-pressed', reveal);
  e.currentTarget.textContent = reveal ? 'Hide model' : 'Show model';
  renderTargets(POSTS[idx]);
};
document.getElementById('autosave').onclick = () => connectAutosave();
document.getElementById('export').onclick = () =>
  download(`stance_coding_${suffix()}.csv`, buildCsv(), 'text/csv');
document.getElementById('exportJson').onclick = () =>
  download(`stance_coding_${suffix()}.json`, JSON.stringify(codes, null, 2), 'application/json');
document.getElementById('import').onclick = () => document.getElementById('importFile').click();
document.getElementById('importFile').onchange = async e => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const incoming = JSON.parse(await file.text());
    codes = Object.assign(codes, incoming);
    save();
    render({embed: false});
  } catch (err) { alert('Could not read that JSON: ' + err.message); }
  e.target.value = '';
};

const coderInput = document.getElementById('coder');
coderInput.value = localStorage.getItem(STORE_KEY + ':coder') || '';
coderInput.oninput = () => localStorage.setItem(STORE_KEY + ':coder', coderInput.value);

document.getElementById('targets').addEventListener('click', e => {
  const btn = e.target.closest('[data-act]');
  if (!btn) return;
  const act = btn.dataset.act;
  if (act === 'peek') { btn.classList.toggle('hidden-val'); return; }
  const i = +btn.dataset.i;
  activeTarget = i;
  if (act === 'rel') setRelevant(i, btn.dataset.val === '1');
  else if (act === 'stance') setStance(i, btn.dataset.val);
});
document.getElementById('targets').addEventListener('input', e => {
  const box = e.target.closest('[data-act=note]');
  if (!box) return;
  const t = POSTS[idx].targets[+box.dataset.i];
  codeFor(t.pair_id).note = box.value;
  save();
});

document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea') return;
  const n = POSTS[idx].targets.length;
  const k = e.key.toLowerCase();
  if (k === 'arrowright') go(idx + 1);
  else if (k === 'arrowleft') go(idx - 1);
  else if (k === 'arrowdown' || k === 'tab') { activeTarget = (activeTarget + 1) % n; renderTargets(POSTS[idx]); }
  else if (k === 'arrowup') { activeTarget = (activeTarget - 1 + n) % n; renderTargets(POSTS[idx]); }
  else if (k === '1') setRelevant(activeTarget, true);
  else if (k === '2') setRelevant(activeTarget, false);
  else if (k === '0') clearCode(activeTarget);
  else if (k === 'f') setStance(activeTarget, 'FAVOR');
  else if (k === 'a') setStance(activeTarget, 'AGAINST');
  else if (k === 'n') setStance(activeTarget, 'NEUTRAL');
  else return;
  e.preventDefault();
});

// embeds report their rendered height; trust it only for the frame we asked for
window.addEventListener('message', e => {
  const frame = document.getElementById('embedFrame');
  if (!frame || e.source !== frame.contentWindow) return;
  let h = null;
  const d = e.data;
  if (typeof d === 'number') h = d;
  else if (d && typeof d === 'object') {
    h = d.height ?? d.offsetHeight ?? d['twttr.embed']?.params?.[0]?.height ?? null;
  } else if (typeof d === 'string') {
    try { h = (JSON.parse(d) || {}).height ?? null; } catch (err) { /* not ours */ }
  }
  if (h && h > 100 && h < 3000) frame.height = Math.ceil(h);
});

const fromHash = parseInt((location.hash.match(/^#p(\d+)$/) || [])[1], 10);
if (fromHash) idx = Math.min(POSTS.length, Math.max(1, fromHash)) - 1;
render();
const autosaveReady = restoreAutosave();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--input', default='./out/classified_post_sample.csv',
                        help='CSV written by sample_classified_posts.py (long format)')
    parser.add_argument('--output', default='./out/coding_page.html',
                        help='where to write the self-contained HTML page')
    parser.add_argument('--title', default='Stance target coding',
                        help='page title')
    args = parser.parse_args()

    posts = load_posts(args.input)
    n_targets = sum(len(p['targets']) for p in posts)
    missing = [p for p in posts if not p['embed_url']]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(render_page(posts, args.title))

    print(f'wrote {args.output}')
    print(f'  {len(posts)} posts, {n_targets} (post, target) pairs to code')
    by_platform = {}
    for p in posts:
        by_platform[p['platform']] = by_platform.get(p['platform'], 0) + 1
    for platform, count in sorted(by_platform.items()):
        print(f'  {platform:<10} {count}')
    if missing:
        print(f'  WARNING: {len(missing)} posts have no embeddable URL '
              f'(link-out only): {[p["post_url"] for p in missing][:3]}')


if __name__ == '__main__':
    main()
