"""Score the model's stance targets and stance labels against manual coding.

Reads one or more CSVs exported by the coding page (make_coding_page.py) and reports:

  * coverage -- how much of the sample was actually coded
  * target relevance -- the share of extracted targets a coder judged relevant to
    the post, which is the precision of target extraction, broken down by category
  * stance -- accuracy, per-class precision/recall/F1 (the coder is the gold
    standard), macro/micro/weighted averages, the confusion matrix, and
    chance-corrected agreement
  * the same numbers per platform / year / actor type / party / model label
  * every pair where the model and the coder disagree, for error analysis

Agreement statistics: with one coder the model-vs-coder pair is scored with Cohen's
kappa (each rater gets its own marginals) and with Fleiss's kappa, which for two
raters is Scott's pi (raters share pooled marginals). Fleiss's kappa is the headline
number once several coders' files are passed, in which case coder-vs-coder agreement
is reported too and the model is scored against the coders' majority label.

Confidence intervals are percentile bootstrap over pairs (seeded, so reruns match).

Examples:
    python scripts/target_mining/score_coded_posts.py -i out/stance_coding_coder_2026-07-29.csv
    python scripts/target_mining/score_coded_posts.py -i out/coder_bs.csv -i out/coder_jd.csv \
        --output out/coding_report.md
    python scripts/target_mining/score_coded_posts.py --selftest
"""

import argparse
import csv
import math
import os
import random
from collections import Counter, OrderedDict, defaultdict

STANCES = ['FAVOR', 'AGAINST', 'NEUTRAL']

# (column, label) pairs the report breaks results down by
BREAKDOWNS = [
    ('platform', 'platform'),
    ('year', 'year'),
    ('main_type', 'actor type'),
    ('actor_group', 'actor group'),
    ('party', 'party'),
    ('model_stance', 'model label'),
]

# conditioning on the model's own label makes its predictions constant within a
# group, which forces kappa to 0 and makes macro-F1 meaningless -- the per-class
# precision table already carries that information
STANCE_BREAKDOWNS = [b for b in BREAKDOWNS if b[0] != 'model_stance']

# ---------------------------------------------------------------- statistics


def weighted_stats(triples, labels):
    """Accuracy, per-class P/R/F1 and Cohen's kappa from (gold, pred, weight) triples.

    Weights undo the two-stage design: capping targets per post under-samples pairs
    from posts with many classified targets, so each pair stands for
    n_post_targets / n_sampled_targets of them.
    """
    total = sum(w for _, _, w in triples)
    if not total:
        return None
    accuracy = sum(w for g, p, w in triples if g == p) / total
    per = OrderedDict()
    for c in labels:
        tp = sum(w for g, p, w in triples if g == c and p == c)
        fp = sum(w for g, p, w in triples if g != c and p == c)
        fn = sum(w for g, p, w in triples if g == c and p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[c] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': tp + fn}
    present = [c for c in labels if per[c]['support'] or
               any(p == c for _, p, _ in triples)]
    macro_f1 = sum(per[c]['f1'] for c in present) / len(present) if present else 0.0
    gold_share = {c: sum(w for g, _, w in triples if g == c) / total for c in labels}
    pred_share = {c: sum(w for _, p, w in triples if p == c) / total for c in labels}
    expected = sum(gold_share[c] * pred_share[c] for c in labels)
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else float('nan')
    # Kish effective sample size: how much precision the weights cost
    effective = total ** 2 / sum(w * w for _, _, w in triples)
    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'kappa': kappa,
            'per_class': per, 'n': len(triples), 'effective_n': effective}


def weights_of(rows):
    """pair_weight column if the sample recorded it, else 1.0 (unweighted)."""
    weights = []
    for r in rows:
        raw = (r.get('pair_weight') or '').strip()
        try:
            w = float(raw)
        except ValueError:
            w = 1.0
        weights.append(w if w > 0 else 1.0)
    return weights


def confusion(gold, pred, labels):
    """counts[(gold, pred)] for aligned label sequences."""
    counts = Counter(zip(gold, pred))
    return {(g, p): counts.get((g, p), 0) for g in labels for p in labels}


def prf(gold, pred, labels):
    """Per-class precision/recall/F1/support plus the three averages."""
    per = OrderedDict()
    for c in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[c] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': tp + fn}

    n = len(gold)
    present = [c for c in labels if per[c]['support'] or
               any(p == c for p in pred)]        # ignore classes nobody used
    macro = {k: (sum(per[c][k] for c in present) / len(present) if present else 0.0)
             for k in ('precision', 'recall', 'f1')}
    total_support = sum(per[c]['support'] for c in labels) or 1
    weighted = {k: sum(per[c][k] * per[c]['support'] for c in labels) / total_support
                for k in ('precision', 'recall', 'f1')}
    accuracy = sum(1 for g, p in zip(gold, pred) if g == p) / n if n else 0.0
    # single-label multiclass: micro-P = micro-R = micro-F1 = accuracy
    return {'per_class': per, 'macro': macro, 'weighted': weighted,
            'accuracy': accuracy, 'micro_f1': accuracy, 'n': n}


def cohen_kappa(a, b, labels):
    """Chance-corrected agreement between two raters with separate marginals."""
    n = len(a)
    if not n:
        return float('nan')
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[c] / n) * (cb[c] / n) for c in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else float('nan')


def fleiss_kappa(table):
    """Fleiss's kappa from a table of per-item category counts.

    `table` is a list of rows, one per item, each row counting how many raters
    assigned each category. Rows must sum to the same number of raters. With two
    raters this is Scott's pi.
    """
    rows = [r for r in table if sum(r) > 0]
    if not rows:
        return float('nan')
    m = sum(rows[0])
    if m < 2 or any(sum(r) != m for r in rows):
        return float('nan')     # every item needs the same rater count
    n_items = len(rows)
    k = len(rows[0])

    agreement = sum((sum(c * c for c in row) - m) / (m * (m - 1)) for row in rows)
    p_bar = agreement / n_items
    p_cat = [sum(row[j] for row in rows) / (n_items * m) for j in range(k)]
    p_e = sum(p * p for p in p_cat)
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else float('nan')


def fleiss_from_pairs(rater_labels, labels):
    """Fleiss's kappa for aligned label lists (one list per rater)."""
    index = {c: i for i, c in enumerate(labels)}
    table = []
    for votes in zip(*rater_labels):
        row = [0] * len(labels)
        for v in votes:
            row[index[v]] += 1
        table.append(row)
    return fleiss_kappa(table)


def bootstrap_ci(items, stat, seed=42, reps=2000, alpha=0.05):
    """Percentile CI for a statistic computed over a list of items."""
    n = len(items)
    if n < 2:
        return (float('nan'), float('nan'))
    rng = random.Random(seed)
    values = []
    for _ in range(reps):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        v = stat(sample)
        if not (isinstance(v, float) and math.isnan(v)):
            values.append(v)
    if not values:
        return (float('nan'), float('nan'))
    values.sort()
    lo = values[max(0, int(math.floor(alpha / 2 * len(values))))]
    hi = values[min(len(values) - 1, int(math.ceil((1 - alpha / 2) * len(values))) - 1)]
    return (lo, hi)


# ---------------------------------------------------------------- loading

def coder_name(path, rows):
    """Prefer the initials typed into the page; fall back to the file name."""
    names = {r['coder'].strip() for r in rows if r.get('coder', '').strip()}
    if len(names) == 1:
        return names.pop()
    return os.path.splitext(os.path.basename(path))[0]


def load_coding(paths):
    """Read each coded CSV into {pair_key: row}, keyed identically across coders."""
    coders = OrderedDict()
    meta = {}
    for path in paths:
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        missing = {'platform', 'post_id', 'target', 'model_stance',
                   'coded_relevant', 'coded_stance'} - set(rows[0] if rows else {})
        if missing:
            raise SystemExit(f'{path} is missing columns: {sorted(missing)}')
        name = coder_name(path, rows)
        if name in coders:
            name = f'{name} ({os.path.basename(path)})'
        table = OrderedDict()
        for r in rows:
            key = (r['platform'], r['post_id'], r['target'])
            table[key] = r
            meta.setdefault(key, r)
        coders[name] = table
    return coders, meta


def relevance_of(row):
    v = (row.get('coded_relevant') or '').strip()
    return {'1': True, '0': False}.get(v)


def stance_of(row):
    v = (row.get('coded_stance') or '').strip().upper()
    return v if v in STANCES else None


# ---------------------------------------------------------------- reporting

class Report:
    """Collects lines for stdout and, optionally, a markdown file."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=''):
        self.lines.append(text)
        print(text)

    def head(self, text, char='='):
        self('')
        self(text)
        self(char * len(text))

    def table(self, headers, rows):
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        self('  '.join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip())
        self('  '.join('-' * widths[i] for i in range(len(headers))))
        for row in rows:
            self('  '.join(str(c).ljust(widths[i]) for i, c in enumerate(row)).rstrip())


def pct(x):
    return 'n/a' if isinstance(x, float) and math.isnan(x) else f'{100 * x:.1f}%'


def num(x):
    return 'n/a' if isinstance(x, float) and math.isnan(x) else f'{x:.3f}'


def ci(bounds):
    lo, hi = bounds
    if any(isinstance(v, float) and math.isnan(v) for v in (lo, hi)):
        return ''
    return f'[{lo:.3f}, {hi:.3f}]'


def report_coverage(out, coders, meta):
    out.head('Coverage')
    rows = []
    for name, table in coders.items():
        both = sum(1 for r in table.values()
                   if relevance_of(r) is not None and stance_of(r) is not None)
        rel_only = sum(1 for r in table.values()
                       if relevance_of(r) is not None and stance_of(r) is None)
        st_only = sum(1 for r in table.values()
                      if relevance_of(r) is None and stance_of(r) is not None)
        blank = len(table) - both - rel_only - st_only
        rows.append([name, len(table), both, rel_only, st_only, blank])
    out.table(['coder', 'pairs', 'fully coded', 'relevance only', 'stance only', 'blank'], rows)
    posts = {(k[0], k[1]) for k in meta}
    out('')
    out(f'{len(meta)} distinct (post, target) pairs over {len(posts)} posts.')


def report_relevance(out, name, table, seed):
    """Share of extracted targets the coder judged relevant to the post."""
    coded = [(k, r) for k, r in table.items() if relevance_of(r) is not None]
    if not coded:
        out('No relevance codes.')
        return
    flags = [relevance_of(r) for _, r in coded]
    share = sum(flags) / len(flags)
    lo, hi = bootstrap_ci(flags, lambda s: sum(s) / len(s), seed=seed)
    out.head(f'Target relevance -- {name}', '-')
    out(f'{sum(flags)} of {len(flags)} extracted targets judged relevant to their post '
        f'= {pct(share)} 95% CI [{pct(lo)}, {pct(hi)}]')
    out('(this is the precision of target extraction as the coder sees it)')

    ws = weights_of([r for _, r in coded])
    if any(abs(w - 1.0) > 1e-9 for w in ws):
        wshare = sum(w for f, w in zip(flags, ws) if f) / sum(ws)
        eff = sum(ws) ** 2 / sum(w * w for w in ws)
        items = list(zip(flags, ws))
        wlo, whi = bootstrap_ci(
            items, lambda s: sum(w for f, w in s if f) / sum(w for _, w in s), seed=seed)
        out(f'weighted to all extracted pairs: {pct(wshare)} '
            f'95% CI [{pct(wlo)}, {pct(whi)}] (effective n {eff:.0f} of {len(ws)})')
        out('  the unweighted figure treats every sampled pair equally, which favours '
            'posts with few targets; the weighted one estimates the share over every '
            'classified pair in the corpus')
    else:
        out('unweighted: this sample carries no pair_weight column, so it estimates the '
            'per-post average rather than the share over all extracted pairs')

    for col, label in BREAKDOWNS:
        groups = defaultdict(list)
        for _, r in coded:
            groups[r.get(col, '') or '(blank)'].append(relevance_of(r))
        if len(groups) < 2:
            continue
        rows = [[g, len(v), sum(v), pct(sum(v) / len(v))]
                for g, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))]
        out('')
        out(f'by {label}:')
        out.table(['group', 'n', 'relevant', 'share'], rows)

    # how the coder's two answers interact: if irrelevant targets nearly all get one
    # stance, the "all pairs" stance scores are partly measuring relevance
    both = [r for _, r in coded if stance_of(r) is not None]
    if both:
        out('')
        out('coder relevance x coder stance:')
        out.table(['relevance'] + STANCES + ['total'],
                  [[label] + [sum(1 for r in both if relevance_of(r) is flag
                                  and stance_of(r) == s) for s in STANCES]
                   + [sum(1 for r in both if relevance_of(r) is flag)]
                   for flag, label in [(True, 'relevant'), (False, 'not relevant')]])


def report_stance(out, name, table, seed, relevant_only):
    pairs = []
    for k, r in table.items():
        gold, pred = stance_of(r), (r['model_stance'] or '').strip().upper()
        if gold is None or pred not in STANCES:
            continue
        if relevant_only and relevance_of(r) is not True:
            continue
        pairs.append((k, r, gold, pred))
    scope = 'targets coded relevant' if relevant_only else 'all coded pairs'
    out.head(f'Stance vs {name} -- {scope} (n={len(pairs)})', '-')
    if len(pairs) < 2:
        out('Not enough coded pairs to score.')
        return None

    gold = [g for _, _, g, _ in pairs]
    pred = [p for _, _, _, p in pairs]
    scores = prf(gold, pred, STANCES)

    out(f'accuracy / micro-F1  {num(scores["accuracy"])}  '
        f'95% CI {ci(bootstrap_ci(list(zip(gold, pred)), lambda s: prf([g for g, _ in s], [p for _, p in s], STANCES)["accuracy"], seed=seed))}')
    out(f'macro-F1             {num(scores["macro"]["f1"])}  '
        f'95% CI {ci(bootstrap_ci(list(zip(gold, pred)), lambda s: prf([g for g, _ in s], [p for _, p in s], STANCES)["macro"]["f1"], seed=seed))}')
    out(f'weighted F1          {num(scores["weighted"]["f1"])}')
    ck = cohen_kappa(gold, pred, STANCES)
    fk = fleiss_from_pairs([gold, pred], STANCES)
    out(f"Cohen's kappa        {num(ck)}  "
        f'95% CI {ci(bootstrap_ci(list(zip(gold, pred)), lambda s: cohen_kappa([g for g, _ in s], [p for _, p in s], STANCES), seed=seed))}')
    out(f"Fleiss's kappa       {num(fk)}  (two raters, so this is Scott's pi)")

    ws = weights_of([r for _, r, _, _ in pairs])
    if any(abs(w - 1.0) > 1e-9 for w in ws):
        triples = [(g, p, w) for (_, _, g, p), w in zip(pairs, ws)]
        w_stat = weighted_stats(triples, STANCES)
        w_ci = bootstrap_ci(triples,
                            lambda s: weighted_stats(s, STANCES)['kappa'], seed=seed)
        out('')
        out('weighted to the pair-level population:')
        out(f'  accuracy {num(w_stat["accuracy"])}   macro-F1 {num(w_stat["macro_f1"])}   '
            f"Cohen's kappa {num(w_stat['kappa'])} 95% CI {ci(w_ci)}")
        out(f'  effective n {w_stat["effective_n"]:.0f} of {w_stat["n"]} coded pairs')

    out('')
    out('per class (coder = gold standard):')
    out.table(['class', 'precision', 'recall', 'f1', 'support (coder)', 'predicted (model)'],
              [[c, num(v['precision']), num(v['recall']), num(v['f1']), v['support'],
                sum(1 for p in pred if p == c)]
               for c, v in scores['per_class'].items()]
              + [['macro', num(scores['macro']['precision']), num(scores['macro']['recall']),
                  num(scores['macro']['f1']), len(gold), len(pred)]])

    out('')
    out('confusion matrix (rows = coder, columns = model):')
    cm = confusion(gold, pred, STANCES)
    out.table(['coder \\ model'] + STANCES + ['total'],
              [[g] + [cm[(g, p)] for p in STANCES] + [sum(cm[(g, p)] for p in STANCES)]
               for g in STANCES]
              + [['total'] + [sum(cm[(g, p)] for g in STANCES) for p in STANCES] + [len(gold)]])

    for col, label in STANCE_BREAKDOWNS:
        groups = defaultdict(list)
        for _, r, g, p in pairs:
            groups[r.get(col, '') or '(blank)'].append((g, p))
        if len(groups) < 2:
            continue
        rows = []
        for g, vals in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            gs = [a for a, _ in vals]
            ps = [b for _, b in vals]
            s = prf(gs, ps, STANCES)
            rows.append([g, len(vals), num(s['accuracy']), num(s['macro']['f1']),
                         num(cohen_kappa(gs, ps, STANCES))])
        out('')
        out(f'by {label}:')
        out.table(['group', 'n', 'accuracy', 'macro-F1', "Cohen's kappa"], rows)
    return pairs


def report_disagreements(out, pairs, limit):
    rows = [(k, r, g, p) for k, r, g, p in pairs if g != p]
    out.head(f'Disagreements ({len(rows)} of {len(pairs)})', '-')
    if not rows:
        out('None.')
        return
    shown = rows if limit <= 0 else rows[:limit]
    for k, r, g, p in shown:
        rel = relevance_of(r)
        flag = '' if rel is None else ('' if rel else '  [coded not relevant]')
        out(f'- "{r["target"]}" -- coder {g}, model {p}{flag}')
        out(f'  {r["seed_name"]} ({r["platform"]}, {r["year"]}) {r["post_url"]}')
        if (r.get('note') or '').strip():
            out(f'  note: {r["note"].strip()}')
    if len(rows) > len(shown):
        out(f'... {len(rows) - len(shown)} more (use --show-disagreements 0 for all)')


def report_between_coders(out, coders, seed):
    """Coder-vs-coder agreement, then the model against the coders' majority."""
    names = list(coders)
    out.head('Between coders')

    for field, getter, labels in [('relevance', relevance_of, [True, False]),
                                  ('stance', stance_of, STANCES)]:
        shared = [k for k in coders[names[0]]
                  if all(getter(coders[n].get(k, {})) is not None for n in names)]
        if len(shared) < 2:
            out(f'{field}: no pairs coded by every coder.')
            continue
        per_rater = [[getter(coders[n][k]) for k in shared] for n in names]
        fk = fleiss_from_pairs(per_rater, labels)
        out('')
        out(f'{field}: {len(shared)} pairs coded by all {len(names)} coders, '
            f"Fleiss's kappa {num(fk)}")
        if len(names) == 2:
            out(f"  Cohen's kappa {num(cohen_kappa(per_rater[0], per_rater[1], labels))}")
        rows = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                agree = sum(1 for a, b in zip(per_rater[i], per_rater[j]) if a == b)
                rows.append([names[i], names[j], f'{agree}/{len(shared)}',
                             pct(agree / len(shared)),
                             num(cohen_kappa(per_rater[i], per_rater[j], labels))])
        out.table(['coder A', 'coder B', 'agree', 'raw', "Cohen's kappa"], rows)

    # majority label per pair, ties dropped, then score the model against it
    shared = [k for k in coders[names[0]]
              if all(stance_of(coders[n].get(k, {})) is not None for n in names)]
    gold, pred, ties = [], [], 0
    for k in shared:
        votes = Counter(stance_of(coders[n][k]) for n in names)
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            ties += 1
            continue
        model = (coders[names[0]][k]['model_stance'] or '').strip().upper()
        if model in STANCES:
            gold.append(top[0][0])
            pred.append(model)
    if len(gold) >= 2:
        s = prf(gold, pred, STANCES)
        out('')
        out(f'model vs coder majority (n={len(gold)}, {ties} ties dropped): '
            f'accuracy {num(s["accuracy"])}, macro-F1 {num(s["macro"]["f1"])}, '
            f"Cohen's kappa {num(cohen_kappa(gold, pred, STANCES))}")


# ---------------------------------------------------------------- self-test

def selftest():
    """Check the statistics against hand-computable and published values."""
    gold = ['A', 'A', 'B', 'B', 'C']
    pred = ['A', 'B', 'B', 'B', 'C']
    s = prf(gold, pred, ['A', 'B', 'C'])
    assert abs(s['accuracy'] - 0.8) < 1e-12, s['accuracy']
    assert abs(s['per_class']['A']['precision'] - 1.0) < 1e-12
    assert abs(s['per_class']['A']['recall'] - 0.5) < 1e-12
    assert abs(s['per_class']['A']['f1'] - 2 / 3) < 1e-12
    assert abs(s['per_class']['B']['precision'] - 2 / 3) < 1e-12
    assert abs(s['per_class']['B']['recall'] - 1.0) < 1e-12
    assert abs(s['per_class']['B']['f1'] - 0.8) < 1e-12
    assert abs(s['macro']['f1'] - (2 / 3 + 0.8 + 1.0) / 3) < 1e-12
    assert abs(s['weighted']['f1'] - (2 * 2 / 3 + 2 * 0.8 + 1.0) / 5) < 1e-12
    assert abs(s['micro_f1'] - s['accuracy']) < 1e-12

    # 2x2 with po=0.70, pe=0.50 -> kappa=0.40
    a = ['y'] * 20 + ['y'] * 5 + ['n'] * 10 + ['n'] * 15
    b = ['y'] * 20 + ['n'] * 5 + ['y'] * 10 + ['n'] * 15
    assert abs(cohen_kappa(a, b, ['y', 'n']) - 0.4) < 1e-12, cohen_kappa(a, b, ['y', 'n'])
    assert abs(cohen_kappa(a, a, ['y', 'n']) - 1.0) < 1e-12

    # published Fleiss example: 10 items, 14 raters, 5 categories, kappa = 0.210
    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
             [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
             [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert abs(fleiss_kappa(table) - 0.2099) < 5e-4, fleiss_kappa(table)
    assert abs(fleiss_kappa([[2, 0], [0, 2], [2, 0]]) - 1.0) < 1e-12

    # Fleiss with two raters == Scott's pi (pooled marginals), and differs from Cohen
    rng = random.Random(0)
    x = [rng.choice(STANCES) for _ in range(200)]
    y = [rng.choice(STANCES) for _ in range(200)]
    n = len(x)
    po = sum(1 for i, j in zip(x, y) if i == j) / n
    pooled = Counter(x) + Counter(y)
    pe = sum((pooled[c] / (2 * n)) ** 2 for c in STANCES)
    scott = (po - pe) / (1 - pe)
    assert abs(fleiss_from_pairs([x, y], STANCES) - scott) < 1e-12

    # weights of 1 must reproduce the unweighted numbers exactly
    triples = [(g, p, 1.0) for g, p in zip(gold, pred)]
    w = weighted_stats(triples, ['A', 'B', 'C'])
    assert abs(w['accuracy'] - s['accuracy']) < 1e-12
    assert abs(w['macro_f1'] - s['macro']['f1']) < 1e-12
    assert abs(w['kappa'] - cohen_kappa(gold, pred, ['A', 'B', 'C'])) < 1e-12
    assert abs(w['effective_n'] - len(gold)) < 1e-12

    # duplicating a row is the same as doubling its weight
    dup = [(g, p, 1.0) for g, p in zip(gold, pred)] + [(gold[0], pred[0], 1.0)]
    wt = [(gold[0], pred[0], 2.0)] + [(g, p, 1.0) for g, p in list(zip(gold, pred))[1:]]
    a, b = weighted_stats(dup, ['A', 'B', 'C']), weighted_stats(wt, ['A', 'B', 'C'])
    for key in ('accuracy', 'macro_f1', 'kappa'):
        assert abs(a[key] - b[key]) < 1e-12, (key, a[key], b[key])

    # uneven weights cost precision: effective n < number of rows
    uneven = weighted_stats([('A', 'A', 10.0), ('B', 'B', 1.0), ('C', 'A', 1.0)],
                            ['A', 'B', 'C'])
    assert uneven['effective_n'] < 3, uneven['effective_n']
    assert weights_of([{'pair_weight': '2.5'}, {'pair_weight': ''},
                       {'pair_weight': 'x'}, {}]) == [2.5, 1.0, 1.0, 1.0]

    lo, hi = bootstrap_ci([1] * 50 + [0] * 50, lambda s: sum(s) / len(s), reps=500)
    assert 0.3 < lo < 0.5 < hi < 0.7, (lo, hi)
    assert math.isnan(fleiss_kappa([[1, 0], [0, 2]]))     # unequal rater counts
    print('selftest: all statistics checks pass')


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-i', '--input', action='append', default=[],
                        help='coded CSV exported by the coding page (repeat per coder)')
    parser.add_argument('--output', help='also write the report to this file')
    parser.add_argument('--show-disagreements', type=int, default=25,
                        help='how many disagreements to list (0 = all)')
    parser.add_argument('--seed', type=int, default=42, help='bootstrap seed')
    parser.add_argument('--selftest', action='store_true',
                        help='verify the statistics against known values and exit')
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.input:
        parser.error('give at least one -i/--input coded CSV (or --selftest)')

    coders, meta = load_coding(args.input)
    out = Report()
    out(f'Coded stance evaluation -- {len(coders)} coder(s): {", ".join(coders)}')
    report_coverage(out, coders, meta)

    for name, table in coders.items():
        report_relevance(out, name, table, args.seed)

    for name, table in coders.items():
        for relevant_only in (False, True):
            pairs = report_stance(out, name, table, args.seed, relevant_only)
            if pairs and not relevant_only:
                report_disagreements(out, pairs, args.show_disagreements)

    if len(coders) > 1:
        report_between_coders(out, coders, args.seed)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write('```\n' + '\n'.join(out.lines) + '\n```\n')
        print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
