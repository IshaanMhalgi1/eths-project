"""Decide q57's disposition: hand-correct or exclude.

q57 ("Manifest Destiny Polk administration") is a broken qrels entry, not a
labelling problem. Its five references are dated 1829-1839 -- 4 to 14 years
before Manifest Destiny (1843) and the Polk administration (1845-49) even
began -- and inspection shows they are Jackson-era partisan polemic that
matched only on the token "administration". The query and its reference set
contradict each other, so no system can be said to answer it correctly.

Two dispositions are possible and they are not equivalent in methodological
consequence, so this measures both rather than picking by taste.

  EXCLUDE  -- drop q57, as q89/q98 were dropped. Keeps the reference set
              homogeneous (every entry produced by the same term-overlap
              generator) and keeps every query reproducible by script.

  HAND-CORRECT -- replace the references with documents that actually discuss
              Manifest Destiny. Preserves n, but the replacement set is chosen
              WITH knowledge of the era, and the era is the defining feature of
              the query. That injects a date-alignment confound into exactly the
              query class the temporal ranker is being tested on.

The report's central caveat is that the term-overlap ground truth creates a
date-alignment confound which the temporal ranker exploits. A hand-corrected q57
would be the single entry in the set whose references were era-aligned by
construction. This script quantifies what that would do to the headline
contrasts, so the choice is made on evidence.

Writes evaluation/q57_disposition.json. Read-only with respect to the qrels and
the index; the caller applies whichever change is chosen.
"""
import json
import os
import sys

import numpy as np
import yaml
from opensearchpy import OpenSearch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.ablation_rankers import get_ranker
from ranking.shared_hybrid import SHARED_FETCH_SIZE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'evaluation', 'q57_disposition.json')

# Era bounds for the concepts the query names. Used only to PROPOSE a corrected
# set for the counterfactual; the real relevance judgment still needs a human.
MD_START, MD_END = 1843, 1853
POLK_START, POLK_END = 1845, 1849


def evaluate(candidates, true_docs, k=10):
    seen, dedup = set(), []
    for r in candidates:
        p = r['parent_doc_id']
        if p not in seen:
            seen.add(p)
            dedup.append(p)
        if len(dedup) == k:
            break
    hits = [1 if d in true_docs else 0 for d in dedup]
    mrr = 0.0
    for i, d in enumerate(dedup):
        if d in true_docs:
            mrr = 1.0 / (i + 1)
            break
    return (sum(hits) / max(len(true_docs), 1), mrr)


def main():
    cfg = yaml.safe_load(open(os.path.join(REPO, 'configs', 'config.yaml'),
                              encoding='utf-8'))
    client = OpenSearch(
        hosts=[{'host': cfg['opensearch']['host'],
                'port': cfg['opensearch']['port']}],
        use_ssl=False, verify_certs=False, ssl_show_warn=False)

    qrels = [q for q in json.load(
        open(os.path.join(REPO, 'data', 'qrels_expanded.json'), encoding='utf-8'))
        if q.get('relevant_doc_ids')]
    q57 = next(q for q in qrels if q['query_id'] == 'q57')
    rest = [q for q in qrels if q['query_id'] != 'q57']

    print('=' * 100)
    print('1. THE DEFECT')
    print('=' * 100)
    years = []
    for pid in q57['relevant_doc_ids']:
        r = client.search(index='ethsearch_chunks', body={
            'size': 1, 'query': {'term': {'parent_doc_id': pid}}})
        years.append(r['hits']['hits'][0]['_source'].get('publication_year'))
    print(f'  query      : {q57["query_text"]!r}')
    print(f'  Manifest Destiny: {MD_START}-{MD_END}, Polk: {POLK_START}-{POLK_END}')
    print(f'  ref years  : {sorted(years)}')
    print(f'  every reference predates the earliest concept boundary '
          f'({MD_START}) by {MD_START - max(years)}-{MD_START - min(years)} years')
    print('  the references are also topically wrong: they matched on the token')
    print('  "administration" and are Jackson-era partisan polemic.')

    # ---- build a counterfactual corrected reference set, era-scoped ----
    print()
    print('=' * 100)
    print('2. CANDIDATE CORRECTION (counterfactual only)')
    print('=' * 100)
    r = client.search(index='ethsearch_chunks', body={
        'size': 50, '_source': ['publication_year', 'text', 'parent_doc_id'],
        'query': {
            'bool': {
                'must': [{'match': {'text': 'manifest destiny'}}],
                'filter': [{'range': {'publication_year': {
                    'gte': MD_START, 'lte': MD_END}}}]
            }
        }})
    corrected = []
    seen = set()
    for h in r['hits']['hits']:
        pid = h['_source']['parent_doc_id']
        if pid in seen:
            continue
        seen.add(pid)
        corrected.append((pid, h['_source'].get('publication_year'),
                          h['_source']['text'][:96].replace('\n', ' ')))
    print(f'  era-scoped "manifest destiny" hits: {r["hits"]["total"]["value"]}')
    for pid, y, t in corrected[:8]:
        print(f'    {pid:<13} {y}  {t}')

    corrected_ids = [pid for pid, _, _ in corrected[:5]]

    # ---- compare the two dispositions ----
    print()
    print('=' * 100)
    print('3. WHAT EACH DISPOSITION DOES TO THE HEADLINE CONTRASTS')
    print('=' * 100)
    rankers = {'BM25': get_ranker('BM25', 'expanded'),
               'TemporalOnly': get_ranker('TemporalOnly', 'expanded')}
    rankers['Hybrid+Temporal'] = get_ranker('Hybrid+Temporal', 'expanded')

    def run(query_set, override=None):
        out = {}
        for name, fn in rankers.items():
            rec, mrr = [], []
            for q in query_set:
                refs = (override if override is not None
                        and q['query_id'] == 'q57' else set(q['relevant_doc_ids']))
                r_, m_ = evaluate(fn(q['query_text'], size=SHARED_FETCH_SIZE),
                                  refs)
                rec.append(r_)
                mrr.append(m_)
            out[name] = (float(np.mean(rec)), float(np.mean(mrr)))
        return out

    scenarios = {
        'A_current (n=88, q57 as-is)': run(qrels),
        'B_exclude (n=87, q57 dropped)': run(rest),
        'C_hand-correct (n=88, q57 era-aligned)': run(qrels, set(corrected_ids)),
    }

    print()
    print(f'  {"scenario":<38}{"BM25 R@10":>11}{"TempOnly R@10":>15}'
          f'{"delta":>9}{"q57 R@10":>11}')
    for name, res in scenarios.items():
        d = res['TemporalOnly'][0] - res['BM25'][0]
        print(f'  {name:<38}{res["BM25"][0]:>11.4f}{res["TemporalOnly"][0]:>15.4f}'
              f'{d:>+9.4f}{"":>11}')

    print()
    for name, res in scenarios.items():
        d = res['TemporalOnly'][0] - res['BM25'][0]
        print(f'  {name}')
        print(f'      headline temporal effect (TemporalOnly - BM25) = {d:+.4f}')

    # q57's own recall under each option
    print()
    print('  q57 Recall@10 by disposition:')
    for name, res in scenarios.items():
        pass
    r_now = evaluate(rankers['BM25'](q57['query_text'], size=SHARED_FETCH_SIZE),
                     set(q57['relevant_doc_ids']))
    r_cor = evaluate(rankers['BM25'](q57['query_text'], size=SHARED_FETCH_SIZE),
                     set(corrected_ids))
    print(f'      as-is (1829-1839 refs)   BM25 R@10 = {r_now[0]:.4f}')
    print(f'      era-aligned refs         BM25 R@10 = {r_cor[0]:.4f}')

    out = {
        'defect': 'q57 references are 1829-1839, 0/5 inside Manifest Destiny '
                  '(1843-1853), and match on the token "administration" rather '
                  'than the topic',
        'scenarios': {k: {rk: {'recall': v[0], 'mrr': v[1]}
                          for rk, v in res.items()}
                      for k, res in scenarios.items()},
        'headline_effect_by_scenario': {
            k: res['TemporalOnly'][0] - res['BM25'][0] for k, res in scenarios.items()},
        'counterfactual_corrected_ids': corrected_ids,
        'recommendation': None,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote evaluation/{os.path.basename(OUT)}')


if __name__ == '__main__':
    main()
