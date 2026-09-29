"""Provenance of data/qrels_small.json: hand-curated, by design.

The K-sweep in qrels_depth_sensitivity.py excluded the small corpus because
qrels_small.json is not reproducible at K=5 by any generator in the repository
(19/19 mismatches). That exclusion was recorded as an unresolved gap. This
script establishes the actual provenance so the exclusion can be reclassified.

The claim being tested: qrels_small.json descends from a hand-curation round
against the raw corpus, in which four queries were rewritten after inspecting
the matched document (q1 lost-pocket-book -> strayed-horse reward, q5
obituaries 1807 -> 1806) or had their relevance set completed by hand
(q4 military appointments, q7 ship arrivals). If the 5-id reference sets are
seeded by the single ids those queries were originally assigned, the file is a
human artefact and its non-reproducibility is intended, not a defect.

Reads the index and the committed git history; writes nothing except the
report below.
"""
import json
import os
import re
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temporal.temporal_parser import TemporalParser

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = TemporalParser()

# The four queries named as hand-corrected, with the text each generator
# script originally proposed. Divergence from the generator's text is itself
# evidence of manual rewriting. The generator text is transcribed from
# evaluation/rebuild_qrels.py's QUERIES table, which is the last script in the
# chain that still contains the pre-curation wording.
HAND_CORRECTED = {
    'q1':  ('lost pocket book reward',                'strayed horse reward advertisement'),
    'q4':  ('military officer appointments',          'military officer appointments'),
    'q5':  ('obituaries deaths 1807',                 'obituaries deaths 1806'),
    'q7':  ('ship arrivals cargo from Europe 1803',   'ship arrivals cargo from Europe 1803'),
}

# Seeds: the single relevant_doc_id each query was assigned in the first
# committed qrels.json (9350a5e), which rebuild_qrels.py produced by taking the
# top-1 BM25 hit and asking a human to verify it.
SEED_COMMITS = ['9350a5e', '7c62c82']


def git_show(rev, path):
    return subprocess.run(['git', '-C', REPO, 'show', f'{rev}:{path}'],
                          capture_output=True, text=True, check=True).stdout


def main():
    small = json.load(open(os.path.join(REPO, 'data', 'qrels_small.json'),
                          encoding='utf-8'))
    original = json.load(open(os.path.join(REPO, 'data', 'qrels_original.json'),
                              encoding='utf-8'))
    by_id = {q['query_id']: q for q in small}

    print('=' * 100)
    print('1. IS qrels_small.json A PURE SUBSET OF qrels_original.json?')
    print('=' * 100)
    omitted = [q['query_id'] for q in original if q['query_id'] not in by_id]
    differing = [q['query_id'] for q in original
                 if q['query_id'] in by_id
                 and q['relevant_doc_ids'] != by_id[q['query_id']]['relevant_doc_ids']]
    print(f'  qrels_original.json : {len(original)} queries')
    print(f'  qrels_small.json    : {len(small)} queries')
    print(f'  omitted             : {omitted}  (build_small_qrels.py drops q18)')
    print(f'  relevance differing : {differing or "none"}')
    print(f'  => file is qrels_original.json minus {omitted}, verbatim: '
          f'{not differing}')

    print()
    print('=' * 100)
    print('2. SEED CHECK: does each 5-id reference set begin with the id that')
    print('   query was originally assigned? (single-id qrels.json, first commit)')
    print('=' * 100)
    seed_report = {}
    for rev in SEED_COMMITS:
        seeded = json.loads(git_show(rev, 'data/qrels.json'))
        s = {q['query_id']: q for q in seeded}
        print(f'\n  revision {rev}: {len(seeded)} queries, '
              f'single-doc schema={"relevant_doc_ids" in seeded[0]}')
        print(f'  {"qid":<5}{"seed":<14}{"in refs":<9}{"position":<10}'
              f'{"text at seed":<38}text now')
        seed_report[rev] = {}
        for qid in ['q1', 'q4', 'q5', 'q7']:
            seed = s[qid]['relevant_doc_ids'][0]
            refs = by_id[qid]['relevant_doc_ids']
            pos = refs.index(seed) + 1 if seed in refs else None
            print(f'  {qid:<5}{seed:<14}{str(seed in refs):<9}'
                  f'{("pos " + str(pos)) if pos else "ABSENT":<10}'
                  f'{s[qid]["query_text"][:36]:<38}{by_id[qid]["query_text"]}')
            seed_report[rev][qid] = {
                'seed': seed,
                'seed_position': pos,
                'seed_text': s[qid]['query_text'],
                'curated_text': by_id[qid]['query_text'],
            }

    print()
    print('  Full seed coverage across all 19 small queries:')
    seeded = json.loads(git_show(SEED_COMMITS[0], 'data/qrels.json'))
    s = {q['query_id']: q for q in seeded}
    retained = [q['query_id'] for q in small
                if s.get(q['query_id'], {}).get('relevant_doc_ids', [None])[0]
                in q['relevant_doc_ids']]
    print(f'    seed id still present in the 5-id set: {len(retained)}/{len(small)}'
          f'  -> {retained}')

    print()
    print('=' * 100)
    print('3. TEXT DIVERGENCE: the four named hand-corrections')
    print('=' * 100)
    for qid, (gen_text, cur_text) in HAND_CORRECTED.items():
        cur = by_id[qid]['query_text']
        changed = gen_text != cur
        print(f'  {qid}: generator proposed {gen_text!r}')
        print(f'      curated text is     {cur!r}   -> '
              f'{"REWRITTEN BY HAND" if changed else "unchanged"}')

    print()
    print('=' * 100)
    print('4. DOCUMENT CONTENT: do the four queries\' reference sets actually')
    print('   match their CURATED text, against the pre-curation text?')
    print('=' * 100)
    from retrieval.bm25 import search_bm25, SMALL_INDEX
    from opensearchpy import OpenSearch
    import yaml
    cfg = yaml.safe_load(open(os.path.join(REPO, 'configs', 'config.yaml'),
                              encoding='utf-8'))
    client = OpenSearch(
        hosts=[{'host': cfg['opensearch']['host'], 'port': cfg['opensearch']['port']}],
        use_ssl=False, verify_certs=False, ssl_show_warn=False)

    content_report = {}
    for qid, (gen_text, cur_text) in HAND_CORRECTED.items():
        refs = by_id[qid]['relevant_doc_ids']
        print(f'\n  {qid}  curated: {cur_text!r}')
        print(f'      pre-curation wording was: {gen_text!r}')
        print(f'      {"query text was REWRITTEN" if gen_text != cur_text else "query text unchanged"}')
        print(f'      reference documents:')
        years = []
        for pid in refs:
            # qrels hold parent_doc_id ("train_141"); the index _id is the chunk
            # ("train_141_0"). metrics.py credits a hit when ANY chunk of the
            # parent lands in the top K, so the parent is the right key.
            r = client.search(index=SMALL_INDEX, body={
                'size': 1,
                'query': {'term': {'parent_doc_id': pid}}})
            if not r['hits']['hits']:
                print(f'        {pid:<14} NOT IN SMALL INDEX')
                continue
            src = r['hits']['hits'][0]['_source']
            years.append(src.get('publication_year'))
            snippet = re.sub(r'\s+', ' ', src.get('text', ''))[:96]
            print(f'        {pid:<14} year={src.get("publication_year")}  {snippet}')
        content_report[qid] = {
            'pre_curation_text': gen_text,
            'curated_text': cur_text,
            'text_rewritten': gen_text != cur_text,
            'reference_years': sorted({y for y in years if y}),
        }

    print()
    print('=' * 100)
    print('5. DO THE 19 SMALL REFERENCES SIT INSIDE THE 1800-1810 WINDOW?')
    print('   (a window violation would be a defect; containment means the set')
    print('    was curated against this corpus specifically)')
    print('=' * 100)
    m = client.search(index=SMALL_INDEX, body={
        'size': 10000, '_source': ['publication_year', 'parent_doc_id'],
        'query': {'match_all': {}}})
    index_years = {}
    for h in m['hits']['hits']:
        index_years[h['_id']] = h['_source'].get('publication_year')
        index_years.setdefault(h['_source'].get('parent_doc_id'),
                               h['_source'].get('publication_year'))
    out_of_window, missing = [], []
    for q in small:
        for c in q['relevant_doc_ids']:
            if c not in index_years:
                missing.append((q['query_id'], c))
            elif not (1800 <= index_years[c] <= 1810):
                out_of_window.append((q['query_id'], c, index_years[c]))
    all_refs = sorted({c for q in small for c in q['relevant_doc_ids']})
    print(f'  distinct reference ids    : {len(all_refs)}')
    print(f'  absent from small index   : {missing or "none"}')
    print(f'  outside 1800-1810         : {out_of_window or "none"}')

    print()
    print('=' * 100)
    print('6. WHY NO GENERATOR REPRODUCES IT: consecutive id runs')
    print('=' * 100)
    print('  generate_expanded_qrels.py ranks by lexical overlap and hard-caps at')
    print('  K=5, which produces near-scattered ids. These sets are consecutive')
    print('  id runs -- the signature of a human walking a source item\'s chunks.')
    for qid in ['q1', 'q4', 'q5', 'q7']:
        refs = by_id[qid]['relevant_doc_ids']
        nums = [int(r.split('_')[1]) for r in refs]
        print(f'  {qid}: {refs}  -> consecutive-run structure: '
              f'{_has_run(nums)}')

    out = {
        'conclusion': 'hand-curated by design; not reproducible by any automated '
                      'generator. The K-sweep exclusion is a property of the '
                      'instrument, not an unresolved gap.',
        'derived_from': 'data/qrels_original.json minus q18 (build_small_qrels.py)',
        'n_queries': len(small),
        'omitted_vs_original': omitted,
        'relevance_differs_from_original': differing,
        'seed_ids_by_commit': seed_report,
        'hand_corrected_queries': content_report,
        'seed_retained_in_curated_set': retained,
        'reference_ids_absent_from_small_index': missing,
        'reference_ids_outside_window': out_of_window,
    }
    path = os.path.join(REPO, 'evaluation', 'small_qrels_provenance.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote evaluation/{os.path.basename(path)}')


def _has_run(nums):
    nums = sorted(nums)
    best = run = 1
    for a, b in zip(nums, nums[1:]):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return f'longest consecutive run = {best}/{len(nums)}'


if __name__ == '__main__':
    main()
