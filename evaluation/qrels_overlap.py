import json
d = json.load(open('data/qrels_human_annotated.json'))
a = json.load(open('data/qrels_human_annotated_audit.json'))
pool = {'q5': 7, 'q9': 10, 'q14': 9, 'q21': 17, 'q31': 17,
        'q42': 20, 'q71': 20, 'q1': 18, 'q3': 20, 'q101': 20}

print("QUERY / POOL / RELEVANT BREAKDOWN")
print(f"{'qid':<7}{'corpus':<11}{'pool':<6}{'rel':<6}{'rel%':<8}{'temporal':<15}text")
for q in d['queries']:
    n = len(q['relevant_doc_ids'])
    p = pool[q['query_id']]
    print(f"{q['query_id']:<7}{q['corpus']:<11}{p:<6}{n:<6}{100*n/p:<7.0f}%"
          f"{q['temporal_type']:<15}{q['query_text'][:34]}")
print(f"\nTOTAL pool={sum(pool.values())}  relevant={sum(len(q['relevant_doc_ids']) for q in d['queries'])}")

print("\nPER-QUERY KAPPA")
for qid, s in sorted(a['per_query_stats'].items()):
    k = s['kappa']
    print(f"  {qid:<7} kappa={k:+.3f}   agree={100*s['pct']:.0f}%  (n={s['n']})")

# Relevant-set overlap between term-overlap and human qrels
small = json.load(open('data/qrels_small.json'))
exp = json.load(open('data/qrels_expanded.json'))
term = {}
for q in small:
    term.setdefault(q['query_id'], set()).update(q['relevant_doc_ids'])
for q in exp:
    term.setdefault(q['query_id'], set()).update(q['relevant_doc_ids'])

print("\nRELEVANT-SET OVERLAP (term-overlap vs human), per query")
tot_j = tot_o = 0
for q in d['queries']:
    h = set(q['relevant_doc_ids'])
    t = term.get(q['query_id'], set())
    inter = h & t
    print(f"  {q['query_id']:<7} human={len(h):<4} term={len(t):<4} "
          f"overlap={len(inter):<4} jaccard={len(inter)/max(len(h|t),1):.2f}")
    tot_j += len(h | t)
    tot_o += len(inter)
print(f"\n  micro Jaccard (relevant sets): {tot_o}/{tot_j} = {tot_o/tot_j:.2f}")
