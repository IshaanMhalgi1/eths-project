import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from temporal.temporal_parser import TemporalParser

p = TemporalParser()
d = json.load(open('data/qrels_human_annotated.json'))
print("TEMPORAL PARSER OUTPUT ON THE 10 ANNOTATED QUERIES")
print(f"{'qid':<7}{'declared':<15}{'start':<7}{'end':<7}{'parsed?':<9}text")
for q in d['queries']:
    ti = p.parse(q['query_text'])
    s, e = ti.get('start_year'), ti.get('end_year')
    ok = 'YES' if (s is not None and e is not None) else 'NO'
    print(f"{q['query_id']:<7}{q['temporal_type']:<15}{str(s):<7}{str(e):<7}{ok:<9}{q['query_text'][:38]}")

n_parsed = sum(1 for q in d['queries']
               if p.parse(q['query_text']).get('start_year') is not None)
print(f"\nparsed with a year constraint: {n_parsed}/10")
temporal_qs = [q for q in d['queries'] if q['temporal_type'] != 'none']
print(f"queries declared temporal:     {len(temporal_qs)}/10")
missed = [q['query_id'] for q in temporal_qs
          if p.parse(q['query_text']).get('start_year') is None]
print(f"declared-temporal but unparsed: {missed if missed else 'none'}")
