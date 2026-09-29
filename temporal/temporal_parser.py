import re
import json
import os

import yaml

GAZETTEER_PATH = os.path.join(os.path.dirname(__file__), 'period_gazetteer.json')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')

# A resolved window is treated as non-discriminating when it covers at least this
# fraction of the corpus year span, because a window that admits almost every
# indexed year cannot separate candidates on date.
NON_DISCRIMINATING_COVERAGE = 0.95

with open(CONFIG_PATH, 'r') as _f:
    _cfg = yaml.safe_load(_f)

# configs/config.yaml states an EXCLUSIVE end bound (1870 => corpus is 1800-1869
# inclusive). The sentinel years below are clamped to this span rather than to 0
# and 2026, because "before Y" and "after Y" were resolving to windows that ran
# far outside the indexed data. A window of (0, 1807) reads as if it excludes
# nothing, when against a 1800-1869 corpus it in fact excludes six of seven
# decades. Clamping makes the window say what it means and lets coverage be
# measured against the real span.
CORPUS_START = int(_cfg.get('corpus_start_year', 1800))
CORPUS_END = int(_cfg.get('corpus_end_year', 1870)) - 1  # exclusive -> inclusive
CORPUS_SPAN = CORPUS_END - CORPUS_START + 1


class TemporalParser:
    def __init__(self):
        with open(GAZETTEER_PATH, 'r', encoding='utf-8') as f:
            self.gazetteer = json.load(f)

    def parse(self, query: str):
        # 1. Check for exact 4 digit years
        years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', query)
        
        # 2. Check for ranges (e.g. 1850-1860, 1850 to 1860, 1850 1860)
        ranges = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])(?:\s*(?:-|to)\s*|\s+)(1[0-9]{3}|20[0-2][0-9])\b', query)
        
        # 3. Check for operators (before, after, around, since)
        before = re.findall(r'before\s+(1[0-9]{3}|20[0-2][0-9])\b', query, re.IGNORECASE)
        after = re.findall(r'(?:after|since)\s+(1[0-9]{3}|20[0-2][0-9])\b', query, re.IGNORECASE)
        
        # 4. Check for period names
        matched_periods = []
        for p in self.gazetteer:
            if re.search(r'\b' + re.escape(p['name']) + r'\b', query, re.IGNORECASE):
                matched_periods.append(p)
            else:
                for alias in p.get('aliases', []):
                    if re.search(r'\b' + re.escape(alias) + r'\b', query, re.IGNORECASE):
                        matched_periods.append(p)
                        break

        # Resolve into a bounding box (start_year, end_year)
        start_year = None
        end_year = None
        
        if ranges:
            start_year = int(ranges[0][0])
            end_year = int(ranges[0][1])
        elif matched_periods:
            start_year = matched_periods[0]['start_year']
            end_year = matched_periods[0]['end_year']
        elif before:
            # Clamped to the corpus, not to 0. "Before 1808" against a
            # 1800-1869 corpus means 1800-1807, not "everything up to 1807".
            start_year = CORPUS_START
            end_year = int(before[0]) - 1
        elif after:
            # Clamped to the corpus, not to 2026. See above.
            start_year = int(after[0]) + 1
            end_year = CORPUS_END
        elif years:
            start_year = int(years[0])
            end_year = int(years[0])

        # is_constrained answers a different question from start_year/end_year
        # being non-None. A window can be resolved and still fail to separate
        # anything, either because it was clamped until it swallowed the corpus
        # (e.g. "since 1802" -> 1803-1869, 67 of 70 years) or because a period
        # name resolved to a range wider than the data. Callers that branch on
        # "did we get a window" would otherwise treat those as real constraints
        # and pay a z-scored temporal adjustment for a term that is constant
        # across the candidate pool.
        is_constrained = False
        coverage = None
        if start_year is not None and end_year is not None:
            lo = max(start_year, CORPUS_START)
            hi = min(end_year, CORPUS_END)
            overlap = max(0, hi - lo + 1)
            coverage = overlap / CORPUS_SPAN
            # An empty intersection cannot select anything, so it is not a
            # constraint either -- but it is a real answer to "restrict to this
            # range", and is_constrained stays False so callers treat it as the
            # no-op it effectively is rather than scoring it as a match.
            is_constrained = coverage < NON_DISCRIMINATING_COVERAGE

        return {
            "start_year": start_year,
            "end_year": end_year,
            "periods": [p['name'] for p in matched_periods],
            "is_constrained": is_constrained,
            "corpus_coverage": coverage,
        }

if __name__ == '__main__':
    parser = TemporalParser()
    print("Testing TemporalParser:")
    print("Civil War ->", parser.parse("what happened during the Civil War?"))
    print("before 1900 ->", parser.parse("events before 1900"))
    print("1850-1860 ->", parser.parse("newspaper from 1850-1860"))
    print("before 1808 ->", parser.parse("news dispatches France Europe before 1808"))
    print("since 1802  ->", parser.parse("medicines apothecary bitters since 1802"))
