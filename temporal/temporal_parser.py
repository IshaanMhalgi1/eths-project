import re
import json
import os

GAZETTEER_PATH = os.path.join(os.path.dirname(__file__), 'period_gazetteer.json')

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
            start_year = 0
            end_year = int(before[0]) - 1
        elif after:
            start_year = int(after[0]) + 1
            end_year = 2026
        elif years:
            start_year = int(years[0])
            end_year = int(years[0])
            
        return {
            "start_year": start_year,
            "end_year": end_year,
            "periods": [p['name'] for p in matched_periods]
        }

if __name__ == '__main__':
    parser = TemporalParser()
    print("Testing TemporalParser:")
    print("Civil War ->", parser.parse("what happened during the Civil War?"))
    print("before 1900 ->", parser.parse("events before 1900"))
    print("1850-1860 ->", parser.parse("newspaper from 1850-1860"))
