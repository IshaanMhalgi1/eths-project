import os
import json
import sys
import re
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from temporal.temporal_parser import TemporalParser

parser = TemporalParser()

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.jsonl')
QRELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels_expanded.json')

# Load all chunks into memory
chunks = []
with open(CHUNKS_PATH, 'r') as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

print(f"Loaded {len(chunks)} chunks")

# Expanded query set for 1800-1900 corpus
# Mix of temporal types across all decades
expanded_queries = [
    # Explicit year queries - spread across decades
    {"query_id": "q1", "query_text": "strayed horse reward advertisement", "temporal_type": "none"},
    {"query_id": "q2", "query_text": "flour and grain market prices", "temporal_type": "none"},
    {"query_id": "q3", "query_text": "runaway slave reward advertisement", "temporal_type": "none"},
    {"query_id": "q4", "query_text": "military officer appointments", "temporal_type": "none"},
    {"query_id": "q5", "query_text": "obituaries deaths 1806", "temporal_type": "explicit_year"},
    {"query_id": "q6", "query_text": "real estate land for sale 1807", "temporal_type": "explicit_year"},
    {"query_id": "q7", "query_text": "ship arrivals cargo from Europe 1803", "temporal_type": "explicit_year"},
    {"query_id": "q8", "query_text": "new store goods for sale 1803", "temporal_type": "explicit_year"},
    {"query_id": "q9", "query_text": "political news congress 1800 to 1805", "temporal_type": "range"},
    {"query_id": "q10", "query_text": "tax land assessment 1804 to 1808", "temporal_type": "range"},
    {"query_id": "q11", "query_text": "legal public notice sheriff 1800 1810", "temporal_type": "range"},
    {"query_id": "q12", "query_text": "weather storm floods 1805 to 1810", "temporal_type": "range"},
    {"query_id": "q13", "query_text": "news dispatches France Europe before 1808", "temporal_type": "before_after"},
    {"query_id": "q14", "query_text": "shipping port arrivals after 1805", "temporal_type": "before_after"},
    {"query_id": "q15", "query_text": "medicines apothecary bitters since 1802", "temporal_type": "before_after"},
    {"query_id": "q16", "query_text": "poetry verse published before 1810", "temporal_type": "before_after"},
    {"query_id": "q17", "query_text": "European war news Jeffersonian Era", "temporal_type": "period"},
    {"query_id": "q18", "query_text": "election campaign republican Early Republic", "temporal_type": "period"},
    {"query_id": "q19", "query_text": "trade embargo Britain Jefferson administration", "temporal_type": "period"},
    {"query_id": "q20", "query_text": "letters editor public opinion Early National Period", "temporal_type": "period"},
    
    # 1810s
    {"query_id": "q21", "query_text": "War of 1812 navy battles", "temporal_type": "explicit_year"},
    {"query_id": "q22", "query_text": "British blockade American ports 1813", "temporal_type": "explicit_year"},
    {"query_id": "q23", "query_text": "Treaty of Ghent peace negotiations 1814", "temporal_type": "explicit_year"},
    {"query_id": "q24", "query_text": "Battle of New Orleans Jackson 1815", "temporal_type": "explicit_year"},
    {"query_id": "q25", "query_text": "Era of Good Feelings Monroe presidency", "temporal_type": "period"},
    {"query_id": "q26", "query_text": "Missouri Compromise debates 1819 1820", "temporal_type": "range"},
    {"query_id": "q27", "query_text": "steamboat navigation Mississippi River 1817", "temporal_type": "explicit_year"},
    {"query_id": "q28", "query_text": "Panic of 1819 economic depression", "temporal_type": "explicit_year"},
    {"query_id": "q29", "query_text": "Adams Onis Treaty Florida 1819", "temporal_type": "explicit_year"},
    {"query_id": "q30", "query_text": "coroner inquest sudden death 1815", "temporal_type": "explicit_year"},
    
    # 1820s
    {"query_id": "q31", "query_text": "Monroe Doctrine foreign policy 1823", "temporal_type": "explicit_year"},
    {"query_id": "q32", "query_text": "corrupt bargain election 1824 Adams Jackson", "temporal_type": "explicit_year"},
    {"query_id": "q33", "query_text": "Erie Canal completion 1825", "temporal_type": "explicit_year"},
    {"query_id": "q34", "query_text": "railroad construction Baltimore Ohio 1828", "temporal_type": "explicit_year"},
    {"query_id": "q35", "query_text": "tariff abominations nullification crisis 1828", "temporal_type": "explicit_year"},
    {"query_id": "q36", "query_text": "temperance movement societies 1826", "temporal_type": "explicit_year"},
    {"query_id": "q37", "query_text": "Lafayette tour United States 1824 1825", "temporal_type": "range"},
    {"query_id": "q38", "query_text": "Sunday mail controversy sabbath 1829", "temporal_type": "explicit_year"},
    {"query_id": "q39", "query_text": "Indian removal debates Georgia 1828 1830", "temporal_type": "range"},
    {"query_id": "q40", "query_text": "Masonic anti-Masonic party 1826", "temporal_type": "explicit_year"},
    
    # 1830s
    {"query_id": "q41", "query_text": "Jackson Bank War veto 1832", "temporal_type": "explicit_year"},
    {"query_id": "q42", "query_text": "Trail of Tears Cherokee removal 1838", "temporal_type": "explicit_year"},
    {"query_id": "q43", "query_text": "abolitionist movement Liberator Garrison 1831", "temporal_type": "explicit_year"},
    {"query_id": "q44", "query_text": "Panic of 1837 financial crisis", "temporal_type": "explicit_year"},
    {"query_id": "q45", "query_text": "Texas independence revolution 1836", "temporal_type": "explicit_year"},
    {"query_id": "q46", "query_text": "Mormon settlement Missouri Illinois 1839", "temporal_type": "explicit_year"},
    {"query_id": "q47", "query_text": "patent office inventions agriculture 1836", "temporal_type": "explicit_year"},
    {"query_id": "q48", "query_text": "colera epidemic prevention 1832", "temporal_type": "explicit_year"},
    {"query_id": "q49", "query_text": "Whig party formation opposition 1834", "temporal_type": "explicit_year"},
    {"query_id": "q50", "query_text": "Samuel Morse telegraph demonstration 1838", "temporal_type": "explicit_year"},
    
    # 1840s
    {"query_id": "q51", "query_text": "Mexican American War 1846 1848", "temporal_type": "range"},
    {"query_id": "q52", "query_text": "Oregon Trail pioneers westward 1843", "temporal_type": "explicit_year"},
    {"query_id": "q53", "query_text": "gold rush California forty niners 1849", "temporal_type": "explicit_year"},
    {"query_id": "q54", "query_text": "Irish potato famine immigration 1847", "temporal_type": "explicit_year"},
    {"query_id": "q55", "query_text": "Seneca Falls women rights convention 1848", "temporal_type": "explicit_year"},
    {"query_id": "q56", "query_text": "telegraph lines expansion 1844", "temporal_type": "explicit_year"},
    {"query_id": "q57", "query_text": "Manifest Destiny Polk administration", "temporal_type": "period"},
    {"query_id": "q58", "query_text": "Wilmot Proviso slavery territories 1846", "temporal_type": "explicit_year"},
    {"query_id": "q59", "query_text": "railroad mania stock speculation 1847", "temporal_type": "explicit_year"},
    {"query_id": "q60", "query_text": "Cholera epidemic New York 1849", "temporal_type": "explicit_year"},
    
    # 1850s
    {"query_id": "q61", "query_text": "Compromise of 1850 Fugitive Slave Act", "temporal_type": "explicit_year"},
    {"query_id": "q62", "query_text": "Kansas Nebraska Act popular sovereignty 1854", "temporal_type": "explicit_year"},
    {"query_id": "q63", "query_text": "Dred Scott decision Supreme Court 1857", "temporal_type": "explicit_year"},
    {"query_id": "q64", "query_text": "Lincoln Douglas debates Illinois 1858", "temporal_type": "explicit_year"},
    {"query_id": "q65", "query_text": "John Brown raid Harpers Ferry 1859", "temporal_type": "explicit_year"},
    {"query_id": "q66", "query_text": "Pony Express mail delivery 1860", "temporal_type": "explicit_year"},
    {"query_id": "q67", "query_text": "Know Nothing party nativism 1855", "temporal_type": "explicit_year"},
    {"query_id": "q68", "query_text": "transcontinental railroad surveys 1853", "temporal_type": "explicit_year"},
    {"query_id": "q69", "query_text": "crimean war European news 1854 1856", "temporal_type": "range"},
    {"query_id": "q70", "query_text": "cotton market prices Liverpool 1850", "temporal_type": "explicit_year"},
    
    # 1860s
    {"query_id": "q71", "query_text": "Civil War Fort Sumter 1861", "temporal_type": "explicit_year"},
    {"query_id": "q72", "query_text": "Emancipation Proclamation 1863", "temporal_type": "explicit_year"},
    {"query_id": "q73", "query_text": "Gettysburg battle casualties 1863", "temporal_type": "explicit_year"},
    {"query_id": "q74", "query_text": "Sherman march to sea 1864", "temporal_type": "explicit_year"},
    {"query_id": "q75", "query_text": "Lincoln assassination 1865", "temporal_type": "explicit_year"},
    {"query_id": "q76", "query_text": "Reconstruction acts Congress 1867", "temporal_type": "explicit_year"},
    {"query_id": "q77", "query_text": "transcontinental railroad completed 1869", "temporal_type": "explicit_year"},
    {"query_id": "q78", "query_text": "Homestead Act western settlement 1862", "temporal_type": "explicit_year"},
    {"query_id": "q79", "query_text": "Morrill Land Grant colleges 1862", "temporal_type": "explicit_year"},
    {"query_id": "q80", "query_text": "freedmen bureau education 1866", "temporal_type": "explicit_year"},
    
    # 1870s
    {"query_id": "q81", "query_text": "Grant administration corruption scandals 1872", "temporal_type": "explicit_year"},
    {"query_id": "q82", "query_text": "Panic of 1873 Long Depression", "temporal_type": "explicit_year"},
    {"query_id": "q83", "query_text": "Centennial Exhibition Philadelphia 1876", "temporal_type": "explicit_year"},
    {"query_id": "q84", "query_text": "Battle of Little Bighorn Custer 1876", "temporal_type": "explicit_year"},
    {"query_id": "q85", "query_text": "telephone invention Bell 1876", "temporal_type": "explicit_year"},
    {"query_id": "q86", "query_text": "electric light Edison 1879", "temporal_type": "explicit_year"},
    {"query_id": "q87", "query_text": "Chinese Exclusion Act 1882", "temporal_type": "explicit_year"},
    {"query_id": "q88", "query_text": "labor strikes railroad 1877", "temporal_type": "explicit_year"},
    {"query_id": "q89", "query_text": "women suffrage movement 1870s", "temporal_type": "range"},
    {"query_id": "q90", "query_text": "yellow fever epidemic Memphis 1878", "temporal_type": "explicit_year"},
    
    # 1880s
    {"query_id": "q91", "query_text": "Brooklyn Bridge opening 1883", "temporal_type": "explicit_year"},
    {"query_id": "q92", "query_text": "Haymarket affair labor 1886", "temporal_type": "explicit_year"},
    {"query_id": "q93", "query_text": "Statue of Liberty dedication 1886", "temporal_type": "explicit_year"},
    {"query_id": "q94", "query_text": "Dawes Act Native American lands 1887", "temporal_type": "explicit_year"},
    {"query_id": "q95", "query_text": "Johnstown flood disaster 1889", "temporal_type": "explicit_year"},
    {"query_id": "q96", "query_text": "interstate commerce act regulation 1887", "temporal_type": "explicit_year"},
    {"query_id": "q97", "query_text": "typewriter commercial production 1884", "temporal_type": "explicit_year"},
    {"query_id": "q98", "query_text": "bicycle craze transportation 1880s", "temporal_type": "range"},
    {"query_id": "q99", "query_text": "time zones standardization 1883", "temporal_type": "explicit_year"},
    {"query_id": "q100", "query_text": "weather bureau forecasts 1880", "temporal_type": "explicit_year"},
    
    # Non-temporal / general
    {"query_id": "q101", "query_text": "agricultural fair prize livestock", "temporal_type": "none"},
    {"query_id": "q102", "query_text": "temperance pledge total abstinence", "temporal_type": "none"},
    {"query_id": "q103", "query_text": "marriage license ceremony announcement", "temporal_type": "none"},
    {"query_id": "q104", "query_text": "church revival camp meeting", "temporal_type": "none"},
    {"query_id": "q105", "query_text": "patent medicine cure all remedy", "temporal_type": "none"},
    {"query_id": "q106", "query_text": "fire engine company volunteer", "temporal_type": "none"},
    {"query_id": "q107", "query_text": "school examination commencement exercises", "temporal_type": "none"},
    {"query_id": "q108", "query_text": "theater performance benefit concert", "temporal_type": "none"},
    {"query_id": "q109", "query_text": "merchant tailor clothing advertisement", "temporal_type": "none"},
    {"query_id": "q110", "query_text": "lost found stray animals reward", "temporal_type": "none"},
]

def get_terms(text):
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    stopwords = {"and", "for", "the", "in", "to", "of", "with", "from", "on", "at", "by", "this", "that", "it", "as", "is", "are", "was", "were", "be", "been", "has", "have", "had", "will", "would", "could", "should", "not", "no", "since", "before", "after", "during"}
    return set(w for w in words if w not in stopwords)

new_qrels = []

for q in expanded_queries:
    query = q['query_text']
    intent = parser.parse(query)
    start_y = intent.get('start_year')
    end_y = intent.get('end_year')
    
    query_terms = get_terms(query)
    
    scored_chunks = []
    
    for c in chunks:
        doc_year = c.get('publication_year')
        
        # Exact year filtering independent of BM25
        if start_y is not None and end_y is not None:
            if doc_year is None or not (start_y <= doc_year <= end_y):
                continue
                
        chunk_terms = get_terms(c['text'])
        overlap = len(query_terms.intersection(chunk_terms))
        
        if overlap > 0:
            scored_chunks.append({
                'parent_doc_id': c['parent_doc_id'],
                'overlap': overlap
            })
            
    # Sort by overlap
    scored_chunks.sort(key=lambda x: x['overlap'], reverse=True)
    
    # Deduplicate by parent_doc_id
    seen = set()
    dedup = []
    for sc in scored_chunks:
        if sc['parent_doc_id'] not in seen:
            seen.add(sc['parent_doc_id'])
            dedup.append(sc)
            
    # Get top chunks
    if not dedup:
        print(f"Warning: No chunks found for '{query}' with range {start_y}-{end_y}")
        relevant_docs = []
    else:
        # Take chunks that have the highest overlap, up to 5
        top_overlap = dedup[0]['overlap']
        threshold = max(1, top_overlap - 1)
        relevant_docs = [d['parent_doc_id'] for d in dedup if d['overlap'] >= threshold][:5]
        
    q['relevant_doc_ids'] = relevant_docs
    new_qrels.append(q)
    print(f"  {q['query_id']}: {len(relevant_docs)} docs (top overlap: {dedup[0]['overlap'] if dedup else 0})")

# Save expanded qrels
os.makedirs(os.path.dirname(QRELS_PATH), exist_ok=True)
with open(QRELS_PATH, 'w') as f:
    json.dump(new_qrels, f, indent=2)

print(f"\nGenerated {len(new_qrels)} queries in {QRELS_PATH}")
print(f"Avg docs per query: {sum(len(q['relevant_doc_ids']) for q in new_qrels) / len(new_qrels):.2f}")

# Temporal type distribution
type_counts = Counter(q['temporal_type'] for q in new_qrels)
print("Temporal type distribution:")
for t, c in type_counts.items():
    print(f"  {t}: {c}")

# Year coverage
years_covered = set()
for q in new_qrels:
    intent = parser.parse(q['query_text'])
    if intent.get('start_year'):
        years_covered.add(intent['start_year'])
    if intent.get('end_year') and intent.get('end_year') != intent.get('start_year'):
        years_covered.add(intent['end_year'])
print(f"Years explicitly referenced: {sorted(years_covered)}")