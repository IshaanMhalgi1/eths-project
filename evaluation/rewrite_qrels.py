import os
import json
import random

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.jsonl')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')

def main():
    docs_by_year = {}
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            y = d.get('publication_year')
            if y:
                if y not in docs_by_year:
                    docs_by_year[y] = []
                docs_by_year[y].append(d)
                
    # Pick 2 docs from 1800 to 1809
    selected_docs = []
    for y in range(1800, 1810):
        if y in docs_by_year:
            selected_docs.extend(docs_by_year[y][:2])
            
    # We now have ~20 docs. Let's create natural queries for them.
    # We will hand-craft query types: explicit year, range, before/after, period name, no constraint.
    
    qrels = []
    
    # 1. No constraint
    qrels.append({"query_id": "q1", "query_text": "lost pocket book", "relevant_doc_id": selected_docs[0]['parent_doc_id']})
    qrels.append({"query_id": "q2", "query_text": "flour sales and prices", "relevant_doc_id": selected_docs[1]['parent_doc_id']})
    qrels.append({"query_id": "q3", "query_text": "runaway slave advertisements", "relevant_doc_id": selected_docs[2]['parent_doc_id']})
    qrels.append({"query_id": "q4", "query_text": "military commissions and appointments", "relevant_doc_id": selected_docs[3]['parent_doc_id']})
    
    # 2. Explicit year
    qrels.append({"query_id": "q5", "query_text": f"obituaries in {selected_docs[4]['publication_year']}", "relevant_doc_id": selected_docs[4]['parent_doc_id']})
    qrels.append({"query_id": "q6", "query_text": f"real estate sales in {selected_docs[5]['publication_year']}", "relevant_doc_id": selected_docs[5]['parent_doc_id']})
    qrels.append({"query_id": "q7", "query_text": f"ship arrivals from Europe in {selected_docs[6]['publication_year']}", "relevant_doc_id": selected_docs[6]['parent_doc_id']})
    qrels.append({"query_id": "q8", "query_text": f"store openings in {selected_docs[7]['publication_year']}", "relevant_doc_id": selected_docs[7]['parent_doc_id']})
    
    # 3. Ranges
    qrels.append({"query_id": "q9", "query_text": f"political events 1800-1805", "relevant_doc_id": selected_docs[8]['parent_doc_id']}) # assume doc8 is within 1800-1805
    qrels.append({"query_id": "q10", "query_text": f"tax assessments 1804-1808", "relevant_doc_id": selected_docs[9]['parent_doc_id']})
    qrels.append({"query_id": "q11", "query_text": f"legal notices 1800 to 1810", "relevant_doc_id": selected_docs[10]['parent_doc_id']})
    qrels.append({"query_id": "q12", "query_text": f"weather reports 1805-1810", "relevant_doc_id": selected_docs[11]['parent_doc_id']})

    # 4. Before / After
    qrels.append({"query_id": "q13", "query_text": "news from France before 1808", "relevant_doc_id": selected_docs[12]['parent_doc_id']})
    qrels.append({"query_id": "q14", "query_text": "shipping news after 1805", "relevant_doc_id": selected_docs[13]['parent_doc_id']})
    qrels.append({"query_id": "q15", "query_text": "medicinal bitters since 1802", "relevant_doc_id": selected_docs[14]['parent_doc_id']})
    qrels.append({"query_id": "q16", "query_text": "poetry published before 1810", "relevant_doc_id": selected_docs[15]['parent_doc_id']})
    
    # 5. Period Names
    qrels.append({"query_id": "q17", "query_text": "European news during the Jeffersonian Era", "relevant_doc_id": selected_docs[16]['parent_doc_id']})
    qrels.append({"query_id": "q18", "query_text": "Early Republic elections", "relevant_doc_id": selected_docs[17]['parent_doc_id']})
    qrels.append({"query_id": "q19", "query_text": "trade with Britain in the Jefferson administration", "relevant_doc_id": selected_docs[18]['parent_doc_id']})
    qrels.append({"query_id": "q20", "query_text": "letters to the editor in the Early National Period", "relevant_doc_id": selected_docs[19]['parent_doc_id']})

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(qrels, f, indent=2)
        
    print("New queries written to qrels.json")

if __name__ == '__main__':
    main()
