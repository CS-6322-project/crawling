import os
import json
import re
import math
import networkx as nx
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')

# Configuration
OUTPUT_JSON = 'output_300k/output_300k.json'
URL_MAP_FILE = 'output_300k/url_ids.jsonl'
OUTPUT_FOLDER = 'project_output_5k'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialization
inverted_index = defaultdict(list)
doc_term_freq = defaultdict(lambda: defaultdict(int))
tf_idf = defaultdict(lambda: defaultdict(float))
graph = nx.DiGraph()
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
url_map = {}
all_docs = set()

# Loading URL mapping
print("Loading URL map from url_ids.jsonl...")
with open(URL_MAP_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            url_map[str(data['url_id'])] = data['url']

# Build index and graphs
print("Parsing records and building inverted index & web graph...")
buffer = ""
inside_object = False
count = 0
with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line in ('[', ']', '],'):
            continue

        if line.startswith('{'):
            buffer = line
            inside_object = True
        elif inside_object:
            buffer += line

        if line.endswith('},') or line.endswith('}'):
            try:
                record = json.loads(buffer.rstrip(',').strip())
            except json.JSONDecodeError:
                buffer = ""
                continue
            buffer = ""

            #print(f"Processing record: {str(record.get('doc_id'))}")
            doc_id = str(record.get('url_id'))
            if not doc_id:
                print(f"Skipping record due to missing url_id: {record}")
                continue

            content = record.get('text') or record.get('content') or ''
            outlinks = record.get('outlinks', [])

            if not content.strip():
                continue

            all_docs.add(doc_id)
            count += 1
            
            if count > 100000:
                break
            
            if count % 1000 == 0:
                print(f"Processed {count} documents...")

            # Tokenize and build inverted index
            words = re.findall(r'\w+', content.lower())
            for word in words:
                if word not in stop_words:
                    stemmed = ps.stem(word)
                    #if doc_id not in inverted_index[stemmed]:
                    inverted_index[stemmed].append(doc_id)
                    doc_term_freq[doc_id][stemmed] += 1
                
            # Build web graph
            source_url = url_map.get(doc_id)
            if source_url:
                for out_url in outlinks:
                    #if out_url in url_map.values():
                    graph.add_edge(source_url, out_url)

print(f"Documents processed: {len(all_docs)}")
print(f"Graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")

# Calculating TF-IDF
print("Calculating TF-IDF scores...")
N = len(all_docs)
idf = {}

for term in inverted_index:
    inverted_index[term] = list(set(inverted_index[term]))
    df = len(inverted_index[term])
    #idf[term] = math.log(N / (1 + df))
    idf[term] = math.log(N / df)

for doc in doc_term_freq:
    for term in doc_term_freq[doc]:
        tf = 1 + math.log(doc_term_freq[doc][term])
        tf_idf[doc][term] = tf * idf[term]
        
for doc in doc_term_freq:
    s = 0
    for term in doc_term_freq[doc]:
        s += tf_idf[doc][term] ** 2 
    s = math.sqrt(s)
    for term in doc_term_freq[doc]:
        tf_idf[doc][term] = round(tf_idf[doc][term] / s, 5)

        

# =Page Rank and HITS
print("Calculating PageRank & HITS...")
pagerank_scores = nx.pagerank(graph)
hubs, authorities = nx.hits(graph, max_iter=50)

# Output
print("Saving output files...")

with open(os.path.join(OUTPUT_FOLDER, 'inverted_index.json'), 'w') as f:
    json.dump(inverted_index, f)

with open(os.path.join(OUTPUT_FOLDER, 'tf_idf_scores.json'), 'w') as f:
    json.dump(tf_idf, f)

edges_list = [{"source": u, "target": v} for u, v in graph.edges()]
with open(os.path.join(OUTPUT_FOLDER, 'web_graph_edges.json'), 'w') as f:
    json.dump(edges_list, f)

with open(os.path.join(OUTPUT_FOLDER, 'pagerank_scores.json'), 'w') as f:
    json.dump(pagerank_scores, f)

with open(os.path.join(OUTPUT_FOLDER, 'hits_scores.json'), 'w') as f:
    json.dump({"hubs": hubs, "authorities": authorities}, f)

print("Indexing & Ranking complete. All outputs saved in 'project_output/' folder.")
