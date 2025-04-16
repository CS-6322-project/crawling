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

# CONFIG
OUTPUT_JSON = 'test_data/test_pages.json'
URL_MAP_FILE = 'test_data/test_url_ids.jsonl'
OUTPUT_FOLDER = 'project_output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# INITIALIZATION
inverted_index = defaultdict(list)
doc_term_freq = defaultdict(lambda: defaultdict(int))
tf_idf = defaultdict(lambda: defaultdict(float))
graph = nx.DiGraph()
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
url_map = {}
all_docs = set()

# Load URL map
print("Loading URL map from url_ids.jsonl...")
with open(URL_MAP_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            url_map[str(data['url_id'])] = data['url']

# Parse test records
print("Parsing records and building inverted index & web graph...")
with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
    records = json.load(f)

    for record in records:
        doc_id = str(record.get('url_id'))
        content = record.get('text', '')
        outlinks = record.get('outlinks', [])
        if not doc_id or not content.strip():
            continue

        all_docs.add(doc_id)

        words = re.findall(r'\w+', content.lower())
        for word in words:
            if word not in stop_words:
                stemmed = ps.stem(word)
                inverted_index[stemmed].append(doc_id)
                doc_term_freq[doc_id][stemmed] += 1

        source_url = url_map.get(doc_id)
        if source_url:
            for out_url in outlinks:
                if out_url in url_map.values():
                    graph.add_edge(source_url, out_url)

print(f"Documents processed: {len(all_docs)}")
print(f"Graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")

# TF-IDF
print("Calculating TF-IDF scores...")
N = len(all_docs)
idf = {}
for term in inverted_index:
    df = len(set(inverted_index[term]))
    idf[term] = math.log(N / (1 + df))
for doc in doc_term_freq:
    for term in doc_term_freq[doc]:
        tf = 1 + math.log(doc_term_freq[doc][term])
        tf_idf[doc][term] = tf * idf[term]

# PageRank & HITS
print("Calculating PageRank & HITS...")
pagerank_scores = nx.pagerank(graph)
hubs, authorities = nx.hits(graph, max_iter=50)

# Save files
print("Saving output files...")
with open(os.path.join(OUTPUT_FOLDER, 'inverted_index.json'), 'w') as f:
    json.dump(inverted_index, f)

with open(os.path.join(OUTPUT_FOLDER, 'tf_idf_scores.json'), 'w') as f:
    json.dump(tf_idf, f)

with open(os.path.join(OUTPUT_FOLDER, 'web_graph_edges.json'), 'w') as f:
    json.dump([{"source": u, "target": v} for u, v in graph.edges()], f)

with open(os.path.join(OUTPUT_FOLDER, 'pagerank_scores.json'), 'w') as f:
    json.dump(pagerank_scores, f)

with open(os.path.join(OUTPUT_FOLDER, 'hits_scores.json'), 'w') as f:
    json.dump({"hubs": hubs, "authorities": authorities}, f)

print("✅ Mini Indexing & Ranking complete. Outputs saved in 'project_output/' folder.")
