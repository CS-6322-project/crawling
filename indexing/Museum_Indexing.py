import os
import json
import re
import math
import networkx as nx
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

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
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
url_map = {}
all_docs = set()
term_doc_freq = Counter()

# Load URL mapping
with open(URL_MAP_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            url_map[str(data['url_id'])] = data['url']

# Read JSON records and build index
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

            doc_id = str(record.get('url_id'))
            content = record.get('text') or record.get('content') or ''
            outlinks = record.get('outlinks', [])
            if not doc_id or not content.strip():
                continue

            all_docs.add(doc_id)
            count += 1
            if count > 100000:
                break

            words = re.findall(r'\w+', content.lower())
            seen_terms = set()
            for word in words:
                if word not in stop_words:
                    lemma = lemmatizer.lemmatize(word)
                    inverted_index[lemma].append(doc_id)
                    doc_term_freq[doc_id][lemma] += 1
                    seen_terms.add(lemma)

            for term in seen_terms:
                term_doc_freq[term] += 1

            source_url = url_map.get(doc_id)
            if source_url:
                for out_url in outlinks:
                    graph.add_edge(source_url, out_url)

# Remove the most frequent term globally (optional)
top_term = term_doc_freq.most_common(1)[0][0]
if top_term in inverted_index:
    del inverted_index[top_term]
for doc in doc_term_freq:
    doc_term_freq[doc].pop(top_term, None)

# TF-IDF Calculation
N = len(all_docs)
idf = {}

for term in inverted_index:
    inverted_index[term] = list(set(inverted_index[term]))
    df = len(inverted_index[term])
    idf[term] = math.log(N / df)

for doc in doc_term_freq:
    for term in doc_term_freq[doc]:
        tf = 1 + math.log(doc_term_freq[doc][term])
        tf_idf[doc][term] = tf * idf[term]

for doc in doc_term_freq:
    norm = math.sqrt(sum(tf_idf[doc][term] ** 2 for term in doc_term_freq[doc]))
    for term in doc_term_freq[doc]:
        tf_idf[doc][term] = round(tf_idf[doc][term] / norm, 5)

# PageRank and HITS
pagerank_scores = nx.pagerank(graph)
hubs, authorities = nx.hits(graph, max_iter=50)

# Save output
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

print("✅ Indexing & ranking completed.")
