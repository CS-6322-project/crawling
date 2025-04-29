from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import math
import re
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

# ==== CONFIG ====
INVERTED_INDEX_FILE = 'project_output/inverted_index.json'
TF_IDF_FILE = 'project_output/tf_idf_scores.json'
URL_MAP_FILE = 'output_300k/url_ids.jsonl'
PAGERANK_FILE = 'project_output/pagerank_scores.json'

# ==== INIT ====
app = Flask(__name__)
CORS(app)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ==== LOAD FILES ====
with open(INVERTED_INDEX_FILE) as f:
    inverted_index = json.load(f)

with open(TF_IDF_FILE) as f:
    tf_idf_scores = json.load(f)

with open(PAGERANK_FILE) as f:
    pagerank_scores = json.load(f)

url_map = {}
with open(URL_MAP_FILE, 'r') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            url_map[str(data['url_id'])] = data['url']

# ==== RANKING LOGIC ====
def rank_documents(query, top_k=10):
    query_terms = re.findall(r'\w+', query.lower())
    query_terms = [ps.stem(w) for w in query_terms if w not in stop_words]
    print(f"Processed query terms: {query_terms}")

    terms_cnt = Counter(query_terms)
    s = sum([terms_cnt[t]**2 for t in terms_cnt])
    s = math.sqrt(s)
    for t in terms_cnt:
        terms_cnt[t] /= s

    doc_scores = {}
    for term in terms_cnt:
        if term not in inverted_index:
            continue
        for doc_id in inverted_index[term]:
            doc_vector = tf_idf_scores.get(doc_id, {})
            score = sum(doc_vector.get(t, 0) * terms_cnt[t] for t in query_terms)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score

    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for doc_id, score in ranked:
        results.append({
            "doc_id": doc_id,
            "url": url_map.get(doc_id, "URL not found"),
            "score": round(score, 5)
        })

    return results

# ==== ROUTES ====
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', "")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    print(f"Received query: {query}")
    results = rank_documents(query)
    return jsonify(results)

@app.route('/rocchio', methods=['POST'])
def rocchio():
    data = request.get_json()
    relevant = data.get('relevant', [])
    non_relevant = data.get('non_relevant', [])
    query = data.get('query', "")

    expanded_terms = ["expanded", "dummy", "terms"]
    return jsonify({"query": query, "expanded_terms": expanded_terms})

# ==== DUMMY ENDPOINTS ====
@app.route('/pagerank', methods=['GET'])
def pagerank_endpoint():
    return jsonify({"method": "PageRank", "status": "dummy response"})

@app.route('/hits', methods=['GET'])
def hits_endpoint():
    return jsonify({"method": "HITS", "status": "dummy response"})

@app.route('/association', methods=['GET'])
def association_endpoint():
    return jsonify({"method": "Association Clustering", "status": "dummy response"})

@app.route('/metric', methods=['GET'])
def metric_endpoint():
    return jsonify({"method": "Metric Clustering", "status": "dummy response"})

@app.route('/scalar', methods=['GET'])
def scalar_endpoint():
    return jsonify({"method": "Scalar Clustering", "status": "dummy response"})

# ==== MAIN ====
if __name__ == '__main__':
    app.run(debug=True)