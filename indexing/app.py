from flask import Flask, request, jsonify
import json
import re
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from collections import Counter
nltk.download('stopwords')

# === CONFIG ===
INVERTED_INDEX_FILE = 'project_output_5k/inverted_index.json'
TF_IDF_FILE = 'project_output_5k/tf_idf_scores.json'
URL_MAP_FILE = 'output_300k/url_ids.jsonl'  # (same as Museum_Indexing path!)
PAGERANK_FILE = 'project_output/pagerank_scores.json'

# === LOAD FILES ===
print("Loading inverted index and tf-idf scores...")
with open(INVERTED_INDEX_FILE) as f:
    inverted_index = json.load(f)

with open(TF_IDF_FILE) as f:
    tf_idf_scores = json.load(f)

try:
    with open(PAGERANK_FILE) as f:
        pagerank_scores = json.load(f)
except:
    pagerank_scores = {}

print("Loading URL map...")
url_map = {}
with open(URL_MAP_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            url_map[str(data['url_id'])] = data['url']

# === SETUP ===
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# === DOCUMENT RANKING FUNCTION ===
def rank_documents(query, top_k=10):
    query_terms = re.findall(r'\w+', query.lower())
    query_terms = [ps.stem(w) for w in query_terms if w not in stop_words]
    print(f"Processed query terms: {query_terms}")
    
    terms_cnt = Counter(query_terms)
    s = 0
    for t in terms_cnt:
        s = terms_cnt[t] ** 2
    s = math.sqrt(s)
    for t in terms_cnt:
        terms_cnt[t] = terms_cnt[t] / s


    doc_scores = {}

    for term in terms_cnt:
        if term not in inverted_index:
            continue
        for doc_id in inverted_index[term]:
            doc_vector = tf_idf_scores.get(doc_id, {})
            score = sum(doc_vector.get(t, 0) * terms_cnt[t] for t in query_terms)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score

    # Optional: Boost using PageRank
    for doc_id in doc_scores:
        url = url_map.get(doc_id)
        if url and url in pagerank_scores:
            doc_scores[doc_id] *= (1 + pagerank_scores[url])

    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in ranked[:top_k]:
        results.append({
            "doc_id": doc_id,
            "url": url_map.get(doc_id, "Unknown"),
            "score": round(score, 5)
        })
    return results

# === FLASK APP ===
app = Flask(__name__)

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    print(f"Received query: {query}")
    results = rank_documents(query)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
