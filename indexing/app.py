from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
import math
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import sys
import os

from search_engine import SearchEngine

nltk.download('stopwords', quiet=True)

# ==== CONFIG ====
INVERTED_INDEX_FILE = 'project_output_test/5k_result/inverted_index.json'
TF_IDF_FILE = 'project_output_test/5k_result/tf_idf_scores.json'
URL_MAP_FILE = 'output_300k/url_ids.jsonl'
PAGERANK_FILE = 'project_output_test/5k_result/pagerank_scores.json'

# ==== INIT ====
app = Flask(__name__)
CORS(app)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
search_engine = SearchEngine()

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

DATA_FOLDER = "project_output_test/5k_result/"

# Load inverted index (X1)
with open(os.path.join(DATA_FOLDER, 'inverted_index.json'), 'r') as f:
    inverted_index = json.load(f)

# Load TF-IDF scores (X2)
with open(os.path.join(DATA_FOLDER, 'tf_idf_scores.json'), 'r') as f:
    tf_idf = json.load(f)
    
    
# Load doc_term_freq scores (X2)
with open(os.path.join(DATA_FOLDER, 'doc_term_freq.json'), 'r') as f:
    doc_term_freq = json.load(f)

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

# Query expansion script for ASSOCIATION CLUSTER
def build_association_cluster_expansion(query, data_folder, max_docs=500, top_k=5):

    start = timeit.default_timer()
    query = "mona"
    ###CHECK THIS, IF I USE ALL DOCS, IT TAKES A LOT OF TIME###
    ### I KEEP THIS To 500 TO MINIMIZE THE TIME BUT IT SHOULD BE ALL DOCS###
    doc_list = inverted_index[query][0:500]
    doc_count = len(doc_list)

    all_tokens = []
    for doc_id in doc_list:
        all_tokens.extend(list(doc_term_freq[doc_id].keys()))
        
    unique_tokens = set(all_tokens)
    association_matrix = {token: [0] * doc_count for token in unique_tokens}

    stop = timeit.default_timer()
    print('Time: ', stop - start)  

    cnt = 0
    for doc_id in doc_list:
        tokens = doc_term_freq[doc_id].keys()
        for token in tokens:
            association_matrix[token][cnt] += doc_term_freq[doc_id][token]
        cnt += 1
        
    stop = timeit.default_timer()
    print('Time: ', stop - start)  

    association_cluster_result = {}
    for term in unique_tokens:
        if term != query:
            c_uv =  sum([x * y for x, y in zip(association_matrix[term], association_matrix[query])])
            c_uu =  sum([x * y for x, y in zip(association_matrix[query], association_matrix[query])])
            c_vv =  sum([x * y for x, y in zip(association_matrix[term], association_matrix[term])])
            association_cluster_result[term] = c_uv / (c_uu + c_uv + c_vv)

    print("Association Cluster Result:")
    top_results = sorted(association_cluster_result.items(), key=lambda x: x[1], reverse=True)[:top_k]
    for term, score in sorted(association_cluster_result.items(), key=lambda x: x[1], reverse=True)[0:5]:
        print(f"Query: {query}, Term: {term}, Score: {score}")
        
    stop = timeit.default_timer()
    print('Time: ', stop - start)  

    return {
        "query": query,
        "results": [{"term": term, "score": round(score, 5)} for term, score in top_results]
    }

# Query expansion script for Scaler Clustering
def build_scalar_cluster_expansion(query="mona", data_folder=DATA_FOLDER, max_docs=25, top_k=5):
    start = timeit.default_timer()
    query = "mona"

    ###CHECK THIS, IF I USE ALL DOCS, IT TAKES A LOT OF TIME###
    ### I KEEP THIS To 500 TO MINIMIZE THE TIME BUT IT SHOULD BE ALL DOCS###
    doc_list = inverted_index[query][0:25]
    doc_count = len(doc_list)

    all_tokens = []
    for doc_id in doc_list:
        all_tokens.extend(list(doc_term_freq[doc_id].keys()))
        
    unique_tokens = list(set(all_tokens))
    association_matrix = {token: [0] * doc_count for token in unique_tokens}

    stop = timeit.default_timer()
    print('Time: ', stop - start)  

    token_mapping = {unique_tokens[i]:i for i in range(len(unique_tokens))}
    idx_mapping = {i:unique_tokens[i] for i in range(len(unique_tokens))}
    CM = np.zeros((len(unique_tokens), doc_count), dtype=int)

    cnt = 0
    for doc_id in doc_list:
        tokens = doc_term_freq[doc_id].keys()
        for token in tokens:
            CM[token_mapping[token]][cnt] += doc_term_freq[doc_id][token]
        cnt += 1
        
    SM = np.matmul(CM, CM.T)
        
    stop = timeit.default_timer()
    print('Time: ', stop - start)  

    query = "gogh"
    scaler_cluster_result = {}
    for term in unique_tokens:
        if term != query:
            s_uv = np.dot(SM[token_mapping[term]], SM[token_mapping[query]])
            s_uv = s_uv / (np.linalg.norm(SM[token_mapping[term]]) * np.linalg.norm(SM[token_mapping[query]]))
            scaler_cluster_result[term] = s_uv
            
    scaler_cluster_result = sorted(scaler_cluster_result.items(), key=lambda x: x[1], reverse=True)
    sorted_results = sorted(scaler_cluster_result.items(), key=lambda x: x[1], reverse=True)

    print("Scaler Cluster Result:")
    for term in scaler_cluster_result[0:5]:
        print(term[0], term[1])
        
    stop = timeit.default_timer()
    print('Time: ', stop - start) 

    return [{"query": query, "term": term, "score": score} for term, score in sorted_results[:top_k]]

# Query expansion for METRIC CLUSTERING
def build_metric_cluster_expansion(query="mona", data_folder=DATA_FOLDER, max_docs=25, top_k=5):

    start = timeit.default_timer()
    print("Loading indexes and ranking data...")
    query = "mona"

    ###CHECK THIS, IF I USE ALL DOCS, IT TAKES A LOT OF TIME###
    ### I KEEP THIS To 500 TO MINIMIZE THE TIME BUT IT SHOULD BE ALL DOCS###
    doc_list = inverted_index[query][0:25]
    doc_count = len(doc_list)

    all_tokens = []
    for doc_id in doc_list:
        all_tokens.extend(list(doc_term_freq[doc_id].keys()))
        
    unique_tokens = set(all_tokens)

    stop = timeit.default_timer()
    print('Time: ', stop - start)  

    metric_cluster_result = {}
    score = 0
    for term in unique_tokens:
        for doc_id in doc_list:     
            if query in doc_term_pos[doc_id]:       
                for i in doc_term_pos[doc_id][query]:
                    if term in doc_term_pos[doc_id] and term != query:
                        for j in doc_term_pos[doc_id][term]:
                            if term not in metric_cluster_result:
                                metric_cluster_result[term] = 0
                            metric_cluster_result[term] += 1 / abs(i - j)
                                                  
    stop = timeit.default_timer()
    sorted_result = sorted(metric_cluster_result.items(), key=lambda x: x[1], reverse=True)
    print('Time: ', stop - start)  

    # DIDN't NORMALIZE BECAUSE we only stems...

    sorted_metric_cluster_result = sorted(metric_cluster_result.items(), key=lambda x: x[1], reverse=True)
    print("Metric Cluster Result:")
    for term, score in sorted_metric_cluster_result[0:5]:
        print(f"Term: {term}, Score: {score}")

    stop = timeit.default_timer()
    print('Time: ', stop - start) 

    return [{"query": query, "term": term, "score": score} for term, score in sorted_result[:top_k]] 

# ==== ROUTES ====

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', "")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    print(f"Received query: {query}")
    results = rank_documents(query)
    return jsonify(results)

@app.route('/pagerank', methods=['GET'])
def pagerank_endpoint():
    query = request.args.get('query', "")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    results, _ = search_engine.search(query, ranking_method="pagerank")
    return jsonify(results)

@app.route('/hits', methods=['GET'])
def hits_endpoint():
    query = request.args.get('query', "")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    results, _ = search_engine.search(query, ranking_method="hits")
    return jsonify(results)

@app.route('/association', methods=['GET'])
def association_endpoint():
    query = request.args.get('query', "")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    results, _ = search_engine.search(query, expansion_method="association")
    return jsonify(results)

@app.route('/metric', methods=['GET'])
def metric_endpoint():
    query = request.args.get('query', "")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    results, _ = search_engine.search(query, expansion_method="metric")
    return jsonify(results)

@app.route('/scalar', methods=['GET'])
def scalar_endpoint():
    query = request.args.get('query', "")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    results, _ = search_engine.search(query, expansion_method="scalar")
    return jsonify(results)

@app.route('/rocchio', methods=['POST'])
def rocchio():
    data = request.get_json()
    query = data.get('query', "")
    relevant = data.get('relevant', [])
    non_relevant = data.get('non_relevant', [])

    if not query:
        return jsonify({"error": "Query is required"}), 400

    results, _ = search_engine.search(
        query_text=query,
        expansion_method="rocchio",
        relevant_docs=relevant,
        non_relevant_docs=non_relevant
    )
    return jsonify(results)

# ==== MAIN ====
if __name__ == '__main__':
    app.run(debug=True)
