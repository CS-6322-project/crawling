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
import timeit
import numpy as np

nltk.download('stopwords', quiet=True)

# ==== CONFIG ====
INVERTED_INDEX_FILE = 'project_output_test/5k_result/inverted_index.json'
TF_IDF_FILE = 'project_output_test/5k_result/tf_idf_scores.json'
URL_MAP_FILE = 'test_data/test_url_ids.jsonl'
PAGERANK_FILE = 'project_output_test/5k_result/pagerank_scores.json'

# ==== INIT ====
app = Flask(__name__)
CORS(app)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ==== LOAD FILES ====
HITS_FILE = 'project_output_test/5k_result/hits_scores.json'
with open(HITS_FILE, 'r') as f:
    hits_data = json.load(f)
    hubs = hits_data.get('hubs', {})
    authorities = hits_data.get('authorities', {})

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
    
with open(PAGERANK_FILE) as f:
    pagerank_scores = json.load(f)
    
# Load doc_term_freq scores (X2)
with open(os.path.join(DATA_FOLDER, 'doc_term_freq.json'), 'r') as f:
    doc_term_freq = json.load(f)

with open(os.path.join(DATA_FOLDER, 'doc_term_pos.json'), 'r') as f:
    doc_term_pos = json.load(f)

# ==== RANKING LOGIC ====
def rank_documents(query, top_k=None):
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
            doc_vector = tf_idf.get(doc_id, {})
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
def build_association_cluster_expansion(query, data_folder, max_docs=500, top_k=None):

    start = timeit.default_timer()
    ###CHECK THIS, IF I USE ALL DOCS, IT TAKES A LOT OF TIME###
    ### I KEEP THIS To 500 TO MINIMIZE THE TIME BUT IT SHOULD BE ALL DOCS###
    if query not in inverted_index:
        return {
            "query": query,
            "results": []
        }
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
def build_scalar_cluster_expansion(query="mona", tf_idf=None, ps=None, stop_words=None, doc_ids=None, *, top_k=None):
    if ps is None:
        ps = PorterStemmer()
    if stop_words is None:
        stop_words = set(stopwords.words("english"))
    start = timeit.default_timer()

    ###CHECK THIS, IF I USE ALL DOCS, IT TAKES A LOT OF TIME###
    ### I KEEP THIS To 500 TO MINIMIZE THE TIME BUT IT SHOULD BE ALL DOCS###
    if query not in inverted_index:
        return {
            "query": query,
            "results": []
        }
    doc_list = doc_ids if doc_ids else inverted_index[query][0:25]
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

    scaler_cluster_result = {}
    for term in unique_tokens:
        if term != query:
            s_uv = np.dot(SM[token_mapping[term]], SM[token_mapping[query]])
            s_uv = s_uv / (np.linalg.norm(SM[token_mapping[term]]) * np.linalg.norm(SM[token_mapping[query]]))
            scaler_cluster_result[term] = s_uv
            
    scaler_cluster_result = sorted(scaler_cluster_result.items(), key=lambda x: x[1], reverse=True)

    print("Scaler Cluster Result:")
    for term in scaler_cluster_result[0:5]:
        print(term[0], term[1])
        
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    top_k = int(top_k)

    return [{"query": query, "term": term, "score": score} for term, score in scaler_cluster_result[:top_k]]

# Query expansion for METRIC CLUSTERING
def build_metric_cluster_expansion(query, tf_idf, ps, stop_words, *, top_k=None):

    start = timeit.default_timer()
    print("Loading indexes and ranking data...")

    ###CHECK THIS, IF I USE ALL DOCS, IT TAKES A LOT OF TIME###
    ### I KEEP THIS To 500 TO MINIMIZE THE TIME BUT IT SHOULD BE ALL DOCS###
    if query not in inverted_index:
        return {
            "query": query,
            "results": []
        }
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
    top_k = int(top_k)

    return [{"query": query, "term": term, "score": score} for term, score in sorted_result[:top_k]] 

# Search Engine Ro
def rocchio_expansion(query_text, rel_docs, nonrel_docs, alpha=1, beta=0.75, gamma=0.15):
        orig = calculate_query_vector(preprocess_query(query_text))
        new_query = {t: w * alpha for t, w in orig.items()}
        for doc in rel_docs:
            for t, w in tf_idf.get(doc, {}).items():
                new_query[t] = new_query.get(t, 0) + beta * w / len(rel_docs)
        for doc in nonrel_docs:
            for t, w in tf_idf.get(doc, {}).items():
                new_query[t] = new_query.get(t, 0) - gamma * w / len(nonrel_docs)
        return [t for t, w in sorted(new_query.items(), key=lambda x: x[1], reverse=True) if w > 0]

# Search engine
def preprocess_query(text):
        return [ps.stem(w) for w in re.findall(r'\w+', text.lower()) if w not in stop_words]

def get_matching_docs(query_tokens):
        docs = set()
        for token in query_tokens:
            docs.update(inverted_index.get(token, []))
        return docs

def calculate_query_vector(query_tokens):
        query_tf = Counter(query_tokens)
        query_vector = {}
        N = len(tf_idf)

        for token, freq in query_tf.items():
            tf = 1 + math.log(freq)
            df = len(set(inverted_index.get(token, [])))
            idf = math.log(N / (1 + df)) if df else 0
            query_vector[token] = tf * idf
        return query_vector

def calculate_cosine_similarity(query_vec, doc_id):
        dot, q_mag, d_mag = 0.0, 0.0, 0.0
        doc_vec = tf_idf.get(doc_id, {})
        for term, q_w in query_vec.items():
            d_w = doc_vec.get(term, 0.0)
            dot += q_w * d_w
        q_mag = math.sqrt(sum(w ** 2 for w in query_vec.values()))
        d_mag = math.sqrt(sum(w ** 2 for w in doc_vec.values()))
        return dot / (q_mag * d_mag) if q_mag and d_mag else 0.0

def expand_query(method, query, relevant=None, non_relevant=None):
        if method == "rocchio":
            return rocchio_expansion(query, relevant or [], non_relevant or [])
        elif method == "association":
            return build_association_cluster_expansion(query, tf_idf, inverted_index, ps, stop_words)
        elif method == "metric":
            return build_metric_cluster_expansion(query, tf_idf, ps, stop_words)
        elif method == "scalar":
            return build_scalar_cluster_expansion(query, tf_idf, ps, stop_words)
        else:
            return preprocess_query(query)

def rocchio_expansion(query_text, rel_docs, nonrel_docs, alpha=1, beta=0.75, gamma=0.15):
        orig = calculate_query_vector(preprocess_query(query_text))
        new_query = {t: w * alpha for t, w in orig.items()}
        for doc in rel_docs:
            for t, w in tf_idf.get(doc, {}).items():
                new_query[t] = new_query.get(t, 0) + beta * w / len(rel_docs)
        for doc in nonrel_docs:
            for t, w in tf_idf.get(doc, {}).items():
                new_query[t] = new_query.get(t, 0) - gamma * w / len(nonrel_docs)
        return [t for t, w in sorted(new_query.items(), key=lambda x: x[1], reverse=True) if w > 0]

def rank_results(docs, query_vec, method="combined"):
        scores = {}
        for doc in docs:
            tfidf_score = calculate_cosine_similarity(query_vec, doc)
            pr = pagerank_scores.get(url_map.get(doc, ""), 0)
            hit = authorities.get(url_map.get(doc, ""), 0)

            if method == "tfidf":
                scores[doc] = tfidf_score
            elif method == "pagerank":
                scores[doc] = pr
            elif method == "hits":
                scores[doc] = hit
            else:  # combined
                scores[doc] = tfidf_score * 0.6 + pr * 0.3 + hit * 0.1

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def run_search(query, rank_method="combined", expand_method=None, rel=None, nonrel=None, max_results=None):
        if expand_method:
            expanded = expand_query(expand_method, query, rel, nonrel)
            if isinstance(expanded[0], dict):
                tokens = [t["term"] for t in expanded]
            else:
                tokens = expanded
        else:
            tokens = preprocess_query(query)
        docs = get_matching_docs(tokens)
        qvec = calculate_query_vector(tokens)
        ranked = rank_results(docs, qvec, rank_method)
        return [{"doc_id": doc, "url": url_map.get(doc, "N/A"), "score": round(score, 4)} for doc, score in ranked[:max_results]]

# ==== ROUTES ====

@app.route('/search', methods=['GET'])
def search_route():
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
    results = run_search(query, rank_method="pagerank")
    return jsonify(results)

@app.route('/hits', methods=['GET'])
def hits_endpoint():
    query = request.args.get('query', "")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    results = run_search(query, rank_method="hits")
    return jsonify(results)

@app.route('/association', methods=['GET'])
def association_endpoint():
    query = request.args.get('query', "")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    results = rank_documents(query)
    doc_ids = [r['doc_id'] for r in results]
    expanded_terms = build_association_cluster_expansion(query, doc_ids)
    return jsonify({'expanded_terms': expanded_terms, "original_results": results })

@app.route('/metric', methods=['GET'])
def metric_endpoint():
    query = request.args.get('query', "")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    results = run_search(query, expand_method="metric")
    doc_ids = [r['doc_id'] for r in results]
    expanded_terms = build_metric_cluster_expansion(query, tf_idf, ps, stop_words)
    return jsonify({'expanded_terms': expanded_terms, "original_results": results })

@app.route('/scalar', methods=['GET'])
def scalar_endpoint():
    query = request.args.get('query', "")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    results = run_search(query, expand_method="scalar")
    doc_ids = [r['doc_id'] for r in results]
    expanded_terms = build_scalar_cluster_expansion(query, doc_ids)
    return jsonify({'expanded_terms': expanded_terms, "original_results": results })

@app.route('/rocchio', methods=['POST'])
def rocchio():
    data = request.get_json()
    query = data.get('query', "")
    relevant = data.get('relevant', [])
    non_relevant = data.get('non_relevant', [])

    if not query:
        return jsonify({"error": "Query is required"}), 400

    results = run_search(
        query_text=query,
        expand_method="rocchio",
        relevant_docs=relevant,
        non_relevant_docs=non_relevant
    )
    return jsonify(results)

# @app.route('/rocchio-test', methods=['GET'])
# def rocchio_test():
#     query = request.args.get('query', "")
#     if not query:
#         return jsonify({"error": "Query is required"}), 400

#     # 🔧 Hardcoded doc IDs for quick testing (you can replace them)
#     relevant = ["123", "456"]
#     non_relevant = ["789"]

#     results = run_search(
#         query=query,
#         expand_method="rocchio",
#         rel=relevant,
#         nonrel=non_relevant
#     )
#     return jsonify(results)


# ==== MAIN ====
if __name__ == '__main__':
    app.run(debug=True)
